"""AND-OR proof hypergraph for the hybrid GNN/PLN search
A hypergraph rooted at the
original goal, whose nodes are subgoals (carrying both the GNN's prior
probability and the PLN-derived STV) and whose hyperedges are tactic
applications. Applying a tactic can spawn *several* subgoals that must
*all* be proven — that is the "hyper" part (an AND-edge), and alternative
tactics on the same goal are OR-alternatives.

Conventional choice and its trade-off:
this implementation keeps each subgoal node attached to exactly one parent
edge (a tree of AND-hyperedges, as in most neural-prover search trees — DeepHOL/
HOList, GPT-f). A fully general HTPS-style hypergraph additionally *merges*
syntactically-identical subgoals discovered via different branches into  one
shared, multi-parent node, which needs more involved backprop (a node can feed
several parents). That sharing is *not* implemented here — it is left as a
documented open item. What *is* implemented is cycle-breaking: a subgoal identical
to one of its own ancestors is treated as unprovable along that path (standard in tree-search
provers to avoid infinite regress), using the same concrete
(non-normalized) state fingerprint the translator recommends for exact
branch identity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from math import isfinite
from typing import Deque, Dict, List, Optional, Protocol, Set, Tuple

from maths_ai.data_models.proof_components import (
    LOG_PRIOR_TOLERANCE,
    STV,
    Goal,
    ProposedAction,
    TacticCandidate,
)
from maths_ai.data_models.proof_components import GoalState as GoalStateModel
from pantograph_async import PantographStateScope, TacticState

class NodeStatus:
    OPEN = "open"          # not yet expanded; eligible for the search frontier
    EXPANDED = "expanded"  # has ≥1 outgoing edge, outcome still undetermined
    SOLVED = "solved"      # ≥1 outgoing edge is fully solved (OR-success)
    DEAD = "dead"          # exhausted candidates and every edge is dead (AND-failure)


class EdgeStatus:
    PENDING = "pending"  # children still open/expanded; outcome undetermined
    SOLVED = "solved"    # no children, or every child is SOLVED (AND-success)
    DEAD = "dead"        # the tactic failed to apply, or some child is DEAD


class NodeClosureReason(str, Enum):
    NONE = "none"
    CANDIDATES_EXHAUSTED = "candidates_exhausted"
    WIDENING_BOUND = "widening_bound"
    NO_CANDIDATES = "no_candidates"
    DEPTH_LIMIT = "depth_limit"
    CYCLE = "cycle"
    ELABORATION_ERROR = "elaboration_error"
    EXTERNAL_ABORT = "external_abort"


class SearchEndReason(str, Enum):
    RUNNING = "running"
    ROOT_SOLVED = "root_solved"
    ROOT_DEAD = "root_dead"
    MAX_NODES = "max_nodes"
    NUM_SIMULATIONS = "num_simulations"
    DEADLINE = "deadline"
    FRONTIER_EMPTY = "frontier_empty"
    EXTERNAL_ABORT = "external_abort"


class BackupValidity(str, Enum):
    VALID = "valid"
    UNKNOWN = "unknown"


class BackupSource(str, Enum):
    NONE = "none"
    LEAN_STATUS = "lean_status"
    SEARCH_FAILURE = "search_failure"
    ROOT_EPISODE = "root_episode"
    VISIT_MEAN = "visit_mean"
    CRITIC_BOOTSTRAP = "critic_bootstrap"


@dataclass(frozen=True)
class BackupValue:
    value: Optional[float]
    validity: BackupValidity
    source: BackupSource

    def __post_init__(self) -> None:
        if self.validity == BackupValidity.UNKNOWN:
            if self.value is not None or self.source != BackupSource.NONE:
                raise ValueError("Unknown backups must have no numeric value or source.")
            return
        if self.value is None or not isfinite(self.value) or not 0.0 <= self.value <= 1.0:
            raise ValueError("Valid backups require a finite value in [0, 1].")
        if self.source == BackupSource.NONE:
            raise ValueError("Valid backups require provenance.")

    @classmethod
    def unknown(cls) -> "BackupValue":
        return cls(None, BackupValidity.UNKNOWN, BackupSource.NONE)

    @classmethod
    def valid(cls, value: float, source: BackupSource) -> "BackupValue":
        return cls(float(value), BackupValidity.VALID, source)


def _state_key(goal: Goal) -> str:
    """Stable identity including the normalized target and ordered local context."""
    return goal.state_fingerprint()


@dataclass
class ProofNode:
    id: int
    goal: Goal
    depth: int
    gnn_probability: float = 1.0
    stv: Optional[STV] = None
    status: str = NodeStatus.OPEN
    incoming_edge_id: Optional[int] = None
    outgoing_edge_ids: List[int] = field(default_factory=list)
    combined_rank: float = 0.0
    # The node's proposal source: rounds and raw draws spent so far. The source
    # stays open across rounds until a bound or a path limit closes it
    # (``closure_reason``); finishing one round does not close it.
    proposal_rounds: int = 0
    proposal_draws: int = 0
    # Widening capacity when the last round was scheduled; the next widening
    # round waits until the capacity has grown past it.
    round_capacity: int = 0
    # Completed simulations whose selection included this node, whether or not
    # they chose one of its edges; progressive widening grows with it.
    visits: int = 0
    # Checked execution outcome of each command tried at this node: the edge
    # it created, or None for a Lean rejection. Infrastructure failures are
    # never recorded. A later draw of the same command reuses the outcome.
    command_outcomes: Dict[tuple, Optional[int]] = field(default_factory=dict)
    note: Optional[str] = None
    closure_reason: NodeClosureReason = NodeClosureReason.NONE
    """Free-text annotation for terminal states the graph can't infer on its
    own — e.g. why a node was force-marked dead (cycle, depth limit, no
    candidates) — useful for inspecting/visualizing a finished search."""

    def __post_init__(self) -> None:
        self.combined_rank = self.local_score

    @property
    def exhausted(self) -> bool:
        """The proposal source is closed: no further actions will be proposed."""
        return self.closure_reason != NodeClosureReason.NONE

    @property
    def attempted_commands(self) -> int:
        """Distinct commands with a checked outcome at this node."""
        return len(self.command_outcomes)

    @property
    def local_score(self) -> float:
        """gnn_prob × STV(strength × confidence) — this node's own
        provability heuristic in isolation, before any of *its* children
        have been explored. Exactly ``RankedSubgoal.combined_rank`` for
        non-root nodes; for the root (no incoming tactic / no STV yet) both
        factors default to neutral so it is always explored first.
        """
        stv_score = self.stv.score if self.stv is not None else 1.0
        return self.gnn_probability * stv_score

    def as_state(self, tactic_path: List[str]) -> GoalStateModel:
        return GoalStateModel(goal=self.goal, depth=self.depth, tactic_path=tactic_path)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "expression": self.goal.expression,
            "depth": self.depth,
            "status": self.status,
            "gnn_probability": self.gnn_probability,
            "stv": None if self.stv is None else {"strength": self.stv.strength, "confidence": self.stv.confidence},
            "combined_rank": self.combined_rank,
            "note": self.note,
            "closure_reason": self.closure_reason.value,
        }


@dataclass
class EdgeVisitStats:
    """Per-edge MCTS visit statistics (HTPS-style).

    ``N`` counts completed simulations whose backup passed through this
    edge. ``N_v`` counts the subset with a valid numeric backup, and ``W``
    accumulates only those values, so ``W / N_v`` is the empirical action
    value. ``virtual_loss`` counts
    in-flight simulations that selected this edge but have not backed up
    yet — each one is treated as a pending loss (contributes 0 to ``W``
    while inflating the visit count), which steers concurrent simulations
    in the same batch toward different branches. The policy prior is not
    stored here: ``ProofHypergraph.puct_priors`` derives it from the edges'
    raw action log-priors whenever the node's selectable support changes.
    """

    N: int = 0
    N_v: int = 0
    W: float = 0.0
    virtual_loss: int = 0

    @property
    def Q(self) -> float:
        """Mean action value under virtual loss.

        An unvisited edge (no completed backups, no in-flight selections)
        returns the first-play-urgency prior 0.5 rather than 0.0, so a
        fresh edge is neither maximally attractive nor pre-emptively
        written off before its first simulation.
        """
        if self.N_v == 0:
            return 0.5
        return self.W / self.N_v


# Prior contract, recorded in saved search settings and reports.
PRIOR_SEMANTICS = "joint-action-v1"
PRIOR_NORMALIZATION = "viable-observed-support-v1"
PRIOR_ALIAS_POLICY = "unique-raw-action-sum-v1"


def logsumexp(values: List[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(value - top) for value in values))


@dataclass
class ProofHyperedge:
    """One executed command and every raw action observed to render it.

    ``raw_log_priors`` maps each distinct raw action id to its immutable
    log-prior. The edge's mass is their log-sum-exp: each distinct raw action
    counts once, however often it was drawn. This is mass over the raw actions
    this search observed, not the command's exact probability over every
    possible alias.
    """

    id: int
    source_id: int
    tactic: TacticCandidate
    child_ids: List[int] = field(default_factory=list)
    status: str = EdgeStatus.PENDING
    visit_stats: EdgeVisitStats = field(default_factory=EdgeVisitStats)
    raw_log_priors: Dict[tuple, float] = field(default_factory=dict)
    provider: str = ""

    @property
    def log_mass(self) -> float:
        return logsumexp(list(self.raw_log_priors.values()))

    def summary(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "tactic": self.tactic.tactic_name,
            "arguments": list(self.tactic.arguments),
            "tactic_confidence": self.tactic.probability,
            "child_ids": list(self.child_ids),
            "status": self.status,
            "prior": {
                "provider": self.provider,
                "raw_actions": len(self.raw_log_priors),
                "raw_log_priors": sorted(self.raw_log_priors.values(), reverse=True),
                "log_mass": self.log_mass,
            },
            "visits": {
                "N": self.visit_stats.N,
                "N_v": self.visit_stats.N_v,
                "W": self.visit_stats.W,
                "Q": self.visit_stats.Q,
                "virtual_loss": self.visit_stats.virtual_loss,
            },
        }


@dataclass(frozen=True)
class TacticOutcome:
    """Result of attempting to apply one tactic (with arguments) to a goal.

    ``subgoals`` holds the resulting proof obligations; an *empty* list with
    ``success=True`` is the "no-goal" terminal state from objective 1 — the
    tactic fully discharged the goal. ``subgoals`` is meaningless when
    ``success`` is False.
    """

    success: bool
    subgoals: List[Goal] = field(default_factory=list)
    error: Optional[str] = None


class TacticExecutor(Protocol):
    """Applies a predicted tactic to a goal state and reports the outcome.

    Turning a ``(tactic_name, arguments)`` pair into an actual Lean state
    transition needs a running Lean/Pantograph backend
    (``scope.apply_tactic(state, tactic)``); ``HybridReasoner``
    is built against ``PantographExecutor`` for that.
    ``NullTacticExecutor`` implements the same protocol for tests/tooling
    that don't have a Lean/Pantograph backend available.
    """

    async def apply(
        self, scope: PantographStateScope, state: TacticState, tactic: TacticCandidate
    ) -> TacticOutcome:
        ...


class NullTacticExecutor:
    """Stand-in executor for tests/tooling that don't have a Lean/Pantograph
    backend available: explicitly reports that no backend is configured.

    Every application is reported as *failed* rather than as a vacuous
    success (zero subgoals) — a vacuous success would be indistinguishable
    from a real QED and would silently fabricate proofs. Surfacing the
    failure keeps the missing piece visible instead of masking it.
    """

    async def apply(
        self, scope: PantographStateScope, state: TacticState, tactic: TacticCandidate
    ) -> TacticOutcome:
        rendered = " ".join([tactic.tactic_name, *(arg.rstrip(":") for arg in tactic.arguments)]).strip()
        target = state.goals[0].target.pp if state.goals else "<no goals>"
        return TacticOutcome(
            success=False,
            error=(
                f"no TacticExecutor configured — cannot apply '{rendered}' to "
                f"{target!r}; requires a Lean/Pantograph backend "
                "(see design report: tactic-application edge case)"
            ),
        )


class ProofHypergraph:
    """Root-at-goal AND-OR hypergraph, with bottom-up status/rank propagation.

    Build it exclusively through ``add_edge`` / ``mark_edge_dead`` /
    ``mark_node_exhausted`` — these keep ``status`` and ``combined_rank``
    consistent across the whole graph (objective 2: "when PLN ranks a
    subgoal, the GNN-derived score should be updated", here generalized to
    "...and that update should propagate to every ancestor").
    """

    def __init__(self, root_goal: Goal) -> None:
        self._next_node_id = 0
        self._next_edge_id = 0
        self.nodes: Dict[int, ProofNode] = {}
        self.edges: Dict[int, ProofHyperedge] = {}
        # Simulation selections that reached an open node with nothing to
        # propose or select yet (its widening capacity had not grown).
        self.deferred_selections = 0
        self.end_reason = SearchEndReason.RUNNING

        self.root_id = self._new_node(goal=root_goal, depth=0, gnn_probability=1.0, stv=None)

    # Construction
    def _new_node(
        self,
        *,
        goal: Goal,
        depth: int,
        gnn_probability: float,
        stv: Optional[STV],
        incoming_edge_id: Optional[int] = None,
    ) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        node = ProofNode(
            id=node_id,
            goal=goal,
            depth=depth,
            gnn_probability=gnn_probability,
            stv=stv,
            incoming_edge_id=incoming_edge_id,
        )
        self.nodes[node_id] = node
        return node_id

    def add_edge(
        self,
        source_id: int,
        action: ProposedAction,
        ranked_subgoals: List[Tuple[Goal, Optional[STV]]],
    ) -> ProofHyperedge:
        """Record the application of ``action`` from ``source_id`` producing the given
        ``(subgoal, stv)`` children (already scored by PLN and capped to the
        chosen top-k — see ``HybridReasoner.rank_subgoals``).

        ``stv=None`` is accepted for each child when PLN is disabled
        (``use_pln=False``): ``ProofNode.local_score`` already handles
        ``stv is None`` by returning ``gnn_probability × 1.0``, and
        ``potential()`` in ``pln_reward`` returns ``0.0``, so reward shaping
        vanishes automatically with no change to either module.

        An empty ``ranked_subgoals`` list models the "no-goal" terminal state
        (the tactic fully discharges ``source_id``'s goal): the edge is
        immediately SOLVED.

        Edge case — cycles: any subgoal whose concrete state matches one of
        ``source_id``'s own ancestors is created but force-marked DEAD with
        ``note="cycle"`` rather than re-explored, so the search can't loop
        forever rediscovering the same state.

        The edge starts with ``action``'s raw log-prior; ``observe_alias`` adds
        further raw actions that render the same command.
        """
        if not isinstance(action, ProposedAction):
            raise TypeError(f"add_edge needs a ProposedAction, got {type(action).__name__}.")
        tactic = action.candidate
        for eid in self.nodes[source_id].outgoing_edge_ids:
            if self.edges[eid].tactic.command == tactic.command:
                raise ValueError(f"Node {source_id} already has an edge for {tactic.command}.")
        edge_id = self._next_edge_id
        self._next_edge_id += 1
        depth = self.nodes[source_id].depth + 1
        ancestry = self._ancestor_state_keys(source_id)

        child_ids: List[int] = []
        for goal, stv in ranked_subgoals:
            child_id = self._new_node(
                goal=goal,
                depth=depth,
                gnn_probability=tactic.probability,
                stv=stv,
                incoming_edge_id=edge_id,
            )
            if _state_key(goal) in ancestry:
                child = self.nodes[child_id]
                child.status = NodeStatus.DEAD
                child.note = "cycle: identical to an ancestor goal"
                child.closure_reason = NodeClosureReason.CYCLE
            child_ids.append(child_id)

        edge = ProofHyperedge(
            id=edge_id,
            source_id=source_id,
            tactic=tactic,
            child_ids=child_ids,
            raw_log_priors={action.raw_action_id: float(action.log_prior)},
            provider=action.provider,
        )
        edge.status = self._derive_edge_status(edge)
        self.edges[edge_id] = edge
        self.nodes[source_id].outgoing_edge_ids.append(edge_id)

        self._propagate(source_id)
        return edge

    def observe_alias(self, edge_id: int, action: ProposedAction) -> bool:
        """Record another raw action that renders ``edge_id``'s command.

        Return True when it is new, which adds its probability to the edge's
        mass once. A raw action seen again must carry the same log-prior; a
        different value means the model, graph or provider changed under the
        search and is an error, not new mass.
        """
        edge = self.edges[edge_id]
        if action.command != edge.tactic.command:
            raise ValueError(f"{action.command} is not an alias of edge {edge_id}.")
        if action.provider != edge.provider:
            raise ValueError(
                f"Edge {edge_id} has priors from {edge.provider!r}, not {action.provider!r}."
            )
        known = edge.raw_log_priors.get(action.raw_action_id)
        if known is not None:
            if abs(known - action.log_prior) > 1e-4:
                raise ValueError(
                    f"Raw action {action.raw_action_id} changed its log-prior from "
                    f"{known} to {action.log_prior}."
                )
            return False
        edge.raw_log_priors[action.raw_action_id] = float(action.log_prior)
        return True

    def puct_priors(self, node_id: int) -> Dict[int, float]:
        """Effective PUCT priors of ``node_id``'s selectable (non-dead) edges.

        Each edge's mass is normalized in log space over the node's current
        non-dead edges, so the priors sum to one. They are derived on every
        call, so adding an edge, a new alias or a dead edge renormalizes them
        without touching any visit statistics or raw log-priors.
        """
        viable = [
            self.edges[eid]
            for eid in self.nodes[node_id].outgoing_edge_ids
            if self.edges[eid].status != EdgeStatus.DEAD
        ]
        if not viable:
            return {}
        masses = [edge.log_mass for edge in viable]
        normalizer = logsumexp(masses)
        if not isfinite(normalizer):
            raise ValueError(f"Node {node_id} has no finite prior mass to normalize.")
        return {edge.id: math.exp(mass - normalizer) for edge, mass in zip(viable, masses)}

    def mark_edge_dead(self, edge_id: int, *, note: Optional[str] = None) -> None:
        """The tactic failed to apply (Lean rejected it / executor error)."""
        edge = self.edges[edge_id]
        edge.status = EdgeStatus.DEAD
        if note:
            self.nodes[edge.source_id].note = note
        self._propagate(edge.source_id)

    def mark_node_exhausted(
        self,
        node_id: int,
        *,
        reason: NodeClosureReason,
        note: Optional[str] = None,
    ) -> None:
        """Close ``node_id``'s proposal source (its candidates or widening
        bounds are spent, the depth limit is reached, the provider proposed
        nothing, or replay failed). Lets ``_recompute_node`` decide DEAD vs.
        staying EXPANDED/SOLVED. A node is never reopened.
        """
        node = self.nodes[node_id]
        if reason == NodeClosureReason.NONE:
            raise ValueError("Closing a proposal source needs a reason.")
        node.closure_reason = reason
        if note:
            node.note = note
        self._propagate(node_id)

     
    def settle_proposal_round(self, node_id: int) -> None:
        """One proposal round at ``node_id`` finished and its source stays open.

        The node keeps its status (OPEN without edges, EXPANDED otherwise) even
        when every known edge failed: it can still be widened, so its known
        alternatives are not a complete failure.
        """
        self._propagate(node_id)

    def search_counters(self) -> dict[str, float]:
        """Proposal and widening work over the whole graph, for reports."""
        nodes = list(self.nodes.values())
        closures: Dict[str, int] = {}
        for node in nodes:
            if node.closure_reason != NodeClosureReason.NONE:
                closures[node.closure_reason.value] = closures.get(node.closure_reason.value, 0) + 1
        return {
            "proposal_rounds": float(sum(node.proposal_rounds for node in nodes)),
            "widening_rounds": float(sum(max(node.proposal_rounds - 1, 0) for node in nodes)),
            "raw_draws": float(sum(node.proposal_draws for node in nodes)),
            "distinct_commands": float(sum(node.attempted_commands for node in nodes)),
            "accepted_edges": float(len(self.edges)),
            "deferred_selections": float(self.deferred_selections),
            **{f"closed_{reason}": float(count) for reason, count in sorted(closures.items())},
        }

    # Queries
     
    @property
    def root(self) -> ProofNode:
        return self.nodes[self.root_id]

    def is_solved(self) -> bool:
        return self.root.status == NodeStatus.SOLVED

    def is_exhausted(self) -> bool:
        return self.root.status == NodeStatus.DEAD

    def frontier(self) -> List[ProofNode]:
        """Open nodes ranked by ``combined_rank``, descending.

        Recomputed on demand rather than cached in a heap: ranks mutate via
        backprop (objective 2), and a stale-priority heap is its own class
        of bugs (see design report, "re-ranking thrashing"). For the node
        counts a tactic search realistically holds open at once, an O(n log n)
        re-sort per pop is the simpler-correct choice over indexed-heap
        bookkeeping.
        """
        open_nodes = [node for node in self.nodes.values() if node.status == NodeStatus.OPEN]
        open_nodes.sort(key=lambda node: node.combined_rank, reverse=True)
        return open_nodes

    def tactic_path(self, node_id: int) -> List[str]:
        path: List[str] = []
        node = self.nodes[node_id]
        while node.incoming_edge_id is not None:
            edge = self.edges[node.incoming_edge_id]
            rendered = " ".join([edge.tactic.tactic_name, *edge.tactic.arguments]).strip()
            path.append(rendered)
            node = self.nodes[edge.source_id]
        path.reverse()
        return path

    def proof_trace(self) -> Optional[dict]:
        """If solved, the witnessing proof as a nested dict; else ``None``.

        A flat list can't faithfully represent an AND-edge with several
        subgoals — all of them had to be proven, not just one "main" branch
        — so this returns a tree: each node carries the tactic that closed
        it and a ``subgoals`` list with one fully-recursed entry per child
        (empty ⇒ the "no-goal" terminal). Where multiple tactics could have
        closed a node, ``_best_solved_edge`` picks the one with the highest
        ``_edge_value`` (tactic-probability × weakest-conjunct rank).
        """
        if not self.is_solved():
            return None
        return self._trace_from(self.root_id)

    def _trace_from(self, node_id: int) -> dict:
        node = self.nodes[node_id]
        edge = self._best_solved_edge(node_id)
        assert edge is not None, "proof_trace reached a node without a solved edge"
        return {
            "goal": node.goal.expression,
            "tactic": edge.tactic.tactic_name,
            "arguments": list(edge.tactic.arguments),
            "subgoals": [self._trace_from(child_id) for child_id in edge.child_ids],
        }

    def summary(self) -> dict:
        return {
            "root_id": self.root_id,
            "solved": self.is_solved(),
            "exhausted": self.is_exhausted(),
            "end_reason": self.end_reason.value,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "nodes": [node.summary() for node in self.nodes.values()],
            "edges": [edge.summary() for edge in self.edges.values()],
        }

    def __len__(self) -> int:
        return len(self.nodes)

     
    # Internals: bottom-up propagation (status + combined_rank)
     
    def _ancestor_state_keys(self, node_id: int) -> Set[str]:
        keys: Set[str] = set()
        node = self.nodes[node_id]
        while True:
            keys.add(_state_key(node.goal))
            if node.incoming_edge_id is None:
                return keys
            node = self.nodes[self.edges[node.incoming_edge_id].source_id]

    def _derive_edge_status(self, edge: ProofHyperedge) -> str:
        if not edge.child_ids:
            return EdgeStatus.SOLVED
        statuses = [self.nodes[cid].status for cid in edge.child_ids]
        if any(status == NodeStatus.DEAD for status in statuses):
            return EdgeStatus.DEAD
        if all(status == NodeStatus.SOLVED for status in statuses):
            return EdgeStatus.SOLVED
        return EdgeStatus.PENDING

    def _edge_value(self, edge: ProofHyperedge) -> float:
        """tactic-prior × AND-aggregate(children).

        AND-aggregate = ``min`` over child ``combined_rank``s — "a
        conjunction is only as strong as its weakest still-open conjunct" —
        the conventional pessimistic aggregation for AND-nodes (mirrors
        value backup in AND-OR search / HTPS). A childless edge (the
        "no-goal" terminal) contributes just the tactic's own probability.
        """
        if not edge.child_ids:
            return edge.tactic.probability
        return edge.tactic.probability * min(self.nodes[cid].combined_rank for cid in edge.child_ids)

    def _best_solved_edge(self, node_id: int) -> Optional[ProofHyperedge]:
        solved = [self.edges[eid] for eid in self.nodes[node_id].outgoing_edge_ids if self.edges[eid].status == EdgeStatus.SOLVED]
        if not solved:
            return None
        return max(solved, key=self._edge_value)

    def _recompute_node(self, node: ProofNode) -> bool:
        before = (node.status, node.combined_rank)

        if node.status not in (NodeStatus.SOLVED, NodeStatus.DEAD):
            edge_statuses = [self.edges[eid].status for eid in node.outgoing_edge_ids]
            if any(status == EdgeStatus.SOLVED for status in edge_statuses):
                node.status = NodeStatus.SOLVED
            elif edge_statuses:
                node.status = NodeStatus.EXPANDED
                if node.exhausted and all(status == EdgeStatus.DEAD for status in edge_statuses):
                    node.status = NodeStatus.DEAD
            elif node.exhausted:
                # Expanded zero edges (no candidates / executor rejected all)
                # and the search loop says it won't try again ⇒ a dead end.
                node.status = NodeStatus.DEAD

        viable = [
            self.edges[eid]
            for eid in node.outgoing_edge_ids
            if self.edges[eid].status != EdgeStatus.DEAD
        ]
        best_edge_value = max((self._edge_value(edge) for edge in viable), default=None)
        node.combined_rank = node.local_score if best_edge_value is None else max(node.local_score, best_edge_value)

        return (node.status, node.combined_rank) != before

    def _propagate(self, start_id: int) -> None:
        """Bottom-up fixpoint walk: recompute ``start_id``, refresh the edge
        above it, and keep climbing while something keeps changing.

        BFS with a `queued` set rather than plain recursion: a change can
        legitimately re-enqueue an already-visited ancestor once its other
        children settle, and recursion depth would otherwise track proof
        depth (fine for shallow goals, riskier for deep ones).
        """
        queue: Deque[int] = deque([start_id])
        queued: Set[int] = {start_id}
        while queue:
            current_id = queue.popleft()
            queued.discard(current_id)
            node = self.nodes[current_id]

            node_changed = self._recompute_node(node)

            propagate_further = node_changed
            if node.incoming_edge_id is not None:
                edge = self.edges[node.incoming_edge_id]
                new_status = self._derive_edge_status(edge)
                if new_status != edge.status:
                    edge.status = new_status
                    propagate_further = True

            if propagate_further and node.incoming_edge_id is not None:
                parent_id = self.edges[node.incoming_edge_id].source_id
                if parent_id not in queued:
                    queue.append(parent_id)
                    queued.add(parent_id)
