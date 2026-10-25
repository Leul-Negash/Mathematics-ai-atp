import asyncio
import argparse
import random
import json
import time
from enum import Enum
try:
    from graphviz import Digraph
except ImportError:
    Digraph = None
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set
from dataclasses import dataclass, field
from pantograph_async import (
    PantographClient,
    PantographConfigurationError,
    PantographEnv,
    PantographProtocolError,
    PantographStateError,
    PantographStateScope,
    PantographTacticRejected,
    PantographTransportError,
    TacticState,
)
from maths_ai.hybrid_reasoner.pantograph_adapter import execution_state_to_goals

from maths_ai.data_models.model_identity import carry_model_identity
from maths_ai.data_models.proof_components import (
    Goal,
    ProposedAction,
    RankedSubgoal,
    STV,
    TacticCandidate,
)
from maths_ai.lean_replay import LeanGoalSeed, ReplayInputError, seed_from_text
from maths_ai.gnn_inference.inference_engine import GNNModelEngine
from maths_ai.pln_inference.model import PLNInference
from maths_ai.pln_inference.metta.translator.translator_modules.runner import DynamicThompsonSampler

from maths_ai.hybrid_reasoner.hypergraph import (
    BackupSource,
    BackupValidity,
    BackupValue,
    EdgeStatus,
    NodeClosureReason,
    NodeStatus,
    ProofHypergraph,
    ProofNode,
    TacticExecutor,
    TacticOutcome,
    SearchEndReason,
)
from maths_ai.hybrid_reasoner.selection_policy import (
    puct_score,
    resolve_search_params,
    resolve_widening,
)
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    create_model_sexpr_client,
    materialize_seed,
)
from maths_ai.core.config import settings
from maths_ai.gnn_inference.atp_lean_gnn.reporting import console_print



def _is_lean_identifier(value: str) -> bool:
    parts = value.split(".")
    return bool(parts) and all(
        bool(core := part.rstrip("'!?")) and core.isidentifier()
        for part in parts
    )
_BRACKET_REQUIRED_TACTICS = frozenset({
    "rw", "rwa", "rewrite", "simp_rw", "simp only", "erw",
})
_BRACKET_OPTIONAL_TACTICS = frozenset({
    "simp", "simpa", "field_simp", "norm_num", "linarith", "nlinarith", "aesop",
})


def render_tactic_command(tactic: TacticCandidate) -> Optional[str]:
    name = tactic.tactic_name.strip()
    arguments = [arg.rstrip(":").strip() for arg in tactic.arguments]
    arguments = [arg for arg in arguments if arg]
    if name in _BRACKET_REQUIRED_TACTICS or name in _BRACKET_OPTIONAL_TACTICS:
        rules = [arg for arg in arguments if _is_lean_identifier(arg)]
        if rules:
            return f"{name} [{', '.join(rules)}]"
        if name in _BRACKET_REQUIRED_TACTICS:
            return None
        return name
    if not arguments:
        return name
    return " ".join([name, *arguments])


@dataclass
class _Simulation:
    """One selected partial hypertree.

    ``chosen_edges[node_id] = edge_id`` is the OR-choice at each node the
    descent passed through. ``leaves`` are the nodes where it stopped to
    request a proposal round: an OPEN node's first round, or a widening round
    at an expanded node whose capacity grew. ``deferred`` are unresolved nodes
    with nothing to propose or select yet. ``selected`` is every node the
    simulation included; each gets one visit when the simulation backs up.
    """

    chosen_edges: Dict[int, int] = field(default_factory=dict)
    leaves: List[int] = field(default_factory=list)
    deferred: List[int] = field(default_factory=list)
    selected: List[int] = field(default_factory=list)


# Failures of the session itself, never of the goal or tactic being tried.
_INFRASTRUCTURE_ERRORS = (
    PantographProtocolError,
    PantographTransportError,
    PantographStateError,
    PantographConfigurationError,
)


@dataclass(frozen=True)
class MaterializedGoal:
    """One canonical goal paired with the scoped state that produced it.

    The state is owned by the search's state scope and released once every
    candidate tactic for the node has finished; graph nodes keep only ``goal``.
    """

    goal: Goal
    state: TacticState
    fingerprint: str


class ExpansionResult(str, Enum):
    TACTICS_EXECUTED = "tactics_executed"
    NO_CANDIDATES = "no_candidates"
    DEPTH_LIMIT = "depth_limit"
    ELABORATION_ERROR = "elaboration_error"
    EXTERNAL_ABORT = "external_abort"


def plot_hypergraph(graph: ProofHypergraph) -> None:
    """Utility to visualize the proof hypergraph with Graphviz (for debugging
    and analysis).

    Nodes are labeled with their goal expressions; edges are labeled with
    the tactic applied and the STV of each resulting subgoal (if any).
    """

    dot = Digraph(comment="Proof Hypergraph")
    for node_id, node in graph.nodes.items():
        label = f"{node.goal.expression}\n" if node.stv is not None else node.goal.expression
        dot.node(str(node_id), label=label)
    print(graph.edges.items())
    for edge_id, edge in graph.edges.items():
        if edge.source_id is None:
            continue  # skip the root node's incoming edge (which has no tactic or subgoals)
        parent_id = edge.source_id
        child_ids = edge.child_ids
        tactic_label = f"{edge.tactic.tactic_name} {' '.join(edge.tactic.arguments)}"
        dot.edge(str(parent_id), str(child_ids[0]), label=tactic_label)

    dot.render("proof_hypergraph", format="png", cleanup=True)
class PantographExecutor(TacticExecutor):
    """Real executor: applies predicted tactics to actual Lean states via the
    Pantograph API.
    """
    def __init__(self, server: PantographClient):
        self.server = server

    async def apply(
        self,
        scope: PantographStateScope,
        state: TacticState,
        tactic: TacticCandidate,
    ) -> TacticOutcome:
        """Apply a tactic to a Lean goal state and report the outcome.

        Args:
            scope: Active allocation owner for this search.
            state: Current Lean goal state.
            tactic: Tactic to apply.

        Returns:
            ``TacticOutcome(success=True, subgoals=...)`` with the
            resulting subgoals (empty ⇒ this branch is fully discharged),
            translated from Pantograph's ``Goal``s into
            ``maths_ai`` ``Goal``s (``expression`` = the goal's target,
            ``hypotheses`` = its local variables rendered as ``name : type``).
            On a checked Lean rejection (the tactic doesn't apply, or it
            admits an unsafe dependency), returns
            ``TacticOutcome(success=False, error=...)``. Session failures
            propagate; they say nothing about the tactic.
        """
        tactic_cmd = render_tactic_command(tactic)
        if tactic_cmd is None:
            return TacticOutcome(
                success=False,
                subgoals=[],
                error=(
                    f"{tactic.tactic_name} requires a bracketed rule list and none of "
                    f"its sampled arguments {tactic.arguments} is a usable name"
                ),
            )

        try:
            new_state = await scope.apply_tactic(state, tactic_cmd)
        except PantographTacticRejected as exc:
            return TacticOutcome(success=False, subgoals=[], error=str(exc))
        # Keep only canonical goals; the alternative's allocation is released
        # at once. Errors leave cleanup to the enclosing scope.
        subgoals = execution_state_to_goals(new_state)
        await scope.release(new_state)
        return TacticOutcome(success=True, subgoals=subgoals, error=None)


class HybridReasoner:
    """Best-first AND-OR proof search guided by a GNN tactic policy and a
    PLN symbolic ranker (see the hybrid-reasoner design report for the full
    architecture rationale, the HTPS-style hypergraph rationale, and the
    enumerated edge cases referenced throughout this module's docstrings).

    Pipeline per expansion step (objective 1):
      1. ``predict_next_tactic``  — GNN: top-k ``(tactic, args, probability)``
      2. ``self.executor.apply``  — apply each tactic to the goal (the
         "no-goal" terminal is an empty subgoal list on success)
      3. ``rank_subgoals``        — PLN: STV per subgoal, blended with the
         tactic's GNN probability into ``combined_rank``
      4. link every Lean-returned subgoal into the hypergraph

    Each link triggers ``ProofHypergraph``'s bottom-up propagation, which is
    objective 2: PLN-derived ranks continuously update the GNN-seeded scores
    of every ancestor, all the way back to the root.
    """

    def __init__(
        self,
        config_path: Path,
        tactic_model_path: Path,
        argument_model_path: Path,
        *,
        executor: PantographExecutor,
        index_path: Optional[Path] = None,
        corpus_path: Optional[Path] = None,
        top_k_tactics: int = 3,
        max_depth: int = 10,
        max_nodes: int = 500,
        selection_policy: Literal["legacy", "puct", "rp"] = "legacy",
        num_simulations: Optional[int] = None,
        sim_batch_size: Optional[int] = None,
        puct_c: Optional[float] = None,
        widening_initial_actions: Optional[int] = None,
        widening_coefficient: Optional[float] = None,
        widening_exponent: Optional[float] = None,
        widening_max_actions: Optional[int] = None,
        widening_max_draws: Optional[int] = None,
        dts_sampler: Optional[DynamicThompsonSampler] = None,
        dts_c: float = None,
        dts_random_seed: Optional[int] = None,
        use_pln: bool = True,
    ) -> None:
        self.use_pln = use_pln
        self._proving = False
        self._scope: PantographStateScope | None = None
        self.gnn_engine = self._build_gnn_engine(
            config_path=config_path,
            tactic_model_path=tactic_model_path,
            argument_model_path=argument_model_path,
            index_path=index_path,
            corpus_path=corpus_path,
        )

        self.atomic_tactics = {}

        # Always use Thompson sampling as fallback - it's automatic and internal
        self.pln_fallback_strategy = "thompson"

        # Use config defaults if not provided
        if dts_c is None:
            dts_c = settings.dts_default_c
        if dts_random_seed is None:
            dts_random_seed = settings.dts_default_seed

        # _dts_rng is cheap and may be referenced by subclasses — always create it.
        self._dts_rng = random.Random(dts_random_seed)

        if not use_pln:
            # PLN disabled: no petta subprocess ever spawns; no DTS I/O.
            self.petta_chainer = None
            self.dts_sampler = None
        else:
            self.petta_chainer = PLNInference()
            # Initialize or use provided DTS sampler
            if dts_sampler is not None:
                self.dts_sampler = dts_sampler
                self.dts_sampler.C = dts_c
            else:
                # Try to load existing state from default location
                if settings.dts_state_file.exists():
                    try:
                        self.dts_sampler = DynamicThompsonSampler.load_from(
                            str(settings.dts_state_file), C=dts_c
                        )
                    except Exception:
                        self.dts_sampler = DynamicThompsonSampler(C=dts_c)
                else:
                    self.dts_sampler = DynamicThompsonSampler(C=dts_c)

        self.executor = executor
        self.server = executor.server
        self._cached_root_materialization: MaterializedGoal | None = None

        self.top_k_tactics = top_k_tactics
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.selection_policy = selection_policy
        # resolve_search_params rejects a budget set under "legacy" and requires
        # num_simulations under "puct"; the resolved numeric values are stored
        # (legacy receives placeholders its loop never reads).
        self.num_simulations, self.sim_batch_size, self.puct_c = resolve_search_params(
            selection_policy, num_simulations, sim_batch_size, puct_c
        )
        # PUCT widens every node within these bounds; best-first has none.
        self.widening = resolve_widening(
            selection_policy,
            widening_initial_actions,
            widening_coefficient,
            widening_exponent,
            widening_max_actions,
            widening_max_draws,
        )

    # GNN side
    def _build_gnn_engine(
        self,
        *,
        config_path: Path,
        tactic_model_path: Path,
        argument_model_path: Path,
        index_path: Optional[Path],
        corpus_path: Optional[Path],
    ) -> Optional[GNNModelEngine]:
        """Construct the tactic-prediction engine.

        Split out of ``__init__`` so subclasses can substitute a different
        policy source: ``RLHybridReasoner`` overrides this to return ``None``
        and instead drives ``predict_next_tactic`` from a live
        ``ActorCriticTacticModel`` whose sampled actions must stay attached
        to the training graph.
        """
        return GNNModelEngine(
            config_path=config_path,
            tactic_predictor_model_path=tactic_model_path,
            argument_predictor_model_path=argument_model_path,
            index_path=index_path,
            corpus_path=corpus_path,
        )

    def predict_next_tactic(self, sub_goal: Goal) -> List[ProposedAction]:
        """
            Args:
                sub_goal: the sanitized target sub_goal (expression plus its
                    local hypotheses) for which tactics are predicted. The
                    base engine and the RL subclass consume the full canonical
                    goal, so sampled arguments resolve against the same DAG
                    the encoder saw.
            Returns:
                up to `top_k_tactics` ProposedAction records, each a
                TacticCandidate(tactic_name, arguments, probability) ranked by
                tactic probability, descending, with the complete action's
                log-prior under the engine's documented argument law.

                An empty list means the GNN found no viable tactic for this
                goal (see GNNModelEngine.inference's "degenerate prediction"
                edge case) — callers must treat that as a dead branch, which
                `_expand` below does via `graph.mark_node_exhausted`.
        """
        return self.gnn_engine.propose(sub_goal, top_k=self.top_k_tactics)

    # PLN side
    def _make_dts_key(self, parent_goal: str, tactic: TacticCandidate, subgoal: Goal) -> str:
        """Create a meaningful DTS state key that identifies the subgoal in context.
        
        The key includes:
        - Parent goal expression
        - Tactic name and arguments (what was applied to get here)
        - Subgoal expression
        
        This makes the DTS state more interpretable and allows tracking
        which tactics lead to which subgoals.
        """
        tactic_str = f"{tactic.tactic_name}({' '.join(tactic.arguments)})" if tactic.arguments else tactic.tactic_name
        # Truncate long expressions to keep keys manageable
        max_len = 50
        parent_trunc = parent_goal[:max_len] + "..." if len(parent_goal) > max_len else parent_goal
        subgoal_trunc = subgoal.expression[:max_len] + "..." if len(subgoal.expression) > max_len else subgoal.expression
        return f"{parent_trunc} --[{tactic_str}]--> {subgoal_trunc}"

    async def rank_subgoals(
        self,
        goal: str,
        sub_goals: List[Goal],
        tactic: TacticCandidate,
        *,
        gnn_probability: float = 1.0,
    ) -> List[RankedSubgoal]:
        """Score ``sub_goals`` with PLN and rank them best-first.

        Now async: the per-subgoal PLN queries (blocking ``subprocess.run`` inside
        ``PLNInference.evaluate``) are dispatched concurrently via ``evaluate_async`` +
        ``asyncio.gather``, so scoring N subgoals overlaps their process waits instead of
        serializing them, and the event loop stays free for other concurrent searches.

        Args:
            goal: the parent goal's expression — passed to PLN as extra
                local context (one more hypothesis the subgoal may rely on).
            sub_goals: candidate subgoals produced by applying one tactic to
                ``goal``, each carrying its own local hypotheses (the
                executor's variable context for that subgoal).
            tactic: the tactic that was applied to produce these subgoals
                (used for DTS key generation to track tactic->subgoal relationships).
            gnn_probability: that tactic's predicted probability. Folding it
                in here is what makes ``combined_rank = gnn_prob × STV.score``
                (objective 2's "the GNN score should be updated [by the PLN
                rank]"); the default ``1.0`` makes this usable as a
                standalone PLN-only ranking utility too.

        Returns:
            ``RankedSubgoal``s sorted by ``combined_rank``, descending.

        Note (design-report edge case — PLN soundness): a high STV here
        reflects what PeTTaChainer can derive from the asserted local facts
        (themselves asserted at ``(STV 1.0 1.0)`` regardless of whether
        they're true), not a guarantee that the subgoal is actually provable.
        It is the best automatic heuristic available, not ground truth.
        """
        if not self.use_pln:
            raise RuntimeError(
                "rank_subgoals requires PLN; the reasoner was constructed with use_pln=False"
            )
        # Dispatch all PLN queries concurrently (off the event-loop thread).
        results = await asyncio.gather(
            *(
                self.petta_chainer.evaluate_async(
                    subgoal.expression,
                    hypotheses=[goal, *subgoal.hypotheses],
                )
                for subgoal in sub_goals
            )
        )

        # DTS bookkeeping is cheap and stateful — process results sequentially, in order.
        ranked: List[RankedSubgoal] = []
        for subgoal, result in zip(sub_goals, results):
            stv = result.stv
            if (
                self.pln_fallback_strategy == "thompson"
                and result.is_fallback
                and self.dts_sampler is not None
            ):
                key = self._make_dts_key(goal, tactic, subgoal)
                sampled = self.dts_sampler.sample(key, self._dts_rng)
                stv = STV(strength=sampled, confidence=1.0)
            elif self.pln_fallback_strategy == "thompson" and self.dts_sampler is not None:
                key = self._make_dts_key(goal, tactic, subgoal)
                self.dts_sampler.record_observation(key, reward=stv.score)

            ranked.append(
                RankedSubgoal(
                    goal=subgoal,
                    stv=stv,
                    gnn_probability=gnn_probability,
                )
            )

        ranked.sort(key=lambda candidate: candidate.combined_rank, reverse=True)
        return ranked

     
    # Joint search
     
    async def prove(
        self,
        seed: LeanGoalSeed,
        *,
        deadline: Optional[float] = None,
    ) -> ProofHypergraph:
        """Run AND-OR search over the hypergraph rooted at ``seed``.

        A seed that cannot be replayed faithfully raises ``ReplayInputError``
        before any graph exists; that is a property of the input, not a failed
        proof, and it leaves no search evidence.

        Two search modes, selected by ``selection_policy``:

        * ``"legacy"`` (default) — the best-first loop: pop the
          highest-``combined_rank`` open node, expand it (``_expand``), which
          links any new subgoals into the hypergraph — and every link
          immediately backpropagates updated status/rank to every ancestor
          (``ProofHypergraph._propagate``), re-ordering the frontier for the
          next iteration. Terminates on root SOLVED/DEAD, empty frontier,
          ``max_nodes``, or deadline.
        * ``"puct"`` — HTPS-style repeated simulation (``_prove_mcts``):
          PUCT-guided partial-hypertree selection with virtual loss, batched
          leaf expansion, and per-edge N/N_v/W/Q visit statistics backed up after
          every simulation, for ``num_simulations`` simulations.

        ``deadline`` is a ``time.monotonic()`` timestamp: when exceeded, the
        loop stops between expansions (or between simulation batches) and the
        partial graph is returned — the caller keeps whatever experience was
        gathered rather than losing it to a hard cancellation.

        Termination (design-report "no-goal ambiguity" — resolved here as):
          * root SOLVED  → proof found; ``graph.proof_trace()`` replays it
          * root DEAD    → provably unsolvable within the explored space
          * frontier empty / ``max_nodes`` reached / deadline exceeded →
            budget exhaustion (open design question: what to return — we
            return the partial graph so the caller can inspect
            ``graph.frontier()``, resume, or visualize it; see
            ``ProofHypergraph.summary``)

        ``max_depth`` bounds branch depth and ``max_nodes`` bounds total
        graph size — the design report's "branching-factor explosion"
        safeguards. Cycle detection (a subgoal identical to one of its own
        ancestors) is handled inside ``ProofHypergraph.add_edge``.

        Every Lean state of one search is allocated in one state scope, so the
        cached root and every batched materialization are released when the
        search ends, on any exit path. One reasoner runs one search at a time.
        """
        if self._proving:
            raise PantographStateError("A reasoner supports only one active search.")
        self._proving = True
        try:
            async with self.server.state_scope() as scope:
                self._scope = scope
                return await self._prove_scoped(seed, deadline)
        finally:
            self._scope = None
            self._cached_root_materialization = None
            self._proving = False

    async def _prove_scoped(
        self,
        seed: LeanGoalSeed,
        deadline: Optional[float],
    ) -> ProofHypergraph:
        root_materialization = await self._materialize_goal(seed)
        graph = ProofHypergraph(root_materialization.goal)
        self._cached_root_materialization = root_materialization

        if self.selection_policy == "puct":
            await self._prove_mcts(graph, deadline)
            return graph

        loop_count = 0

        #Running through the loop untill the theorem is solved or the depth_limit is reached
        while not graph.is_solved() and not graph.is_exhausted() and len(graph.nodes) < self.max_nodes:
            if not self._within_deadline(deadline):
                graph.end_reason = SearchEndReason.DEADLINE
                break
            frontier = graph.frontier()
            if not frontier:
                graph.end_reason = SearchEndReason.FRONTIER_EMPTY
                break
            loop_count += 1
            node = frontier[0]
            print(f"\n=== Loop {loop_count}: expanding node {node.id} (depth {node.depth}) | goal: {node.goal.expression} ===")
            await self._expand(graph, node)

        if graph.is_solved():
            graph.end_reason = SearchEndReason.ROOT_SOLVED
        elif graph.is_exhausted():
            graph.end_reason = SearchEndReason.ROOT_DEAD
        elif len(graph.nodes) >= self.max_nodes:
            graph.end_reason = SearchEndReason.MAX_NODES
        return graph

    @staticmethod
    def _within_deadline(deadline: Optional[float]) -> bool:
        return deadline is None or time.monotonic() < deadline

    # ------------------------------------------------------------------
    # HTPS-style multi-simulation search (selection_policy == "puct")
    # ------------------------------------------------------------------

    async def _prove_mcts(self, graph: ProofHypergraph, deadline: Optional[float]) -> None:
        """Repeated-simulation loop: select B partial hypertrees under PUCT +
        virtual loss, expand their unexpanded leaves in one batched proposal,
        then back up each simulation's value into the traversed edges' N/N_v/W/Q
        statistics. Unknown outcomes increment N and release virtual loss, but do not
        increment N_v or W.
        """
        simulations_done = 0
        while (
            not graph.is_solved()
            and not graph.is_exhausted()
            and len(graph.nodes) < self.max_nodes
            and simulations_done < self.num_simulations
            and self._within_deadline(deadline)
        ):
            simulations: List[_Simulation] = []
            for _ in range(min(self.sim_batch_size, self.num_simulations - simulations_done)):
                simulation = self._select_partial_hypertree(graph)
                if simulation is None:
                    break  # no selectable path (everything resolved or stuck)
                simulations.append(simulation)
            if not simulations:
                break

            # Deduplicate leaves across the batch: two simulations steered
            # apart by virtual loss can still meet at a shared leaf.
            leaf_ids: List[int] = []
            seen: Set[int] = set()
            for simulation in simulations:
                for node_id in simulation.leaves:
                    if node_id not in seen:
                        seen.add(node_id)
                        leaf_ids.append(node_id)

            try:
                await self._expand_leaves(graph, leaf_ids)
            except BaseException:
                for simulation in simulations:
                    self._release_virtual_losses(graph, simulation)
                graph.end_reason = SearchEndReason.EXTERNAL_ABORT
                raise

            for index, simulation in enumerate(simulations):
                try:
                    self._backup_simulation(graph, simulation)
                except BaseException:
                    # The failed backup released its own losses; the rest of
                    # the batch never completed and records no evidence.
                    for pending in simulations[index + 1 :]:
                        self._release_virtual_losses(graph, pending)
                    graph.end_reason = SearchEndReason.EXTERNAL_ABORT
                    raise
            simulations_done += len(simulations)

        if graph.is_solved():
            graph.end_reason = SearchEndReason.ROOT_SOLVED
        elif graph.is_exhausted():
            graph.end_reason = SearchEndReason.ROOT_DEAD
        elif len(graph.nodes) >= self.max_nodes:
            graph.end_reason = SearchEndReason.MAX_NODES
        elif simulations_done >= self.num_simulations:
            graph.end_reason = SearchEndReason.NUM_SIMULATIONS
        elif not self._within_deadline(deadline):
            graph.end_reason = SearchEndReason.DEADLINE
        else:
            graph.end_reason = SearchEndReason.FRONTIER_EMPTY

    def _select_partial_hypertree(self, graph: ProofHypergraph) -> Optional["_Simulation"]:
        """Descend from the root, entering **every** child of each chosen edge
        — a simulation must reach a full set of leaves consistent with one
        candidate proof (the AND semantics).

        At each unresolved node, in order: if its proposal source may propose
        now (its first round, or its widening capacity has grown since the
        last), stop there and request a round; otherwise pick the non-DEAD
        edge that maximizes ``puct_score`` and increment its
        ``virtual_loss``, so the next selection in the same batch is steered
        away from this in-flight path; with no such edge, the node is
        deferred — the simulation still visits it, so its capacity can grow.

        Returns ``None`` when nothing is selectable (the root is resolved).
        """
        simulation = _Simulation()
        stack: List[int] = [graph.root_id]
        while stack:
            node_id = stack.pop()
            node = graph.nodes[node_id]
            if node.status in (NodeStatus.SOLVED, NodeStatus.DEAD):
                continue  # terminal for this simulation; backup reads the status
            simulation.selected.append(node_id)
            if self._round_draws(node) > 0:
                simulation.leaves.append(node_id)
                continue
            viable = [
                graph.edges[eid]
                for eid in node.outgoing_edge_ids
                if graph.edges[eid].status != EdgeStatus.DEAD
            ]
            if not viable:
                # Open source, nothing known to select, no capacity yet.
                simulation.deferred.append(node_id)
                graph.deferred_selections += 1
                continue
            total_visits = sum(e.visit_stats.N + e.visit_stats.virtual_loss for e in viable)
            priors = graph.puct_priors(node_id)
            best = max(
                viable,
                key=lambda e: puct_score(e.visit_stats, priors[e.id], total_visits, self.puct_c),
            )
            simulation.chosen_edges[node_id] = best.id
            best.visit_stats.virtual_loss += 1
            stack.extend(best.child_ids)

        if not simulation.selected:
            return None
        return simulation

    def _round_draws(self, node: ProofNode) -> int:
        """How many raw draws a proposal round at ``node`` may make now.

        Zero when its source is closed or, under widening, when its capacity
        at the node's completed visits is already used. Without widening
        (best-first), an OPEN node gets one round of ``top_k_tactics``.
        """
        if node.exhausted or node.status in (NodeStatus.SOLVED, NodeStatus.DEAD):
            return 0
        if self.widening is None:
            return self.top_k_tactics if node.status == NodeStatus.OPEN else 0
        return self.widening.draws_for(
            visits=node.visits,
            attempted=node.attempted_commands,
            draws=node.proposal_draws,
            per_round=self.top_k_tactics,
        )

    def _proposal_source_is_finite(self) -> bool:
        """The provider gives the same candidates every round, so one is enough.

        The generic engine ranks deterministically; the RL subclass is finite
        only under a greedy proposal.
        """
        return True

    def predict_next_tactics_batch(self, sub_goals: List[Goal]) -> List[List[ProposedAction]]:
        """Propose tactic candidates for several goals at once.

        Default implementation loops ``predict_next_tactic`` so the base
        class and non-RL callers are unaffected; ``RLHybridReasoner``
        overrides it with one multi-graph policy forward across the batch.
        """
        return [self.predict_next_tactic(sub_goal) for sub_goal in sub_goals]

    def predict_next_tactics_for_nodes(
        self, nodes: List[ProofNode]
    ) -> List[List[ProposedAction]]:
        """Internal node-aware proposal seam used by both search modes.

        The base reasoner needs only sanitized goals. RL overrides this method so
        sampled actions can be owned by the exact graph node even when two nodes
        contain identical goal text.
        """
        return self.predict_next_tactics_batch([node.goal.require_model_state() for node in nodes])

    async def _expand_leaves(self, graph: ProofHypergraph, leaf_ids: List[int]) -> None:
        """Materialize a batch of proposal requests, propose once, then execute
        sequentially.

        A request is a node's first round or a widening round; a node appears
        at most once per batch, and its round's draw count is fixed from its
        completed visits before any draw. Every node state stays allocated
        until all of its candidate tactics have finished, and is released as
        soon as they have. On an exception the search scope releases whatever
        is still allocated.
        """
        materialized: List[tuple[ProofNode, MaterializedGoal]] = []
        draws: List[int] = []
        for node_id in sorted(set(leaf_ids)):
            node = graph.nodes[node_id]
            count = self._round_draws(node)
            if count <= 0:
                continue  # resolved by an earlier leaf's propagation this batch
            if node.depth >= self.max_depth:
                graph.mark_node_exhausted(
                    node.id,
                    reason=NodeClosureReason.DEPTH_LIMIT,
                    note=f"depth limit ({self.max_depth}) reached",
                )
                self._on_expansion_complete(node, ExpansionResult.DEPTH_LIMIT)
                continue
            current = await self._materialize_for_expansion(graph, node)
            if current is not None:
                materialized.append((node, current))
                draws.append(count)
        if not materialized:
            return

        nodes = [node for node, _current in materialized]
        proposals = (
            self.predict_next_tactics_for_nodes(nodes)
            if self.widening is None
            else self.predict_next_tactics_for_nodes(nodes, draws)
        )

        if len(proposals) != len(materialized):
            raise RuntimeError("Batched proposals do not align with the leaf batch.")

        for (node, current), candidates, count in zip(materialized, proposals, draws):
            if not candidates:
                self._settle_round(
                    graph, node, count, closure=NodeClosureReason.NO_CANDIDATES,
                    note="GNN returned no viable tactic",
                )
                self._on_expansion_complete(node, ExpansionResult.NO_CANDIDATES)
            else:
                await self._execute_and_link(
                    graph,
                    node,
                    candidates,
                    materialized=current,
                    draws=count,
                )
            await self._release_materialized(current)

    def _settle_round(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        draws: int,
        *,
        closure: NodeClosureReason,
        note: Optional[str],
    ) -> None:
        """Account one finished round and close the source only if it is spent.

        Best-first and finite providers close after their one round, with
        ``closure``. A widening source closes when a hard bound is reached;
        otherwise it stays open, and a node whose known edges all failed is
        not dead yet.
        """
        node.proposal_rounds += 1
        node.proposal_draws += draws
        if self.widening is None or self._proposal_source_is_finite():
            if note:
                print(f"  [Node {node.id} EXHAUSTED] {note}")
            graph.mark_node_exhausted(node.id, reason=closure, note=note)
        elif self.widening.exhausted(attempted=node.attempted_commands, draws=node.proposal_draws):
            graph.mark_node_exhausted(
                node.id,
                reason=NodeClosureReason.WIDENING_BOUND,
                note=(
                    f"widening bound reached ({node.attempted_commands} commands, "
                    f"{node.proposal_draws} draws)"
                ),
            )
        else:
            graph.settle_proposal_round(node.id)

    @staticmethod
    def _release_virtual_losses(graph: ProofHypergraph, simulation: "_Simulation") -> None:
        for edge_id in simulation.chosen_edges.values():
            stats = graph.edges[edge_id].visit_stats
            stats.virtual_loss = max(0, stats.virtual_loss - 1)

    @staticmethod
    def _failure_backup(node: ProofNode) -> BackupValue:
        if node.closure_reason in (
            NodeClosureReason.ELABORATION_ERROR,
            NodeClosureReason.EXTERNAL_ABORT,
        ):
            return BackupValue.unknown()
        return BackupValue.valid(0.0, BackupSource.SEARCH_FAILURE)

    @staticmethod
    def _and_backup(children: List[BackupValue]) -> BackupValue:
        if not children:
            return BackupValue.valid(1.0, BackupSource.LEAN_STATUS)
        if any(
            child.validity == BackupValidity.VALID and child.value == 0.0
            for child in children
        ):
            return BackupValue.valid(0.0, BackupSource.SEARCH_FAILURE)
        if any(child.validity == BackupValidity.UNKNOWN for child in children):
            return BackupValue.unknown()

        value = 1.0
        sources = set()
        for child in children:
            assert child.value is not None
            value *= child.value
            sources.add(child.source)
        if BackupSource.CRITIC_BOOTSTRAP in sources:
            source = BackupSource.CRITIC_BOOTSTRAP
        elif BackupSource.VISIT_MEAN in sources:
            source = BackupSource.VISIT_MEAN
        elif BackupSource.SEARCH_FAILURE in sources:
            source = BackupSource.SEARCH_FAILURE
        else:
            source = BackupSource.LEAN_STATUS
        return BackupValue.valid(value, source)

    def _backup_simulation(self, graph: ProofHypergraph, simulation: "_Simulation") -> None:
        """Walk the simulated tree bottom-up, updating each traversed edge's
        visit statistics.

        Node values are typed: a Lean-confirmed SOLVED node is valid ``1.0``, a genuine
        searched failure is valid ``0.0``, and an unelaborated or infrastructure-failed
        node is UNKNOWN. An unresolved leaf uses ``_leaf_value`` (the critic in the RL
        subclass), while an interior node uses the validity-aware AND-combination of all
        children on the chosen edge. Each chosen edge always gets ``N += 1`` and releases
        its virtual loss. Only a valid numeric edge backup additionally gets ``N_v += 1``
        and ``W += value``; therefore ``Q = W/N_v`` and UNKNOWN never becomes a
        critic ``0.0`` target. Status propagation is not this walk's job —
        ``add_edge`` already ran ``_propagate`` during expansion.
        """
        values: Dict[int, BackupValue] = {}

        def node_value(node_id: int) -> BackupValue:
            if node_id in values:
                return values[node_id]
            node = graph.nodes[node_id]
            if node.status == NodeStatus.SOLVED:
                backup = BackupValue.valid(1.0, BackupSource.LEAN_STATUS)
            else:
                edge_id = simulation.chosen_edges.get(node_id)
                if edge_id is not None:
                    backup = self._and_backup(
                        [node_value(child_id) for child_id in graph.edges[edge_id].child_ids]
                    )
                elif node.status == NodeStatus.DEAD:
                    backup = self._failure_backup(node)
                else:
                    estimate = min(1.0, max(0.0, self._leaf_value(node)))
                    backup = BackupValue.valid(
                        estimate,
                        BackupSource.CRITIC_BOOTSTRAP,
                    )
            values[node_id] = backup
            return backup

        try:
            backups = {
                edge_id: self._and_backup(
                    [node_value(child_id) for child_id in graph.edges[edge_id].child_ids]
                )
                for edge_id in simulation.chosen_edges.values()
            }
        except BaseException:
            self._release_virtual_losses(graph, simulation)
            raise
        for edge_id, edge_backup in backups.items():
            stats = graph.edges[edge_id].visit_stats
            stats.N += 1
            if edge_backup.validity == BackupValidity.VALID:
                assert edge_backup.value is not None
                stats.N_v += 1
                stats.W += edge_backup.value
            stats.virtual_loss = max(0, stats.virtual_loss - 1)
        # One node visit per completed simulation, whatever its value's
        # validity; no edge visit is invented where none was chosen.
        for node_id in simulation.selected:
            graph.nodes[node_id].visits += 1

    def _leaf_value(self, node: ProofNode) -> float:
        """Value estimate for an unresolved simulation leaf.

        The base class has no critic, so it returns the uninformed 0.5;
        ``RLHybridReasoner`` overrides this with the critic head's value
        estimate (HTPS's ``v_T(g) = c_θ(g)``).
        """
        return 0.5

    def _on_expansion_complete(
        self,
        node: ProofNode,
        result: ExpansionResult,
    ) -> None:
        """Hook: called once a node's expansion has fully finished (all
        candidates executed and linked, or the node was exhausted without a
        proposal). The RL subclass flushes that node's still-pending sampled
        actions to failure records here.
        """

    async def _materialize_goal(self, goal: Goal | LeanGoalSeed) -> MaterializedGoal:
        """Replay a seed or a canonical goal and return the checked live goal.

        The fingerprint is taken from the model-facing goal, after its source
        model identity is restored and its model expressions are checked.
        """
        if self._scope is None:
            raise PantographStateError("Materialization requires an active search scope.")
        seed = goal.replay_seed() if isinstance(goal, Goal) else goal
        state, canonical, _ = await materialize_seed(self._scope, seed)
        return MaterializedGoal(
            goal=canonical,
            state=state,
            fingerprint=canonical.state_fingerprint(),
        )

    async def _release_materialized(self, materialized: MaterializedGoal) -> None:
        if self._scope is not None:
            await self._scope.release(materialized.state)

    async def _materialize_node(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
    ) -> MaterializedGoal:
        cached = self._cached_root_materialization
        if (
            node.id == graph.root_id
            and cached is not None
            and cached.fingerprint == node.goal.state_fingerprint()
        ):
            self._cached_root_materialization = None
            current = cached
        else:
            current = await self._materialize_goal(node.goal)
        node.goal = current.goal
        return current

    async def _materialize_for_expansion(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
    ) -> MaterializedGoal | None:
        """Materialize ``node`` or close it without learning evidence.

        A replay Lean rejects, or one that does not match the goal it describes,
        closes the node as unelaborated and the search continues. A session failure closes it as an external abort
        and propagates, so the caller retires the client.
        """
        try:
            return await self._materialize_node(graph, node)
        except ReplayInputError as exc:
            console_print(f"  [Node {node.id} SKIP] goal elaboration failed: {exc}")
            graph.mark_node_exhausted(
                node.id,
                reason=NodeClosureReason.ELABORATION_ERROR,
                note=f"elaboration error: {exc}",
            )
            self._on_expansion_complete(node, ExpansionResult.ELABORATION_ERROR)
        except _INFRASTRUCTURE_ERRORS as exc:
            console_print(f"  [Node {node.id} ABORT] Pantograph unavailable: {exc}")
            graph.mark_node_exhausted(
                node.id,
                reason=NodeClosureReason.EXTERNAL_ABORT,
                note=f"Pantograph unavailable: {exc}",
            )
            graph.end_reason = SearchEndReason.EXTERNAL_ABORT
            self._on_expansion_complete(node, ExpansionResult.EXTERNAL_ABORT)
            raise
        return None

    def _link(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        action: ProposedAction,
        ranked_subgoals: list,
    ):
        """Link one successful application of ``action`` into the hypergraph
        and return the created hyperedge. Training data is attached through
        ``_on_command_settled``, not here.
        """
        return graph.add_edge(node.id, action, ranked_subgoals=ranked_subgoals)

    async def _expand(self, graph: ProofHypergraph, node: ProofNode) -> None:
        """Try each of the GNN's top-k tactics on ``node`` and link whatever
        survives (executor success) into the hypergraph as new hyperedges.
        """
        if node.depth >= self.max_depth:
            graph.mark_node_exhausted(
                node.id,
                reason=NodeClosureReason.DEPTH_LIMIT,
                note=f"depth limit ({self.max_depth}) reached",
            )
            self._on_expansion_complete(node, ExpansionResult.DEPTH_LIMIT)
            return

        materialized = await self._materialize_for_expansion(graph, node)
        if materialized is None:
            return
        print(f"  [GNN Input] goal={node.goal.expression}  hyps={node.goal.hypotheses}")
        candidates = self.predict_next_tactics_for_nodes([node])[0]
        if not candidates:
            self._settle_round(
                graph, node, self.top_k_tactics, closure=NodeClosureReason.NO_CANDIDATES,
                note="GNN returned no viable tactic",
            )
            self._on_expansion_complete(node, ExpansionResult.NO_CANDIDATES)
        else:
            await self._execute_and_link(
                graph,
                node,
                candidates,
                materialized=materialized,
            )
        await self._release_materialized(materialized)

    async def _execute_and_link(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        proposals: List[ProposedAction],
        *,
        materialized: MaterializedGoal,
        draws: Optional[int] = None,
    ) -> None:
        """Execute one proposal round at ``node`` and link the survivors — the
        per-node execution stage shared by the best-first path (``_expand``)
        and the batched MCTS path (``_expand_leaves``).

        Proposals are grouped by rendered command. A command runs in Lean at
        most once per node: its checked outcome (new edge or rejection) is
        kept in ``node.command_outcomes``, and a later raw action rendering it
        reuses that outcome. A raw action that reaches an existing edge is
        recorded as an alias of it, which adds its prior mass once. Every
        group is reported to ``_on_command_settled``; the round ends with
        ``_settle_round`` (``draws`` raw draws, ``len(proposals)`` by default)
        and ``_on_expansion_complete``.
        """
        if materialized.fingerprint != node.goal.state_fingerprint():
            raise RuntimeError(f"Node {node.id} goal changed after action sampling.")
        state = materialized.state
        any_applied = False

        groups: Dict[tuple, List[ProposedAction]] = {}
        for action in proposals:
            groups.setdefault(action.command, []).append(action)

        for command, actions in groups.items():
            tactic = actions[0].candidate
            if command in node.command_outcomes:
                edge_id = node.command_outcomes[command]
                if edge_id is not None:
                    any_applied = True
                    for action in actions:
                        graph.observe_alias(edge_id, action)
                self._on_command_settled(node, command, actions, edge_id, executed=False)
                continue

            try:
                outcome = await self.executor.apply(self._scope, state, tactic)
            except _INFRASTRUCTURE_ERRORS as exc:
                graph.mark_node_exhausted(
                    node.id,
                    reason=NodeClosureReason.EXTERNAL_ABORT,
                    note=f"Pantograph failed during tactic execution: {exc}",
                )
                graph.end_reason = SearchEndReason.EXTERNAL_ABORT
                self._on_expansion_complete(node, ExpansionResult.EXTERNAL_ABORT)
                raise

            if not outcome.success:
                print(f"  [Tactic FAILED] {tactic.tactic_name} {' '.join(tactic.arguments)} — {outcome.error}")
                node.command_outcomes[command] = None
                self._on_command_settled(node, command, actions, None, executed=True)
                continue
            any_applied = True
            print(f"  [Tactic APPLIED] {tactic.tactic_name} {' '.join(tactic.arguments)} (prob={tactic.probability:.4f})")
            # Successors see the model identity of the goal just materialized,
            # before they are ranked, linked or compared with other nodes.
            subgoals = [
                carry_model_identity(materialized.goal, subgoal)
                for subgoal in outcome.subgoals
            ]

            if not subgoals:
                # "no-goal": this tactic fully discharges the goal (QED for this branch)
                print(f"  [Tactic QED] no subgoals — branch closed!")
                chosen = []
            elif self.use_pln:
                print(f"  [PLN Ranking] scoring {len(subgoals)} subgoal(s)...")
                ranked = await self.rank_subgoals(
                    node.goal.expression, subgoals, tactic,
                    gnn_probability=tactic.probability,
                )
                print(f"  [PLN Done] ranked {len(ranked)} subgoal(s)")
                for i, rs in enumerate(ranked):
                    print(
                        f"    subgoal {i}: {rs.goal.expression} | "
                        f"stv=({rs.stv.strength:.3f}, {rs.stv.confidence:.3f}) | "
                        f"combined_rank={rs.combined_rank:.4f}"
                    )
                chosen = [(candidate.goal, candidate.stv) for candidate in ranked]
            else:
                # PLN disabled: keep every Lean-returned subgoal in executor order.
                # stv=None propagates automatically: ProofNode.local_score degrades to
                # gnn_probability, and potential() returns 0.0 (no shaping).
                chosen = [(subgoal, None) for subgoal in subgoals]

            edge = self._link(graph, node, actions[0], ranked_subgoals=chosen)
            node.command_outcomes[command] = edge.id
            for action in actions[1:]:
                graph.observe_alias(edge.id, action)
            self._on_command_settled(node, command, actions, edge.id, executed=True)

        self._settle_round(
            graph,
            node,
            len(proposals) if draws is None else draws,
            closure=NodeClosureReason.CANDIDATES_EXHAUSTED,
            note=None if any_applied else "executor rejected every candidate tactic",
        )
        self._on_expansion_complete(node, ExpansionResult.TACTICS_EXECUTED)

    def _on_command_settled(
        self,
        node: ProofNode,
        command: tuple,
        actions: List[ProposedAction],
        edge_id: Optional[int],
        *,
        executed: bool,
    ) -> None:
        """Hook: ``actions`` (all rendering ``command``) now have a checked
        outcome — ``edge_id``, or None for a Lean rejection. ``executed`` is
        False when the outcome came from the node's cache. The RL subclass
        turns them into policy observations.
        """



async def main(
    config_path: Path,
    tactic_model_path: Path,
    argument_model_path: Path,
    *,
    index_path: Optional[Path] = None,
    corpus_path: Optional[Path] = None,
    goal_statement: str,
    hypotheses: Optional[List[str]] = None,
    depth_limit: int = 10,
    dts_state_input: Optional[Path] = None,
    dts_state_output: Optional[Path] = None,
    dts_c: float = None,
    dts_random_seed: Optional[int] = None,
    top_k_tactics: int = 3,
    source_root: Path,
    pantograph_repl: Path,

) -> None:
    env = PantographEnv(
        source_root=source_root,
        pantograph_repl=pantograph_repl,
        imports=("Init", "Mathlib"),
    )
    env.verify()

    # Auto-load DTS state from default location if not specified
    if dts_state_input is None and settings.dts_state_file.exists():
        dts_state_input = settings.dts_state_file

    # Auto-set output to default location if not specified
    if dts_state_output is None:
        dts_state_output = settings.dts_state_file

    dts_sampler = None
    if dts_state_input and dts_state_input.exists():
        try:
            # Check if file is non-empty before loading
            if dts_state_input.stat().st_size > 0:
                dts_sampler = DynamicThompsonSampler.load_from(
                    str(dts_state_input), C=dts_c if dts_c else settings.dts_default_c
                )
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not load DTS state from {dts_state_input}: {e}")
            print("Starting with fresh DTS state.")
            dts_sampler = None
    async with await create_model_sexpr_client(env) as server:
        hybrid_reasoner = HybridReasoner(
            config_path=config_path,
            tactic_model_path=tactic_model_path,
            argument_model_path=argument_model_path,
            index_path=index_path,
            corpus_path=corpus_path,
            executor=PantographExecutor(server=server),
            top_k_tactics=top_k_tactics,
            max_depth=depth_limit,
            max_nodes=500,
            dts_sampler=dts_sampler,
            dts_c=dts_c,
            dts_random_seed=dts_random_seed,
        )
        print("Goal:")
        print(repr(goal_statement))

        print("Hypotheses:")
        for h in hypotheses or []:
            print(repr(h))
        try:
            proof_graph = await hybrid_reasoner.prove(
                seed_from_text(goal=goal_statement, hypotheses=hypotheses or [])
            )
            if proof_graph.is_solved():
                print("\n✅ Proof found!")
                print(proof_graph.proof_trace())
            else:
                print("\n❌ Proof NOT found.")
                # Print the deepest path tried
                deepest = max(proof_graph.nodes.values(), key=lambda n: n.depth)
                path = proof_graph.tactic_path(deepest.id)
                print(f"\nDeepest node: {deepest.id} (depth {deepest.depth})")
                print(f"Goal: {deepest.goal.expression}")
                if path:
                    print(f"\nTactic path ({len(path)} steps):")
                    for i, step in enumerate(path):
                        print(f"  {i+1}. {step}")
                else:
                    print("  (root — no tactics applied)")
                print(f"\nStatus: {deepest.status}")
                if deepest.note:
                    print(f"Note: {deepest.note}")
                print(proof_graph.summary())
                plot_hypergraph(proof_graph)
        except Exception as e:
            print(f"An error occurred during proof search: {e}")
        finally:
            if dts_state_output and hybrid_reasoner.dts_sampler is not None:
                dts_state_output.parent.mkdir(parents=True, exist_ok=True)
                hybrid_reasoner.dts_sampler.save_to(str(dts_state_output))


if __name__ == "__main__":
    _argument_selection_run = settings.root_dir / "gnn_inference" / "runs" / "pointer_gnn" / "best_run"
    _premise_selection_run = settings.root_dir / "gnn_inference" / "runs" / "premise_gnn" / "best_run"
    _depth_limit = settings.proof_depth
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--goal_statement", type=str, default="forall (p q: Prop), Or p q -> Or q p")
    args_parser.add_argument("--hypotheses", type=str, default="")
    # DTS parameters are optional - if not provided, config defaults are used
    args_parser.add_argument(
        "--dts-state-input",
        type=Path,
        default=None,
        help="Optional JSON file to load DTS state from.",
    )
    args_parser.add_argument(
        "--dts-state-output",
        type=Path,
        default=None,
        help="Optional JSON file to persist updated DTS state.",
    )
    args_parser.add_argument(
        "--dts-c",
        type=float,
        default=None,
        help="DTS evidence cap C.",
    )
    args_parser.add_argument(
        "--dts-random-seed",
        type=int,
        default=None,
        help="Optional RNG seed for DTS sampling.",
    )
    args_parser.add_argument(
        "--top-k-tactics",
        type=int,
        default=3,
        help="Number of top tactic candidates to try per node (default: 3).",
    )
    args_parser.add_argument("--source-root", type=Path, required=True)
    args_parser.add_argument("--pantograph-repl", type=Path, required=True)
    args = args_parser.parse_args()

    asyncio.run(main(
        config_path=_argument_selection_run / "config.json",
        tactic_model_path=_argument_selection_run / "best.pt",
        argument_model_path=_premise_selection_run / "best.pt",
        index_path=settings.root_dir / "gnn_inference" / "runs" / "lemma_index_v1",
        corpus_path=settings.root_dir / "gnn_inference" / "runs" / "lemma_corpus_v1" / "lemmas.jsonl",
        goal_statement=args.goal_statement,
        hypotheses=args.hypotheses.split(",") if args.hypotheses else None,
        depth_limit=_depth_limit,
        dts_state_input=args.dts_state_input,
        dts_state_output=args.dts_state_output,
        dts_c=args.dts_c,
        dts_random_seed=args.dts_random_seed,
        top_k_tactics=args.top_k_tactics,
        source_root=args.source_root,
        pantograph_repl=args.pantograph_repl,
    ))
