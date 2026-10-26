import asyncio
import argparse
import random
import json

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None
from pathlib import Path
from typing import List, Optional
from pantograph_async import (
    PantographClient,
    PantographEnv,
    PantographStateScope,
    TacticState,
    PantographTacticRejected,
    PantographProtocolError,
    PantographTransportError,
    PantographStateError,
)
from maths_ai.hybrid_reasoner.pantograph_adapter import execution_state_to_goals

from maths_ai.data_models.model_identity import carry_model_identity
from maths_ai.data_models.proof_components import (
    Goal,
    RankedSubgoal,
    STV,
    TacticCandidate,
)
from maths_ai.lean_replay import LeanGoalSeed, ReplayInputError, seed_from_text
from maths_ai.gnn_inference.inference_engine import GNNModelEngine
from maths_ai.pln_inference.model import PLNInference
from maths_ai.pln_inference.metta.translator.translator_modules.runner import (
    DynamicThompsonSampler,
)

from maths_ai.hybrid_reasoner.hypergraph import (
    NodeClosureReason,
    ProofHypergraph,
    ProofNode,
    SearchEndReason,
    TacticExecutor,
    TacticOutcome,
)
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    create_model_sexpr_client,
    materialize_seed,
)
from maths_ai.core.config import settings
from maths_ai.gnn_inference.atp_lean_gnn.reporting import console_print

# A Lean identifier usable as a rewrite rule or simp lemma: hypothesis names and
# dotted lemma names, but not terms like `↑13` or `?m.2235`.


def _is_lean_identifier(value: str) -> bool:
    parts = value.split(".")
    return bool(parts) and all(
        bool(core := part.rstrip("'!?")) and core.isidentifier() for part in parts
    )


# Tactics whose arguments must be a bracketed list. `rw h` is a parse error at the
# column where `[` was expected; the rule list is `rw [h]`. With no arguments these
# tactics have nothing to rewrite with and are unplayable.
_BRACKET_REQUIRED_TACTICS = frozenset(
    {
        "rw",
        "rwa",
        "rewrite",
        "simp_rw",
        "simp only",
        "erw",
    }
)

# Tactics that run bare but take a bracketed list when given lemmas.
_BRACKET_OPTIONAL_TACTICS = frozenset(
    {
        "simp",
        "simpa",
        "field_simp",
        "norm_num",
        "linarith",
        "nlinarith",
        "aesop",
    }
)


class ExpansionResult:
    TACTICS_EXECUTED = "tactics_executed"
    NO_CANDIDATES = "no_candidates"
    DEPTH_LIMIT = "depth_limit"
    ELABORATION_ERROR = "elaboration_error"
    EXTERNAL_ABORT = "external_abort"


def render_tactic_command(tactic: TacticCandidate) -> Optional[str]:
    """Render a tactic and its arguments as Lean surface syntax.

    Argument shape is per tactic, not uniform: ``rw`` needs ``rw [h]`` while
    ``apply`` needs ``apply h``. Rendering everything as space-separated arguments
    makes every ``rw`` draw a parse error before any rewriting is attempted.

    Returns ``None`` when the tactic cannot be rendered into a playable command —
    a bracket-requiring tactic with no usable rule names — so the caller can drop
    the candidate instead of sending Lean something it will certainly reject.
    """
    name = tactic.tactic_name.strip()
    arguments = [arg.rstrip(":").strip() for arg in tactic.arguments]
    arguments = [arg for arg in arguments if arg]

    if name in _BRACKET_REQUIRED_TACTICS or name in _BRACKET_OPTIONAL_TACTICS:
        # A rewrite rule must be a name Lean can look up. The policy samples nodes
        # from the goal's own graph, so a draw can land on a term like `↑13`, which
        # is a valid expression but not a rule.
        if any(not _is_lean_identifier(arg) for arg in arguments):
            return None
        rules = arguments
        if rules:
            return f"{name} [{', '.join(rules)}]"
        if name in _BRACKET_REQUIRED_TACTICS:
            return None
        return name

    if not arguments:
        return name
    return " ".join([name, *arguments])


def plot_hypergraph(graph: ProofHypergraph) -> None:
    """Utility to visualize the proof hypergraph with Graphviz (for debugging
    and analysis).

    Nodes are labeled with their goal expressions; edges are labeled with
    the tactic applied and the STV of each resulting subgoal (if any).
    """

    dot = Digraph(comment="Proof Hypergraph")
    for node_id, node in graph.nodes.items():
        label = (
            f"{node.goal.expression}\n"
            if node.stv is not None
            else node.goal.expression
        )
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
            decoded into canonical ``maths_ai`` goals containing pretty Lean text,
            normalized model S-expressions, ordered locals, and a separate case tag.
            On a Lean-side error (the tactic doesn't apply), returns
            ``TacticOutcome(success=False, error=...)``.
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
        try:
            subgoals = execution_state_to_goals(new_state)
            return TacticOutcome(success=True, subgoals=subgoals, error=None)
        finally:
            if scope.is_usable:
                await scope.release(new_state)


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
      4. rank every returned subgoal and link all Lean obligations into the hypergraph

    Each link triggers ``ProofHypergraph``'s bottom-up propagation, which is
    objective 2: PLN-derived ranks continuously update the GNN-seeded scores
    of every ancestor, all the way back to the root.
    """

    def __init__(
        self,
        config_path: Path,
        tactic_model_path: Path,
        retrieval_config_path: Path,
        *,
        executor: PantographExecutor,
        top_k_tactics: int = 3,
        max_depth: int = 10,
        max_nodes: int = 500,
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
            retrieval_config_path=retrieval_config_path,
        )
        self.atomic_tactics = {}

        # Always use Thompson sampling as fallback - it's automatic and internal
        self.pln_fallback_strategy = "thompson"

        # Use config defaults if not provided
        if dts_c is None:
            dts_c = settings.dts_default_c
        if dts_random_seed is None:
            dts_random_seed = settings.dts_default_seed

        # _dts_rng is cheap; construct it unconditionally so subclasses and other
        # callers can always reference it without an is-None check.
        self._dts_rng = random.Random(dts_random_seed)

        if not use_pln:
            # PLN disabled: no petta subprocess will ever be spawned.
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
                        # If loading fails, create fresh sampler
                        self.dts_sampler = DynamicThompsonSampler(C=dts_c)
                else:
                    # Create new sampler
                    self.dts_sampler = DynamicThompsonSampler(C=dts_c)

        self.executor = executor
        self.server = executor.server

        self.top_k_tactics = top_k_tactics
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self._cached_root_state: TacticState | None = None
        self._cached_root_fingerprint: str | None = None

    # GNN side
    def _build_gnn_engine(
        self,
        *,
        config_path: Path,
        tactic_model_path: Path,
        retrieval_config_path: Path,
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
            retrieval_config_path=retrieval_config_path,
        )

    def predict_next_tactic(self, sub_goal: Goal) -> List[TacticCandidate]:
        """
        Args:
            sub_goal: the sanitized target sub_goal (expression plus its
                local hypotheses) for which tactics are predicted. The
                base engine and RL subclass consume the full canonical goal so sampled
                arguments resolve against the same DAG the encoder saw.
        Returns:
            up to `top_k_tactics` TacticCandidate(tactic_name, arguments, probability),
            ranked by predicted probability, descending.

            An empty list means the GNN found no viable tactic for this
            goal (see GNNModelEngine.inference's "degenerate prediction"
            edge case) — callers must treat that as a dead branch, which
            `_expand` below does via `graph.mark_node_exhausted`.
        """
        return self.gnn_engine.inference(sub_goal, top_k=self.top_k_tactics)

    # PLN side
    def _make_dts_key(
        self, parent_goal: str, tactic: TacticCandidate, subgoal: Goal
    ) -> str:
        """Create a meaningful DTS state key that identifies the subgoal in context.

        The key includes:
        - Parent goal expression
        - Tactic name and arguments (what was applied to get here)
        - Subgoal expression

        This makes the DTS state more interpretable and allows tracking
        which tactics lead to which subgoals.
        """
        tactic_str = (
            f"{tactic.tactic_name}({' '.join(tactic.arguments)})"
            if tactic.arguments
            else tactic.tactic_name
        )
        # Truncate long expressions to keep keys manageable
        max_len = 50
        parent_trunc = (
            parent_goal[:max_len] + "..." if len(parent_goal) > max_len else parent_goal
        )
        subgoal_trunc = (
            subgoal.expression[:max_len] + "..."
            if len(subgoal.expression) > max_len
            else subgoal.expression
        )
        return f"{parent_trunc} --[{tactic_str}]--> {subgoal_trunc}"

    async def rank_subgoals(
        self,
        goal: str,
        sub_goals: List[Goal],
        tactic: TacticCandidate,
        *,
        gnn_probability: float = 1.0,
    ) -> List[RankedSubgoal]:
        if self.petta_chainer is None:
            raise RuntimeError(
                "rank_subgoals requires PLN; the reasoner was constructed with use_pln=False"
            )
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
            elif (
                self.pln_fallback_strategy == "thompson"
                and self.dts_sampler is not None
            ):
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

    async def prove(self, seed: LeanGoalSeed) -> ProofHypergraph:
        if self._proving:
            raise PantographStateError("A reasoner supports only one active search.")
        self._proving = True
        try:
            async with self.server.state_scope() as scope:
                self._scope = scope
                return await self._prove_scoped(seed)
        finally:
            self._scope = None
            self._cached_root_state = None
            self._cached_root_fingerprint = None
            self._proving = False

    async def _prove_scoped(self, seed: LeanGoalSeed) -> ProofHypergraph:
        """Run best-first AND-OR search over the hypergraph rooted at ``seed``.

        A seed that cannot be replayed faithfully raises ``ReplayInputError``
        before any search; that is a property of the input, not a failed proof.

        Loop: pop the highest-``combined_rank`` open node, expand it
        (``_expand``), which links any new subgoals into the hypergraph —
        and every link immediately backpropagates updated status/rank to
        every ancestor (``ProofHypergraph._propagate``), re-ordering the
        frontier for the next iteration.

        Termination (design-report "no-goal ambiguity" — resolved here as):
          * root SOLVED  → proof found; ``graph.proof_trace()`` replays it
          * root DEAD    → structurally closed; closure provenance determines whether
            learning sees failure or unknown evidence
          * frontier empty / ``max_nodes`` reached → budget exhaustion
            (open design question: what to return — we return the partial
            graph so the caller can inspect ``graph.frontier()``, resume, or
            visualize it; see ``ProofHypergraph.summary``)

        ``max_depth`` bounds branch depth and ``max_nodes`` bounds total
        graph size — the design report's "branching-factor explosion"
        safeguards. Cycle detection (a subgoal identical to one of its own
        ancestors) is handled inside ``ProofHypergraph.add_edge``.
        """
        root_state, root_goal = await self._materialize_goal(seed)
        graph = ProofHypergraph(root_goal)
        self._cached_root_state = root_state
        self._cached_root_fingerprint = root_goal.state_fingerprint()
        loop_count = 0

        # Running through the loop untill the theorem is solved or the depth_limit is reached
        try:
            while (
                not graph.is_solved()
                and not graph.is_exhausted()
                and graph.search_end_reason != SearchEndReason.EXTERNAL_ABORT
                and len(graph.nodes) < self.max_nodes
            ):
                frontier = graph.frontier()
                if not frontier:
                    graph.search_end_reason = SearchEndReason.FRONTIER_EMPTY
                    break
                loop_count += 1
                node = frontier[0]
                print(
                    f"\n=== Loop {loop_count}: expanding node {node.id} (depth {node.depth}) | goal: {node.goal.expression} ==="
                )
                await self._expand(graph, node)
        except BaseException:
            graph.search_end_reason = SearchEndReason.EXTERNAL_ABORT
            raise

        if graph.is_solved():
            graph.search_end_reason = SearchEndReason.ROOT_SOLVED
        elif graph.search_end_reason == SearchEndReason.EXTERNAL_ABORT:
            pass
        elif graph.is_exhausted():
            graph.search_end_reason = SearchEndReason.ROOT_DEAD
        elif len(graph.nodes) >= self.max_nodes:
            graph.search_end_reason = SearchEndReason.MAX_NODES
        elif graph.search_end_reason is None:
            graph.search_end_reason = SearchEndReason.FRONTIER_EMPTY
        return graph

    async def _materialize_goal(
        self, goal: Goal | LeanGoalSeed
    ) -> tuple[TacticState, Goal]:
        """Replay a seed or a canonical goal and return the checked live goal."""
        if self._scope is None:
            raise PantographStateError(
                "Materialization requires an active search scope."
            )
        seed = goal.replay_seed() if isinstance(goal, Goal) else goal
        state, materialized, _ = await materialize_seed(self._scope, seed)
        return state, materialized

    def _link(
        self,
        graph: ProofHypergraph,
        node: ProofNode,
        tactic: TacticCandidate,
        ranked_subgoals: list,
    ):
        """Link one successful tactic application into the hypergraph and
        return the created hyperedge (or ``None`` when ``add_edge`` refuses
        the link, e.g. on cycle detection).

        This is the single seam between search and training data collection:
        ``RLHybridReasoner`` overrides it to associate the returned edge id
        with the sampled action indices that produced ``tactic``, so the
        train phase can recompute log-probabilities for exactly the edges
        that made it into the graph.
        """
        return graph.add_edge(node.id, tactic, ranked_subgoals=ranked_subgoals)

    def _on_expansion_complete(self, node: ProofNode, result: str) -> None:
        """Hook invoked exactly once for every non-exceptional expansion exit."""

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

        try:
            fingerprint = node.goal.state_fingerprint()
            if (
                node.id == graph.root_id
                and self._cached_root_state is not None
                and self._cached_root_fingerprint == fingerprint
            ):
                state = self._cached_root_state
                materialized = node.goal
                self._cached_root_state = None
                self._cached_root_fingerprint = None
            else:
                state, materialized = await self._materialize_goal(node.goal)
            node.goal = materialized
        except ReplayInputError as exc:
            graph.mark_node_unelaborated(node.id, note=f"elaboration error: {exc}")
            self._on_expansion_complete(node, ExpansionResult.ELABORATION_ERROR)
            return
        except (
            PantographProtocolError,
            PantographTransportError,
            PantographStateError,
        ) as exc:
            graph.mark_node_infrastructure_failed(node.id, note=str(exc))
            graph.search_end_reason = SearchEndReason.EXTERNAL_ABORT
            raise
        try:
            await self._expand_materialized(graph, node, state)
        finally:
            if self._scope is not None and self._scope.is_usable:
                await self._scope.release(state)

    async def prepare_goal(self, goal):
        """Prepare state-dependent proposal inputs before synchronous model scoring."""
        if self.gnn_engine is not None:
            await self.gnn_engine.prepare_goal(goal)

    async def _expand_materialized(
        self, graph: ProofHypergraph, node: ProofNode, state: TacticState
    ) -> None:
        print(f"  [GNN Input] goal={node.goal.expression}  hyps={node.goal.hypotheses}")
        await self.prepare_goal(node.goal)
        candidates = self.predict_next_tactic(node.goal)
        if not candidates:
            graph.mark_node_exhausted(
                node.id,
                reason=NodeClosureReason.NO_CANDIDATES,
                note="GNN returned no viable tactic",
            )
            self._on_expansion_complete(node, ExpansionResult.NO_CANDIDATES)
            return
        any_applied = False

        for tactic in candidates:
            try:
                outcome = await self.executor.apply(self._scope, state, tactic)
            except (
                PantographProtocolError,
                PantographTransportError,
                PantographStateError,
            ) as exc:
                graph.mark_node_infrastructure_failed(
                    node.id,
                    note=f"Pantograph failed during tactic execution: {exc}",
                )
                graph.search_end_reason = SearchEndReason.EXTERNAL_ABORT
                self._on_expansion_complete(node, ExpansionResult.EXTERNAL_ABORT)
                exc.graph = graph
                raise

            if not outcome.success:
                print(
                    f"  [Tactic FAILED] {tactic.tactic_name} {' '.join(tactic.arguments)} — {outcome.error}"
                )
                continue
            any_applied = True
            print(
                f"  [Tactic APPLIED] {tactic.tactic_name} {' '.join(tactic.arguments)} (prob={tactic.probability:.4f})"
            )
            # Successors see the model identity of the goal they came from.
            subgoals = [carry_model_identity(node.goal, goal) for goal in outcome.subgoals]

            if not subgoals:
                # "no-goal": this tactic fully discharges the goal (QED for this branch)
                print(f"  [Tactic QED] no subgoals — branch closed!")
                self._link(graph, node, tactic, ranked_subgoals=[])
                continue

            if self.use_pln:
                print(f"  [PLN Ranking] scoring {len(subgoals)} subgoal(s)...")
                ranked = await self.rank_subgoals(
                    node.goal.expression,
                    subgoals,
                    tactic,
                    gnn_probability=tactic.probability,
                )
                print(f"  [PLN Done] ranked {len(ranked)} subgoal(s)")
                for i, rs in enumerate(ranked):
                    print(
                        f"    subgoal {i}: {rs.goal.expression} | stv=({rs.stv.strength:.3f}, {rs.stv.confidence:.3f}) | combined_rank={rs.combined_rank:.4f}"
                    )
                self._link(
                    graph,
                    node,
                    tactic,
                    ranked_subgoals=[
                        (candidate.goal, candidate.stv) for candidate in ranked
                    ],
                )
            else:
                # PLN disabled: retain every Lean obligation in executor order.
                self._link(
                    graph,
                    node,
                    tactic,
                    ranked_subgoals=[(subgoal, None) for subgoal in subgoals],
                )

        if not any_applied and self.use_pln:
            print(
                f"  [PLN Fallback] evaluating goal: {node.goal.expression}  hyps: {node.goal.hypotheses}"
            )
            pln_result = await self.petta_chainer.evaluate_async(
                node.goal.expression,
                hypotheses=[*node.goal.hypotheses],
            )

            stv = pln_result.stv
            pln_fb_key = (
                f"PLN_fb::{node.goal.expression}::{'|'.join(node.goal.hypotheses)}"
            )

            if pln_result.is_fallback and self.dts_sampler is not None:
                sampled = self.dts_sampler.sample(pln_fb_key, self._dts_rng)
                stv = STV(strength=sampled, confidence=1.0)
                print(
                    f"  [PLN Fallback] DTS override: sampled={sampled:.3f} (was random fallback)"
                )
            elif self.dts_sampler is not None:
                self.dts_sampler.record_observation(pln_fb_key, reward=stv.score)
                print(
                    f"  [PLN Fallback] DTS recorded observation: score={stv.score:.3f}"
                )

            print(
                f"  [PLN Fallback] final STV=({stv.strength:.3f}, {stv.confidence:.3f})  score={stv.score:.3f}  is_fallback={pln_result.is_fallback}"
            )
            # PLN is a ranking and reward heuristic. Only Lean QED may close a node.

        note = None if any_applied else "executor rejected every candidate tactic"
        if note:
            print(f"  [Node {node.id} EXHAUSTED] {note}")
        graph.mark_node_exhausted(
            node.id,
            reason=NodeClosureReason.CANDIDATES_EXHAUSTED,
            note=note,
        )
        self._on_expansion_complete(node, ExpansionResult.TACTICS_EXECUTED)


async def main(
    config_path: Path,
    tactic_model_path: Path,
    retrieval_config_path: Path,
    *,
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
            retrieval_config_path=retrieval_config_path,
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
            if hybrid_reasoner.gnn_engine is not None:
                hybrid_reasoner.gnn_engine.provider.verify_environment(env)
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
            if hybrid_reasoner.gnn_engine is not None:
                await hybrid_reasoner.gnn_engine.aclose()
            await server.aclose()


if __name__ == "__main__":
    _argument_selection_run = (
        settings.root_dir / "gnn_inference" / "runs" / "pointer_gnn" / "best_run"
    )
    _premise_selection_run = (
        settings.root_dir / "gnn_inference" / "runs" / "premise_gnn" / "best_run"
    )
    _depth_limit = settings.proof_depth
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--goal_statement", type=str, default="forall (p q: Prop), Or p q -> Or q p"
    )
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

    asyncio.run(
        main(
            config_path=_argument_selection_run / "config.json",
            tactic_model_path=_argument_selection_run / "best.pt",
            retrieval_config_path=settings.root_dir / "gnn_inference" / "configs" / "retrieval.example.json",
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
        )
    )
