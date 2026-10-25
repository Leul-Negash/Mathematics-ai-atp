"""MCTS and best-first search over the installed client's scoped execution API.

A recording scoped client stands in for the REPL: it rejects operations on
released states the way the real scope does, and records every allocation and
release, so these tests check ownership as well as search results. The
``pantograph_live`` cases run the same search against the actual executable.
"""

from __future__ import annotations

import asyncio
import itertools
import unittest
from types import MethodType

import pytest
import torch

from maths_ai.lean_replay import seed_from_text
from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import extract_critic_samples
from maths_ai.gnn_inference.tests.model_helpers import (
    actor_critic,
    pantograph_goal,
    structured_goal,
)
from maths_ai.hybrid_reasoner.hypergraph import (
    BackupSource,
    NodeClosureReason,
    NodeStatus,
    SearchEndReason,
)
from maths_ai.hybrid_reasoner.joint_inference import PantographExecutor
from maths_ai.hybrid_reasoner.mcts_diagnostics import (
    verify_learning_round,
    verify_mcts_search,
    verify_shared_device_workers,
)
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
from pantograph_async import (
    PantographGoalRejected,
    PantographProtocolError,
    PantographStateError,
    PantographTacticRejected,
    PantographTransportError,
)


TACTIC_VOCAB = {"trivial": 0, "intro": 1, "exact": 2}
ROOT = "p → p"
HYPS = ["p : Prop"]
SEED = seed_from_text(goal=ROOT, hypotheses=HYPS)


class _State:
    _ids = itertools.count()

    def __init__(self, goals, *, expression=None):
        # A stable identity: id() values are reused once states are freed.
        self.sid = next(self._ids)
        self.goals = [pantograph_goal(goal) for goal in goals]
        self.expression = expression


class _RecordingScope:
    def __init__(self, client):
        self.client = client
        self.live: dict[int, _State] = {}
        self.is_usable = True

    async def __aenter__(self):
        self.client.scopes.append(self)
        return self

    async def __aexit__(self, *_exc):
        self.client.left_at_exit.append(len(self.live))
        self.live.clear()
        self.is_usable = False

    def _allocate(self, state):
        self.live[state.sid] = state
        self.client.events.append(("allocate", state.sid))
        return state

    async def start_goal(self, expression, *, levels=()):
        self.client.started_levels = tuple(levels)
        if not self.is_usable:
            raise PantographStateError("State operations require an active scope.")
        if expression in self.client.unelaborated:
            raise PantographGoalRejected(f"cannot elaborate {expression}")
        if expression in self.client.protocol_failures:
            raise PantographProtocolError("malformed goal.start response")
        return self._allocate(_State([], expression=expression))

    async def apply_tactic(self, state, tactic, *, goal_index=0):
        if self.live.get(state.sid) is not state:
            raise PantographStateError("Cannot apply a tactic to a released state.")
        if state.expression is not None:
            # Replay of a start state: serialize the node's canonical goal.
            return self._allocate(_State([self.client.goals[state.expression]]))
        node_id = self.client.current_node
        self.client.events.append(("apply", node_id, state.sid, tactic))
        await self.client.before_apply(node_id)
        outcome = self.client.outcome_for(node_id, state.goals[0].target.pp)
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome == "reject":
            raise PantographTacticRejected(f"{tactic} failed")
        subgoals = [structured_goal(expression, HYPS) for expression in outcome]
        for subgoal in subgoals:
            self.client.goals[subgoal.replay_spec().expression] = subgoal
        return self._allocate(_State(subgoals))

    async def release(self, state):
        if self.live.pop(state.sid, None) is not None:
            self.client.events.append(("release", state.sid))


class _RecordingClient:
    is_usable = True

    def __init__(self, outcomes=None):
        root = structured_goal(ROOT, HYPS)
        self.goals = {"∀ (p : Prop), p → p": root}
        self.outcomes = outcomes or {}
        self.unelaborated: set[str] = set()
        self.protocol_failures: set[str] = set()
        self.current_node: int | None = None
        self.events: list[tuple] = []
        self.scopes: list[_RecordingScope] = []
        self.left_at_exit: list[int] = []
        self.gate: asyncio.Event | None = None
        self.entered_apply: asyncio.Event | None = None

    def state_scope(self):
        return _RecordingScope(self)

    def outcome_for(self, node_id, expression):
        """Script by node id first, then by goal text; unscripted goals close."""
        if node_id in self.outcomes:
            return self.outcomes[node_id]
        return self.outcomes.get(expression, [])

    async def before_apply(self, node_id):
        if self.gate is not None:
            self.entered_apply.set()
            await self.gate.wait()

    async def aclose(self):
        self.is_usable = False


class _NodeTrackingReasoner(RLHybridReasoner):
    """Tell the fake which graph node an application belongs to."""

    async def _execute_and_link(self, graph, node, candidates, *, materialized):
        self.server.current_node = node.id
        self.executed.append((node.id, len(candidates)))
        return await super()._execute_and_link(
            graph, node, candidates, materialized=materialized
        )

    def predict_next_tactics_for_nodes(self, nodes):
        self.proposal_batches.append([node.id for node in nodes])
        return super().predict_next_tactics_for_nodes(nodes)


def _reasoner(client, *, top_k=2, seed=0, **search):
    torch.manual_seed(seed)
    node_vocab = build_vocab([model_goal_to_dag(structured_goal(ROOT, HYPS))])
    model = actor_critic(len(node_vocab), len(TACTIC_VOCAB))
    reasoner = _NodeTrackingReasoner(
        model,
        node_vocab,
        TACTIC_VOCAB,
        executor=PantographExecutor(client),
        top_k_tactics=top_k,
        max_depth=3,
        max_nodes=20,
        use_pln=False,
        **search,
    )
    reasoner.executed = []
    reasoner.proposal_batches = []
    return reasoner


def _puct(**overrides):
    settings = dict(selection_policy="puct", num_simulations=6, sim_batch_size=2)
    settings.update(overrides)
    return settings


def _assert_scoped_cleanly(test, client):
    """Every allocation was released by the search or its scope."""
    test.assertEqual(len(client.scopes), 1)
    test.assertFalse(client.scopes[0].live)
    test.assertFalse(client.scopes[0].is_usable)


def _assert_alternatives_released_promptly(test, client):
    """Each alternative's result is released before the next application."""
    live_results: set[int] = set()
    applied = False
    for event in client.events:
        if event[0] == "apply":
            test.assertFalse(live_results, "an alternative result outlived its use")
            applied = True
        elif event[0] == "allocate" and applied:
            live_results.add(event[1])
            applied = False
        elif event[0] == "release":
            live_results.discard(event[1])


class SearchModeOwnershipTests(unittest.TestCase):
    def _assert_owned_execution(self, client, reasoner, result):
        _assert_scoped_cleanly(self, client)
        _assert_alternatives_released_promptly(self, client)
        self.assertEqual(client.left_at_exit, [0])
        self.assertFalse(reasoner._proving)
        self.assertIsNone(reasoner._scope)
        self.assertIsNone(reasoner._cached_root_materialization)
        self.assertEqual(reasoner._pending, {})
        # Graph nodes keep canonical goals only.
        for node in result.graph.nodes.values():
            node.goal.require_model_state()

    def test_best_first_uses_scoped_execution(self):
        client = _RecordingClient({0: ["A", "B"]})
        reasoner = _reasoner(client)
        result = asyncio.run(reasoner.prove(SEED))
        self.assertTrue(result.graph.is_solved())
        self._assert_owned_execution(client, reasoner, result)
        for edge in result.graph.edges.values():
            self.assertEqual(edge.visit_stats.N, 0)

    def test_puct_uses_scoped_execution(self):
        client = _RecordingClient({0: ["A", "B"]})
        reasoner = _reasoner(client, **_puct())
        result = asyncio.run(reasoner.prove(SEED))
        self.assertTrue(result.graph.is_solved())
        self._assert_owned_execution(client, reasoner, result)
        self.assertTrue(any(e.visit_stats.N for e in result.graph.edges.values()))
        for edge in result.graph.edges.values():
            self.assertEqual(edge.visit_stats.virtual_loss, 0)

    def test_multi_leaf_batch_keeps_each_leaf_state_until_all_alternatives_finish(self):
        client = _RecordingClient({0: ["A", "B"], "A": "reject"})
        reasoner = _reasoner(client, top_k=3, **_puct())
        asyncio.run(reasoner.prove(SEED))

        # One policy forward proposed several leaves together.
        self.assertTrue(any(len(batch) > 1 for batch in reasoner.proposal_batches))
        applies: dict[int, list[int]] = {}
        for event in client.events:
            if event[0] == "apply":
                applies.setdefault(event[1], []).append(event[2])
        leaves = [node_id for node_id in applies if node_id != 0]
        self.assertTrue(leaves)
        self.assertTrue(any(len(applies[node_id]) > 1 for node_id in leaves))
        for node_id in leaves:
            # Every alternative of a leaf ran against one live leaf state,
            # which was released only after the last of them finished.
            self.assertEqual(len(applies[node_id]), dict(reasoner.executed)[node_id])
            self.assertEqual(len(set(applies[node_id])), 1)
            last_apply = max(
                index
                for index, event in enumerate(client.events)
                if event[0] == "apply" and event[1] == node_id
            )
            released = client.events.index(("release", applies[node_id][0]))
            self.assertGreater(released, last_apply)
        _assert_scoped_cleanly(self, client)
        _assert_alternatives_released_promptly(self, client)

    def test_identical_goal_text_in_different_nodes_keeps_node_action_alignment(self):
        # Every root alternative yields two copies of the same goal. The first
        # copy executed can be closed; every other copy rejects all tactics.
        client = _RecordingClient({0: ["A", "A"]})
        first_copy: list[int] = []

        def outcome_for(node_id, expression):
            if node_id == 0:
                return ["A", "A"]
            if not first_copy:
                first_copy.append(node_id)
            return [] if node_id == first_copy[0] else "reject"

        client.outcome_for = outcome_for
        reasoner = _reasoner(client, top_k=3, **_puct(num_simulations=4))
        result = asyncio.run(reasoner.prove(SEED))
        graph = result.graph

        executed = dict(reasoner.executed)
        copies = [node_id for node_id in executed if node_id != 0]
        self.assertGreater(len(copies), 1)
        fingerprints = {graph.nodes[node_id].goal.state_fingerprint() for node_id in copies}
        self.assertEqual(len(fingerprints), 1)
        solved, rejected = first_copy[0], [n for n in copies if n != first_copy[0]]
        self.assertEqual(graph.nodes[solved].status, NodeStatus.SOLVED)
        for node_id in rejected:
            self.assertEqual(graph.nodes[node_id].status, NodeStatus.DEAD)

        # Accepted actions belong to the solved copy's edges (or the root's),
        # and sampled-action multiplicity is kept per node.
        solved_edges = set(graph.nodes[solved].outgoing_edge_ids)
        root_edges = set(graph.nodes[0].outgoing_edge_ids)
        self.assertLessEqual(set(result.edge_actions), solved_edges | root_edges)
        self.assertEqual(
            sum(
                action.multiplicity
                for edge_id, action in result.edge_actions.items()
                if edge_id in solved_edges
            ),
            reasoner.top_k_tactics,
        )
        self.assertEqual(
            sum(record.action.multiplicity for record in result.failure_actions),
            reasoner.top_k_tactics * len(rejected),
        )
        fingerprint = str(reasoner.dag_featurize(graph.nodes[solved].goal)[1].graph_fingerprint)
        for record in result.failure_actions:
            self.assertEqual(record.action.graph_fingerprint, fingerprint)
        _assert_scoped_cleanly(self, client)


class _SkippingReasoner(_NodeTrackingReasoner):
    """Propose nothing for goal ``B``; sample normally for every other node."""

    def predict_next_tactics_for_nodes(self, nodes):
        self.leaf_batches.append([node.id for node in nodes])
        sampled = iter(
            super().predict_next_tactics_for_nodes(
                [node for node in nodes if node.goal.expression != "B"]
            )
        )
        return [[] if node.goal.expression == "B" else next(sampled) for node in nodes]


class BatchCompositionTests(unittest.TestCase):
    def test_mixed_batch_releases_every_input_and_settles_each_node(self):
        # A cannot be re-elaborated, B receives no candidates, C closes.
        client = _RecordingClient({0: ["A", "B", "C"]})
        client.unelaborated.add(structured_goal("A", HYPS).replay_spec().expression)
        torch.manual_seed(0)
        node_vocab = build_vocab([model_goal_to_dag(structured_goal(ROOT, HYPS))])
        reasoner = _SkippingReasoner(
            actor_critic(len(node_vocab), len(TACTIC_VOCAB)),
            node_vocab,
            TACTIC_VOCAB,
            executor=PantographExecutor(client),
            top_k_tactics=1,
            max_depth=3,
            max_nodes=20,
            use_pln=False,
            **_puct(num_simulations=4),
        )
        reasoner.executed, reasoner.proposal_batches, reasoner.leaf_batches = [], [], []
        result = asyncio.run(reasoner.prove(SEED))
        graph = result.graph

        a, b, c = (graph.nodes[child] for child in graph.edges[0].child_ids)
        # A failed to materialize, so the batch proposal held B and C only.
        self.assertIn({b.id, c.id}, [set(batch) for batch in reasoner.leaf_batches])
        self.assertEqual(a.closure_reason, NodeClosureReason.ELABORATION_ERROR)
        self.assertEqual(b.closure_reason, NodeClosureReason.NO_CANDIDATES)
        self.assertEqual(c.status, NodeStatus.SOLVED)
        # Only executed alternatives settle as evidence: C's accepted edge.
        self.assertEqual(result.failure_actions, [])
        self.assertTrue(set(result.edge_actions) <= {0, *c.outgoing_edge_ids})
        # Every materialized input, including B's unused one, was released by
        # the search itself; nothing was left for scope exit.
        self.assertEqual(client.left_at_exit, [0])
        _assert_scoped_cleanly(self, client)

    def test_depth_limited_leaves_are_closed_without_materialization(self):
        client = _RecordingClient({0: ["A", "B"]})
        reasoner = _reasoner(client, **_puct(num_simulations=4))
        reasoner.max_depth = 1
        result = asyncio.run(reasoner.prove(SEED))
        leaves = [
            node for node in result.graph.nodes.values() if node.id != result.graph.root_id
        ]
        self.assertTrue(leaves)
        for node in leaves:
            self.assertEqual(node.closure_reason, NodeClosureReason.DEPTH_LIMIT)
        # Only the root was ever replayed: one start plus its serialization.
        self.assertEqual(
            sum(1 for event in client.events if event[0] == "allocate") - sum(
                1 for event in client.events if event[0] == "apply"
            ),
            2,
        )
        self.assertEqual([node_id for node_id, _ in reasoner.executed], [0])
        self.assertTrue(all(record.goal.expression == ROOT for record in result.failure_actions))
        _assert_scoped_cleanly(self, client)


class OverlappingSearchTests(unittest.TestCase):
    def test_overlapping_search_is_rejected_before_touching_the_running_search(self):
        client = _RecordingClient({0: []})
        reasoner = _reasoner(client, **_puct())

        async def run():
            client.gate = asyncio.Event()
            client.entered_apply = asyncio.Event()
            first = asyncio.create_task(
                reasoner.prove(SEED, greedy=True)
            )
            await client.entered_apply.wait()
            pending, result, greedy = reasoner._pending, reasoner._result, reasoner._greedy
            pending_snapshot = {key: dict(value.actions) for key, value in pending.items()}
            with self.assertRaises(PantographStateError):
                await reasoner.prove(SEED)
            self.assertIs(reasoner._pending, pending)
            self.assertEqual(
                {key: dict(value.actions) for key, value in pending.items()},
                pending_snapshot,
            )
            self.assertIs(reasoner._result, result)
            self.assertIs(reasoner._greedy, greedy)
            self.assertEqual(len(client.scopes), 1)
            client.gate.set()
            return await first

        result = asyncio.run(run())
        self.assertTrue(result.graph.is_solved())
        _assert_scoped_cleanly(self, client)


class FailureInjectionTests(unittest.TestCase):
    def _capture_graph(self, reasoner):
        captured = {}
        original = reasoner._prove_mcts

        async def capture(self, graph, deadline):
            captured["graph"] = graph
            return await original(graph, deadline)

        reasoner._prove_mcts = MethodType(capture, reasoner)
        return captured

    def _assert_released(self, reasoner, client, graph):
        for edge in graph.edges.values():
            self.assertEqual(edge.visit_stats.virtual_loss, 0)
        self.assertEqual(reasoner._pending, {})
        self.assertIsNone(reasoner._result)
        self.assertFalse(reasoner._proving)
        self.assertFalse(reasoner._greedy)
        _assert_scoped_cleanly(self, client)

    def test_cancellation_during_leaf_execution_releases_everything(self):
        # One draw per node gives the root one edge, so both simulations of a
        # batch traverse it and it holds two virtual losses at once.
        client = _RecordingClient({0: ["A", "B"]})
        reasoner = _reasoner(client, top_k=1, **_puct())
        captured = self._capture_graph(reasoner)

        async def run():
            client.gate = None
            first_round = asyncio.Event()
            original = client.before_apply

            async def block_on_leaf(node_id):
                if node_id != 0:
                    first_round.set()
                    await asyncio.Event().wait()
                await original(node_id)

            client.before_apply = block_on_leaf
            task = asyncio.create_task(reasoner.prove(SEED))
            await first_round.wait()
            graph = captured["graph"]
            self.assertEqual(
                max(edge.visit_stats.virtual_loss for edge in graph.edges.values()), 2
            )
            self.assertTrue(reasoner._pending)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        graph = captured["graph"]
        self.assertEqual(graph.end_reason, SearchEndReason.EXTERNAL_ABORT)
        self._assert_released(reasoner, client, graph)

    def test_transport_failure_aborts_without_fabricated_failure_records(self):
        client = _RecordingClient(
            {0: ["A", "B"], 1: PantographTransportError("process exited")}
        )
        reasoner = _reasoner(client, **_puct())
        captured = self._capture_graph(reasoner)
        with self.assertRaises(PantographTransportError):
            asyncio.run(reasoner.prove(SEED))
        graph = captured["graph"]
        self.assertEqual(graph.end_reason, SearchEndReason.EXTERNAL_ABORT)
        self.assertEqual(graph.nodes[1].closure_reason, NodeClosureReason.EXTERNAL_ABORT)
        self.assertEqual(
            [s for s in extract_critic_samples(graph) if s.node_id == 1], []
        )
        self._assert_released(reasoner, client, graph)

    def test_protocol_failure_during_materialization_propagates(self):
        client = _RecordingClient({0: ["A", "B"]})
        client.protocol_failures.add(structured_goal("A", HYPS).replay_spec().expression)
        reasoner = _reasoner(client, **_puct())
        captured = self._capture_graph(reasoner)
        with self.assertRaises(PantographProtocolError):
            asyncio.run(reasoner.prove(SEED))
        graph = captured["graph"]
        self.assertEqual(graph.nodes[1].closure_reason, NodeClosureReason.EXTERNAL_ABORT)
        self._assert_released(reasoner, client, graph)

    def test_elaboration_failure_rejection_and_unknown_backup_stay_distinct(self):
        # A cannot be re-elaborated; every tactic on B is rejected by Lean.
        client = _RecordingClient({0: ["A", "B"], "B": "reject"})
        client.unelaborated.add(structured_goal("A", HYPS).replay_spec().expression)
        reasoner = _reasoner(client, top_k=1, **_puct(num_simulations=4))
        result = asyncio.run(reasoner.prove(SEED))
        graph = result.graph

        self.assertEqual(len(graph.nodes[0].outgoing_edge_ids), 1)
        root_edge = graph.edges[graph.nodes[0].outgoing_edge_ids[0]]
        elaboration, rejected = (graph.nodes[child] for child in root_edge.child_ids)
        self.assertEqual(elaboration.closure_reason, NodeClosureReason.ELABORATION_ERROR)
        self.assertEqual(rejected.closure_reason, NodeClosureReason.CANDIDATES_EXHAUSTED)
        # Ordinary rejection produces actor failure records for B only; the
        # unelaborated node never sampled or executed anything.
        self.assertTrue(result.failure_actions)
        for record in result.failure_actions:
            self.assertEqual(record.goal.state_fingerprint(), rejected.goal.state_fingerprint())
        self.assertNotIn(elaboration.id, dict(reasoner.executed))
        # The AND-backup over (unknown, valid 0.0) is a valid 0.0: completed
        # visits count, and only valid numeric evidence enters W.
        self.assertGreaterEqual(root_edge.visit_stats.N, 1)
        self.assertEqual(root_edge.visit_stats.N, root_edge.visit_stats.N_v)
        self.assertEqual(root_edge.visit_stats.W, 0.0)
        # Learning: B is a valid search failure, A has no critic label.
        samples = {sample.node_id: sample for sample in extract_critic_samples(graph)}
        self.assertNotIn(elaboration.id, samples)
        self.assertEqual(samples[rejected.id].target, 0.0)
        self.assertEqual(samples[rejected.id].source, BackupSource.SEARCH_FAILURE)
        _assert_scoped_cleanly(self, client)

    def test_unknown_backup_counts_the_visit_without_numeric_evidence(self):
        client = _RecordingClient({0: ["A"]})
        client.unelaborated.add(structured_goal("A", HYPS).replay_spec().expression)
        reasoner = _reasoner(client, **_puct(num_simulations=2, sim_batch_size=1))
        result = asyncio.run(reasoner.prove(SEED))
        root_edge = result.graph.edges[result.graph.nodes[0].outgoing_edge_ids[0]]
        self.assertEqual(root_edge.visit_stats.N, 1)
        self.assertEqual(root_edge.visit_stats.N_v, 0)
        self.assertEqual(root_edge.visit_stats.W, 0.0)
        self.assertEqual(result.failure_actions, [])
        _assert_scoped_cleanly(self, client)


class CooperativeDeadlineTests(unittest.TestCase):
    def test_deadline_after_first_batch_returns_completed_experience(self):
        client = _RecordingClient({0: ["A", "B"]})
        reasoner = _reasoner(client, **_puct())
        checks = 0

        def within_deadline(_deadline):
            nonlocal checks
            checks += 1
            return checks == 1

        reasoner._within_deadline = within_deadline
        result = asyncio.run(reasoner.prove(SEED, deadline=0.0))
        graph = result.graph
        self.assertEqual(graph.end_reason, SearchEndReason.DEADLINE)
        self.assertFalse(graph.is_solved())
        # The completed root expansion is kept as experience.
        self.assertTrue(graph.nodes[0].outgoing_edge_ids)
        self.assertTrue(result.edge_actions or result.failure_actions)
        for edge in graph.edges.values():
            self.assertEqual(edge.visit_stats.virtual_loss, 0)
        _assert_scoped_cleanly(self, client)


@pytest.mark.pantograph_live
def test_actual_puct_and_best_first_search_over_the_installed_client(live_env):
    async def run():
        client = await create_model_sexpr_client(live_env)
        async with client:
            baseline = (await client.stat()).n_goals
            checks = await verify_mcts_search(client, searches=20)
            assert (await client.stat()).n_goals == baseline
        assert client.phase.value == "CLOSED"
        return checks

    checks = asyncio.run(run())
    assert {
        "puct-multi-leaf-batch",
        "checked-rejection-of-alternatives",
        "puct-graph-equivalence",
        "best-first-graph-equivalence",
        "cooperative-deadline-scope-closure",
        "repeated-searches-20",
    } <= set(checks)


@pytest.mark.pantograph_live
def test_actual_pool_workers_share_one_model_device(live_env):
    assert asyncio.run(verify_shared_device_workers(live_env)) == [
        "two-workers-one-model-device"
    ]


@pytest.mark.pantograph_live
def test_actual_bounded_updates_with_htps_disabled_and_enabled(live_env):
    async def run():
        client = await create_model_sexpr_client(live_env)
        async with client:
            baseline = (await client.stat()).n_goals
            learning = await verify_learning_round(client)
            assert (await client.stat()).n_goals == baseline
        assert client.phase.value == "CLOSED"
        return learning

    learning = asyncio.run(run())
    disabled, enabled = learning["htps_disabled"], learning["htps_enabled"]
    assert disabled["onpolicy_optimizer_steps"] == 1
    assert disabled["htps_optimizer_steps"] == 0
    assert disabled["model_generation"] == 1
    assert enabled["onpolicy_optimizer_steps"] == 1
    assert enabled["htps_optimizer_steps"] == 1
    assert enabled["model_generation"] == 2
    assert enabled["imitation_samples"] or enabled["critic_samples"]


if __name__ == "__main__":
    unittest.main()
