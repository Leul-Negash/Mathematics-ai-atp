"""The search over retained snapshots: admission once, replay, certification, retention."""

from __future__ import annotations

import asyncio
import unittest

import torch

from pantograph_async import CertificationResult

from maths_ai.data_models.proof_components import TacticCandidate
from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner
from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import (
    compute_backups,
    extract_actor_transitions,
)
from maths_ai.gnn_inference.tests.model_helpers import actor_critic
from maths_ai.gnn_inference.tests.snapshot_fake import FakeSnapshotScope, Spec, Step
from maths_ai.hybrid_reasoner import joint_inference
from maths_ai.hybrid_reasoner.execution_limits import ResidentMemoryMonitor, RetentionLimits
from maths_ai.hybrid_reasoner.hypergraph import NodeClosureReason, SearchEndReason
from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner, PantographExecutor
from maths_ai.hybrid_reasoner.pantograph_adapter import execution_goal_to_goal
from maths_ai.lean_replay import seed_from_text

SEED = seed_from_text(goal="root")


class _Client:
    """A client whose scopes are scripted snapshot sessions."""

    pid = None

    def __init__(self, rules, root: Spec):
        self.rules = rules
        self.root = root
        self.scopes = []

    def state_scope(self):
        scope = FakeSnapshotScope(self.rules, roots={"root": self.root})
        self.scopes.append(scope)
        return scope


async def _admit(scope, seed):
    start = await scope.start_goal("root")
    state = await scope.apply_tactic(start, "skip")
    await scope.release(start)
    return state, execution_goal_to_goal(state.goals[0]), None


def _commands(table):
    def rules(spec, command):
        return table.get((spec.target, command))

    return rules


class _Scripted(HybridReasoner):
    def __init__(self, proposals, **kwargs):
        self.proposals = proposals
        super().__init__(
            config_path=None, tactic_model_path=None, argument_model_path=None,
            use_pln=False, **kwargs,
        )

    def _build_gnn_engine(self, **_kwargs):
        return None

    def predict_next_tactic(self, goal):
        return [
            TacticCandidate(tactic_name=name, arguments=[], probability=p)
            for name, p in self.proposals.get(goal.expression, [])
        ]


class RetainedSearchTests(unittest.TestCase):
    def setUp(self):
        self._original = joint_inference.materialize_seed
        joint_inference.materialize_seed = _admit

    def tearDown(self):
        joint_inference.materialize_seed = self._original

    def search(self, rules, root, proposals, **kwargs):
        client = _Client(rules, root)
        reasoner = _Scripted(proposals, executor=PantographExecutor(client), **kwargs)
        graph = asyncio.run(reasoner.prove(SEED))
        return graph, client.scopes[0]

    def test_a_conjunction_is_certified_through_replay_without_reconstruction(self):
        rules = _commands({
            ("P ∧ Q", "constructor"): Step(children=(Spec("P"), Spec("Q"))),
            ("P", "trivial"): Step(),
            ("Q", "trivial"): Step(),
        })
        proposals = {"P ∧ Q": [("constructor", 0.9)], "P": [("trivial", 0.8)], "Q": [("trivial", 0.7)]}
        graph, scope = self.search(rules, Spec("P ∧ Q"), proposals)
        self.assertTrue(graph.is_solved())
        self.assertEqual(graph.search_end_reason, SearchEndReason.ROOT_CERTIFIED)
        self.assertEqual(scope.count("goal.start"), 1)
        self.assertEqual(graph.execution["intermediate_reconstructions"], 0)
        self.assertEqual(graph.execution["original_tactic_calls"], 3)
        self.assertEqual(graph.execution["replay_tactic_calls"], 3)
        self.assertEqual(
            graph.proof_trace(),
            {"goal": "P ∧ Q", "command": "constructor", "subgoals": [
                {"goal": "P", "command": "trivial", "subgoals": []},
                {"goal": "Q", "command": "trivial", "subgoals": []},
            ]},
        )
        self.assertEqual(graph.artifact.statement["expression"], "root")
        self.assertTrue(all(node.occurrence is None for node in graph.nodes.values()))
        self.assertEqual(scope.live, 0)

    def test_a_ready_root_is_not_a_proof_until_certified(self):
        rules = _commands({("P", "trivial"): Step()})
        client = _Client(rules, Spec("P"))
        reasoner = _Scripted({"P": [("trivial", 0.9)]}, executor=PantographExecutor(client))

        async def run():
            original = FakeSnapshotScope.certify

            async def refuse(self, state, *, max_heartbeats=None):
                return CertificationResult("rejected", "kernel rejected the proof", None)

            FakeSnapshotScope.certify = refuse
            try:
                return await reasoner.prove(SEED)
            finally:
                FakeSnapshotScope.certify = original

        graph = asyncio.run(run())
        self.assertTrue(graph.has_candidate())
        self.assertFalse(graph.is_solved())
        self.assertEqual(graph.execution["last_replay"], "conflict")
        self.assertEqual(extract_actor_transitions(graph), [])
        tables = compute_backups(graph)
        self.assertFalse(any(value.is_known for value in tables.edge_outcomes.values()))

    def test_retention_pressure_prunes_without_rebuilding_goals(self):
        def rules(spec, command):
            if command.startswith("split"):
                return Step(children=tuple(Spec(f"{spec.target}.{i}") for i in range(3)))
            return None

        proposals = {}
        for depth in range(4):
            for name in self._names(depth):
                proposals[name] = [(f"split{k}", 0.9 - 0.1 * k) for k in range(2)]
        bound = 5
        graph, scope = self.search(
            rules, Spec("r"), proposals,
            retention=RetentionLimits(max_retained_snapshots=bound), max_nodes=60, max_depth=6,
        )
        self.assertLessEqual(graph.execution["peak_snapshots"], bound)
        self.assertGreater(graph.execution["pruned_nodes"], 0)
        pruned = [n for n in graph.nodes.values() if n.closure_reason == NodeClosureReason.RESOURCE_PRUNED]
        self.assertTrue(pruned)
        self.assertEqual(scope.count("goal.start"), 1)
        tables = compute_backups(graph)
        for node in pruned:
            self.assertFalse(tables.node_targets[node.id].is_known)

    @staticmethod
    def _names(depth):
        names = ["r"]
        for _ in range(depth):
            names = [f"{name}.{i}" for name in names for i in range(3)]
        return names

    def test_pins_beyond_the_bound_end_with_a_resource_limit(self):
        rules = _commands({("P", "trivial"): Step()})
        graph, scope = self.search(
            rules, Spec("P"), {"P": [("trivial", 0.9)]},
            retention=RetentionLimits(max_retained_snapshots=1),
        )
        self.assertEqual(graph.search_end_reason, SearchEndReason.RESOURCE_LIMIT)
        self.assertEqual(scope.count("goal.tactic"), 1)  # admission only
        self.assertFalse(graph.is_solved())

    def test_memory_pressure_prunes_then_stops(self):
        def rules(spec, command):
            return Step(children=(Spec(spec.target + "'"), Spec(spec.target + '"')))

        monitor = ResidentMemoryMonitor(1.0)
        monitor.exceeded = lambda: True
        proposals = {}
        names = ["r"]
        for _ in range(5):
            for name in names:
                proposals[name] = [("split", 0.9)]
            names = [n + suffix for n in names for suffix in ("'", '"')]
        graph, _scope = self.search(
            rules, Spec("r"), proposals, collection_memory=monitor, max_nodes=50,
        )
        self.assertEqual(graph.search_end_reason, SearchEndReason.RESOURCE_LIMIT)

    def test_a_long_search_stays_within_its_bounds(self):
        def rules(spec, command):
            return Step(children=(Spec(spec.target + "+"),)) if command == "step" else None

        proposals = {}
        name = "r"
        for _ in range(300):
            proposals[name] = [("step", 0.9), ("other", 0.5)]
            name += "+"
        # The anchor, the expanded node and one result: a bound of 4 leaves room
        # for the second proposal, so nothing on the only branch is pruned.
        bound = 4
        graph, scope = self.search(
            rules, Spec("r"), proposals,
            retention=RetentionLimits(max_retained_snapshots=bound), max_nodes=250, max_depth=400,
        )
        self.assertEqual(graph.search_end_reason, SearchEndReason.MAX_NODES)
        self.assertLessEqual(graph.execution["peak_snapshots"], bound)
        self.assertEqual(scope.live, 0)
        # One adoption per accepted attempt plus the admitted root; rejected
        # commands allocate nothing.
        self.assertEqual(graph.execution["snapshots_adopted"], len(graph.edges) + 1)


class ReplayLearningTests(unittest.TestCase):
    def setUp(self):
        self._original = joint_inference.materialize_seed
        joint_inference.materialize_seed = _admit

    def tearDown(self):
        joint_inference.materialize_seed = self._original

    def test_replay_adds_no_policy_samples(self):
        from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
        from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab

        tactics = {"trivial": 0, "exact": 1}
        rules = lambda spec, command: Step() if command == "trivial" else None
        client = _Client(rules, Spec("P", (("h", "P"),)))

        async def root_goal():
            scope = FakeSnapshotScope(rules, roots={"root": client.root})
            _state, goal, _ = await _admit(scope, SEED)
            return goal

        torch.manual_seed(0)
        vocab = build_vocab([model_goal_to_dag(asyncio.run(root_goal()))])
        reasoner = RLHybridReasoner(
            actor_critic(len(vocab), len(tactics)),
            vocab,
            tactics,
            executor=PantographExecutor(client),
            top_k_tactics=4,
            use_pln=False,
        )
        result = asyncio.run(reasoner.prove(SEED))
        draws = sum(a.multiplicity for a in result.edge_actions.values())
        draws += sum(f.action.multiplicity for f in result.failure_actions)
        self.assertEqual(draws, 4)
        if result.certified:
            self.assertGreaterEqual(result.graph.execution["replay_tactic_calls"], 1)
            witnessed = result.graph.witnessed_edges()
            self.assertTrue(witnessed <= set(result.edge_actions))


if __name__ == "__main__":
    unittest.main()


class HoleObservationTests(unittest.TestCase):
    """Goals with unassigned metavariables now reach the model; this is their gate."""

    def goal(self):
        from maths_ai.data_models.proof_components import Goal

        return Goal(
            expression="?w = 0",
            goal_model_sexp="(:app (:c Eq) (:arg :implicit-type 0 (:c Nat)) "
            "(:arg :explicit 1 (:metavar)) (:arg :explicit 2 (:lit 0)))",
            goal_sexp="((:c Eq) (:c Nat) (:mv _uniq.100) (:lit 0))",
            locals=[],
            model_sexp_version=1,
        )

    def test_a_metavariable_featurizes_as_its_structural_label(self):
        from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
        from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import make_dag_featurizer
        from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab

        goal = self.goal()
        dag = model_goal_to_dag(goal)
        self.assertIn(":metavar", [node.label for node in dag.nodes])
        vocab = build_vocab([dag])
        first = make_dag_featurizer(vocab)(goal)[1]
        second = make_dag_featurizer(vocab)(goal)[1]
        self.assertEqual(first.graph_fingerprint, second.graph_fingerprint)
        self.assertTrue(torch.equal(first.x, second.x))
        self.assertEqual(int(first.unknown_label_count), 0)

    def test_a_vocabulary_without_it_closes_the_node_as_unsupported(self):
        from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
        from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import make_dag_featurizer
        from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
        from maths_ai.hybrid_reasoner.hypergraph import ProofHypergraph, UnsupportedObservation

        goal = self.goal()
        vocab = {k: v for k, v in build_vocab([model_goal_to_dag(goal)]).items() if k != ":metavar"}
        with self.assertRaises(UnsupportedObservation):
            make_dag_featurizer(vocab)(goal)
        graph = ProofHypergraph(goal)
        graph.mark_node_unsupported(graph.root_id, note="vocabulary")
        self.assertEqual(graph.root.closure_reason, NodeClosureReason.UNSUPPORTED_OBSERVATION)
        self.assertFalse(compute_backups(graph).node_targets[graph.root_id].is_known)

    def test_search_continues_past_an_unsupported_observation(self):
        from maths_ai.hybrid_reasoner.hypergraph import UnsupportedObservation

        class Partial(_Scripted):
            def predict_next_tactic(self, goal):
                if goal.expression == "Q":
                    raise UnsupportedObservation("missing :metavar")
                return super().predict_next_tactic(goal)

        original = joint_inference.materialize_seed
        joint_inference.materialize_seed = _admit
        try:
            rules = _commands({
                ("P ∧ Q", "constructor"): Step(children=(Spec("P"), Spec("Q"))),
                ("P", "trivial"): Step(),
            })
            client = _Client(rules, Spec("P ∧ Q"))
            reasoner = Partial(
                {"P ∧ Q": [("constructor", 0.9)], "P": [("trivial", 0.8)]},
                executor=PantographExecutor(client),
            )
            graph = asyncio.run(reasoner.prove(SEED))
        finally:
            joint_inference.materialize_seed = original
        reasons = {node.goal.expression: node.closure_reason for node in graph.nodes.values()}
        self.assertEqual(reasons["Q"], NodeClosureReason.UNSUPPORTED_OBSERVATION)
        self.assertEqual(graph.execution["unsupported_observations"], 1)
        self.assertFalse(graph.is_solved())
