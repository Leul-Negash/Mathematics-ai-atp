"""Bounded progressive widening in PUCT search (proposal-source lifecycle)."""

from __future__ import annotations

import asyncio
import math
import unittest

from maths_ai.data_models.proof_components import ProposedAction, TacticCandidate
from maths_ai.gnn_inference.tests.test_rl_reasoner import GOAL_EXPR, HYPS, SEED, _FakeServer
from maths_ai.gnn_inference.tests.model_helpers import structured_goal
from maths_ai.hybrid_reasoner.hypergraph import (
    NodeClosureReason,
    NodeStatus,
    SearchEndReason,
    TacticOutcome,
)
from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner
from maths_ai.hybrid_reasoner.selection_policy import WideningPolicy, resolve_widening


def proposal(name: str, probability: float = 0.25) -> ProposedAction:
    return ProposedAction(
        candidate=TacticCandidate(tactic_name=name, arguments=[], probability=probability),
        raw_action_id=("scripted", name),
        log_prior=math.log(probability),
        provider="scripted-test",
    )


class _ScriptedExecutor:
    """Lean stand-in: commands map to subgoal expressions, or are rejected."""

    def __init__(self, outcomes: dict[str, list[str] | None]):
        self.server = _FakeServer()
        # A widening round replays the root from its canonical goal.
        root = structured_goal(GOAL_EXPR, HYPS)
        self.server.goals_by_expression[root.replay_spec().expression] = root
        self.outcomes = outcomes
        self.calls: list[str] = []

    async def apply(self, scope, state, tactic):
        self.calls.append(tactic.tactic_name)
        outcome = self.outcomes.get(tactic.tactic_name)
        if outcome is None:
            return TacticOutcome(success=False, error="rejected")
        subgoals = [structured_goal(expression, HYPS) for expression in outcome]
        for subgoal in subgoals:
            self.server.goals_by_expression[subgoal.replay_spec().expression] = subgoal
        return TacticOutcome(success=True, subgoals=subgoals)


class _WideningReasoner(HybridReasoner):
    """A sampling-like provider: each round at a node takes the next scripted batch."""

    def __init__(self, executor, rounds, **search):
        self.rounds = rounds  # goal expression -> list of per-round proposal names
        self.served: dict[int, int] = {}
        self.round_draws: list[tuple[int, int]] = []
        self.statuses: list[str] = []
        super().__init__(None, None, None, executor=executor, use_pln=False, **search)

    def _build_gnn_engine(self, **_kwargs):
        return None

    def _proposal_source_is_finite(self) -> bool:
        return False

    def predict_next_tactics_for_nodes(self, nodes, draws=None):
        batches = []
        for node, count in zip(nodes, draws or [self.top_k_tactics] * len(nodes)):
            script = self.rounds.get(node.goal.expression, [])
            index = self.served.get(node.id, 0)
            self.served[node.id] = index + 1
            self.round_draws.append((node.id, count))
            names = script[index] if index < len(script) else script[-1] if script else []
            batches.append([proposal(name) for name in names[:count]])
        return batches

    def _backup_simulation(self, graph, simulation):
        super()._backup_simulation(graph, simulation)
        self.statuses.append(graph.root.status)


def search(rounds, outcomes, **settings):
    executor = _ScriptedExecutor(outcomes)
    params = dict(
        selection_policy="puct",
        num_simulations=12,
        sim_batch_size=1,
        top_k_tactics=2,
        max_depth=4,
        max_nodes=50,
    )
    params.update(settings)
    reasoner = _WideningReasoner(executor, rounds, **params)
    graph = asyncio.run(reasoner.prove(SEED))
    return graph, reasoner, executor


class WideningPolicyTests(unittest.TestCase):
    def test_capacity_grows_sublinearly_within_its_bounds(self):
        policy = WideningPolicy(initial_actions=4, coefficient=2.0, exponent=0.5, max_actions=8)
        capacities = [policy.capacity(v) for v in range(40)]
        self.assertEqual(capacities[:4], [4, 4, 4, 4])
        self.assertEqual(capacities[4], 5)  # ceil(2 * sqrt(5))
        self.assertEqual(max(capacities), 8)
        self.assertEqual(capacities, sorted(capacities))

    def test_draws_respect_the_round_capacity_and_draw_budgets(self):
        policy = WideningPolicy(initial_actions=4, max_actions=8, max_draws=10)
        self.assertEqual(policy.draws_for(visits=0, attempted=0, draws=0, per_round=3), 3)
        self.assertEqual(policy.draws_for(visits=0, attempted=3, draws=3, per_round=3), 1)
        self.assertEqual(policy.draws_for(visits=0, attempted=4, draws=4, per_round=3), 0)
        self.assertEqual(policy.draws_for(visits=50, attempted=4, draws=9, per_round=3), 1)
        self.assertTrue(policy.exhausted(attempted=8, draws=0))
        self.assertTrue(policy.exhausted(attempted=1, draws=10))

    def test_settings_are_validated_and_best_first_rejects_them(self):
        for bad in (
            dict(initial_actions=0),
            dict(initial_actions=5, max_actions=4),
            dict(coefficient=0.0),
            dict(coefficient=float("inf")),
            dict(exponent=1.0),
            dict(exponent=0.0),
            dict(max_draws=0),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                WideningPolicy(**bad)
        self.assertIsNone(resolve_widening("legacy"))
        with self.assertRaisesRegex(ValueError, "widening_max_draws"):
            resolve_widening("legacy", max_draws=10)
        self.assertEqual(resolve_widening("puct"), WideningPolicy())
        self.assertEqual(resolve_widening("puct", max_actions=9).max_actions, 9)


class WideningSearchTests(unittest.TestCase):
    def test_a_later_round_adds_the_solving_action_to_a_rejected_node(self):
        graph, reasoner, executor = search(
            {GOAL_EXPR: [["bad"], ["bad", "good"]]},
            {"good": []},
            widening_initial_actions=1,
            top_k_tactics=2,
        )
        self.assertTrue(graph.is_solved())
        self.assertEqual(graph.end_reason, SearchEndReason.ROOT_SOLVED)
        root = graph.root
        self.assertGreaterEqual(root.proposal_rounds, 2)
        # The rejected command ran in Lean once; its later draw reused the outcome.
        self.assertEqual(executor.calls, ["bad", "good"])
        self.assertIsNone(root.command_outcomes[("bad", ())])
        # The root was never closed or dead before it was solved.
        self.assertNotIn(NodeStatus.DEAD, reasoner.statuses)

    def test_widening_keeps_existing_statistics_and_priors(self):
        snapshots = []

        class Recording(_WideningReasoner):
            def predict_next_tactics_for_nodes(self, nodes, draws=None):
                for node in nodes:
                    for eid in node.outgoing_edge_ids:
                        edge = self._graph.edges[eid]
                        snapshots.append((eid, edge.visit_stats.N, edge.visit_stats.W, dict(edge.raw_log_priors)))
                return super().predict_next_tactics_for_nodes(nodes, draws)

            async def _prove_mcts(self, graph, deadline):
                self._graph = graph
                return await super()._prove_mcts(graph, deadline)

        executor = _ScriptedExecutor({"a": ["A"], "b": ["B"]})
        reasoner = Recording(
            executor,
            {GOAL_EXPR: [["a"], ["b"], ["c"]]},
            selection_policy="puct",
            num_simulations=10,
            sim_batch_size=1,
            top_k_tactics=1,
            max_depth=4,
            max_nodes=50,
            widening_initial_actions=1,
            widening_coefficient=1.0,
        )
        graph = asyncio.run(reasoner.prove(SEED))
        a_edge = next(e for e in graph.edges.values() if e.tactic.tactic_name == "a")
        before = [s for s in snapshots if s[0] == a_edge.id]
        self.assertTrue(before)
        # Visits only grow and the raw prior never changes as siblings arrive.
        counts = [n for _eid, n, _w, _priors in before]
        self.assertEqual(counts, sorted(counts))
        for _eid, _n, _w, priors in before:
            self.assertEqual(priors, {("scripted", "a"): math.log(0.25)})
        self.assertGreaterEqual(len(graph.root.outgoing_edge_ids), 2)
        priors = graph.puct_priors(graph.root_id)
        self.assertAlmostEqual(sum(priors.values()), 1.0)

    def test_a_rejection_only_source_is_bounded_and_ends_with_its_reason(self):
        names = [f"t{i}" for i in range(100)]
        rounds = {GOAL_EXPR: [names[i : i + 2] for i in range(0, 100, 2)]}
        graph, reasoner, executor = search(
            rounds, {}, num_simulations=200, widening_initial_actions=2, widening_max_actions=6
        )
        root = graph.root
        self.assertEqual(root.closure_reason, NodeClosureReason.WIDENING_BOUND)
        self.assertEqual(root.status, NodeStatus.DEAD)
        self.assertEqual(graph.end_reason, SearchEndReason.ROOT_DEAD)
        self.assertEqual(len(executor.calls), 6)
        self.assertEqual(root.attempted_commands, 6)
        # Capacity grew only with completed visits: some simulations had to wait.
        self.assertGreater(graph.deferred_selections, 0)

    def test_a_duplicate_only_source_is_bounded_by_its_draws(self):
        graph, reasoner, executor = search(
            {GOAL_EXPR: [["same", "same"]]},
            {},
            num_simulations=500,
            widening_initial_actions=4,
            widening_max_draws=9,
        )
        root = graph.root
        self.assertEqual(executor.calls, ["same"])
        self.assertEqual(root.closure_reason, NodeClosureReason.WIDENING_BOUND)
        self.assertGreaterEqual(root.proposal_draws, 9)
        self.assertLessEqual(root.proposal_draws, 10)
        self.assertEqual(graph.end_reason, SearchEndReason.ROOT_DEAD)

    def test_an_open_source_waits_without_ending_the_search_early(self):
        graph, _reasoner, executor = search(
            {GOAL_EXPR: [["bad"]] + [["bad"]] * 50},
            {},
            num_simulations=5,
            widening_initial_actions=1,
            widening_coefficient=0.5,
        )
        # Capacity stays 1 for these visits: one Lean call, the rest deferred,
        # and the simulation budget, not an empty frontier, ends the search.
        self.assertEqual(executor.calls, ["bad"])
        self.assertEqual(graph.root.status, NodeStatus.OPEN)
        self.assertEqual(graph.end_reason, SearchEndReason.NUM_SIMULATIONS)
        self.assertEqual(graph.root.visits, 5)

    def test_round_draws_are_fixed_from_completed_visits(self):
        _graph, reasoner, _executor = search(
            {GOAL_EXPR: [["a", "b", "c"]] * 20},
            {},
            num_simulations=8,
            top_k_tactics=3,
            widening_initial_actions=2,
        )
        first = reasoner.round_draws[0]
        self.assertEqual(first, (0, 2))  # capacity 2 at zero visits, not top_k 3

    def test_counters_report_the_widening_work(self):
        graph, _reasoner, executor = search(
            {GOAL_EXPR: [["bad"], ["bad", "good"]]}, {"good": []}, widening_initial_actions=1
        )
        counters = graph.search_counters()
        self.assertEqual(counters["distinct_commands"], 2.0)
        self.assertEqual(counters["accepted_edges"], 1.0)
        self.assertGreaterEqual(counters["widening_rounds"], 1.0)
        self.assertGreaterEqual(counters["raw_draws"], counters["distinct_commands"])
        self.assertGreaterEqual(counters["distinct_commands"], counters["accepted_edges"])


if __name__ == "__main__":
    unittest.main()
