from __future__ import annotations

import asyncio
import unittest

from pantograph_async import (
    PantographGoalRejected,
    PantographTacticRejected,
    PantographTransportError,
    PantographProtocolError,
    PantographStateError,
)
from maths_ai.gnn_inference.tests.model_helpers import FakeStateScope, ScopedFakeClient

from maths_ai.data_models.proof_components import (
    Goal,
    GoalLocal,
    LeanGoalSeed,
    TacticCandidate,
)
from maths_ai.hybrid_reasoner.joint_inference import (
    HybridReasoner,
    PantographExecutor,
    render_tactic_command,
)


def _tactic(name: str, *arguments: str) -> TacticCandidate:
    return TacticCandidate(tactic_name=name, arguments=list(arguments), probability=0.5)


class RenderTacticCommandTests(unittest.TestCase):
    def test_bracket_required_tactic_gets_brackets(self):
        self.assertEqual(render_tactic_command(_tactic("rw", "h")), "rw [h]")

    def test_multiple_rules_are_comma_separated(self):
        self.assertEqual(
            render_tactic_command(_tactic("rw", "h", "Nat.add_comm")),
            "rw [h, Nat.add_comm]",
        )

    def test_bracket_required_tactic_without_usable_names_is_unplayable(self):
        """`↑13` and `?m.2235` are expressions, not rules Lean can look up."""
        self.assertIsNone(render_tactic_command(_tactic("rw", "↑13")))
        self.assertIsNone(render_tactic_command(_tactic("rw", "?m.2235")))
        self.assertIsNone(render_tactic_command(_tactic("rw")))

    def test_unusable_arguments_are_filtered_from_the_rule_list(self):
        self.assertEqual(render_tactic_command(_tactic("rw", "↑13", "h")), "rw [h]")

    def test_bracket_optional_tactic_runs_bare_without_rules(self):
        self.assertEqual(render_tactic_command(_tactic("simp")), "simp")
        self.assertEqual(render_tactic_command(_tactic("simp", "↑13")), "simp")

    def test_bracket_optional_tactic_takes_brackets_with_rules(self):
        self.assertEqual(render_tactic_command(_tactic("simp", "h")), "simp [h]")
        self.assertEqual(
            render_tactic_command(_tactic("linarith", "h1", "h2")), "linarith [h1, h2]"
        )

    def test_ordinary_tactic_keeps_space_separated_arguments(self):
        self.assertEqual(render_tactic_command(_tactic("apply", "h")), "apply h")
        self.assertEqual(render_tactic_command(_tactic("intro", "p", "q")), "intro p q")
        self.assertEqual(
            render_tactic_command(_tactic("exact", "Nat.le_refl")), "exact Nat.le_refl"
        )

    def test_bare_tactic_renders_as_its_name(self):
        self.assertEqual(render_tactic_command(_tactic("rfl")), "rfl")

    def test_trailing_colons_and_blanks_are_stripped(self):
        """Arguments arrive rendered from DAG nodes, which can carry a `name :` form."""
        self.assertEqual(
            render_tactic_command(_tactic("apply", "h :", "  ")), "apply h"
        )

    def test_names_with_lean_punctuation_are_usable_rules(self):
        self.assertEqual(render_tactic_command(_tactic("rw", "h'")), "rw [h']")
        self.assertEqual(render_tactic_command(_tactic("rw", "α")), "rw [α]")
        self.assertEqual(
            render_tactic_command(_tactic("rw", "Nat.factorial_succ")),
            "rw [Nat.factorial_succ]",
        )


class _RecordingServer:
    """Captures the command string the executor sends to Lean."""

    def __init__(self):
        self.commands: list[str] = []

    async def apply_tactic(self, state, tactic):
        self.commands.append(tactic)
        raise PantographTacticRejected(
            "tactic failed"
        )  # the return path is not what these tests check


class ExecutorRenderingTests(unittest.TestCase):
    def test_executor_sends_bracketed_form(self):
        server = _RecordingServer()
        executor = PantographExecutor(server)
        outcome = asyncio.run(executor.apply(server, object(), _tactic("rw", "h")))
        self.assertEqual(server.commands, ["rw [h]"])
        self.assertFalse(outcome.success)

    def test_executor_drops_unplayable_tactic_without_calling_lean(self):
        server = _RecordingServer()
        executor = PantographExecutor(server)
        outcome = asyncio.run(executor.apply(server, object(), _tactic("rw", "↑13")))
        self.assertEqual(server.commands, [])  # never reached Lean
        self.assertFalse(outcome.success)
        self.assertIn("bracketed rule list", outcome.error)


class _HealthyClient(ScopedFakeClient):
    def __init__(self):
        self.started = []

    async def _start_goal(self, expression):
        self.started.append(expression)
        return "goal-state"

    async def _apply_tactic(self, state, tactic):
        return f"{state}+{tactic}"


def _reasoner(client):
    reasoner = object.__new__(HybridReasoner)
    reasoner.server = client
    reasoner._scope = FakeStateScope(client)
    return reasoner


class ScopedReplayTests(unittest.TestCase):
    def test_typed_failures_propagate_without_restart_or_retry(self):
        for error in (
            PantographGoalRejected("bad goal"),
            PantographTransportError("closed pipe"),
            PantographProtocolError("bad contract"),
            PantographStateError("stale state"),
        ):
            client = _HealthyClient()
            calls = []

            async def reject(expression):
                calls.append(expression)
                raise error

            client._start_goal = reject
            with self.assertRaises(type(error)):
                asyncio.run(
                    _reasoner(client)._start_state(LeanGoalSeed(expression="p → p"))
                )
            self.assertEqual(len(calls), 1)

    def test_hypotheses_are_quantified_then_introduced(self):
        healthy = _HealthyClient()
        state = asyncio.run(
            _reasoner(healthy)._start_state(
                LeanGoalSeed(expression="q ∨ p", hypotheses=["p : Prop", "h : p ∨ q"])
            )
        )
        self.assertEqual(healthy.started, ["∀ (p : Prop), ∀ (h : p ∨ q), q ∨ p"])
        self.assertEqual(state, "goal-state+intro p h")

    def test_structured_replay_preserves_binder_roles_and_local_let(self):
        healthy = _HealthyClient()
        reasoner = _reasoner(healthy)
        goal = Goal(
            expression="x = x",
            goal_model_sexp="(:app (:c Eq) (:fv FV2) (:fv FV2))",
            locals=[
                GoalLocal(
                    user_name="α",
                    context_index=0,
                    binder_role=":implicit",
                    type_pp="Type",
                    type_model_sexp="(:sort 1)",
                ),
                GoalLocal(
                    user_name="inst",
                    context_index=1,
                    binder_role=":instImplicit",
                    is_instance=True,
                    type_pp="Inhabited α",
                    type_model_sexp="(:app (:c Inhabited) (:fv FV0))",
                ),
                GoalLocal(
                    user_name="x",
                    context_index=2,
                    binder_role=":explicit",
                    is_let=True,
                    type_pp="α",
                    type_model_sexp="(:fv FV0)",
                    value_pp="default",
                    value_model_sexp="(:c default)",
                ),
            ],
            model_sexp_version=1,
        )

        state = asyncio.run(reasoner._start_state(goal))

        self.assertEqual(
            healthy.started,
            ["∀ {α : Type}, ∀ [inst : Inhabited α], let x : α := default; x = x"],
        )
        self.assertEqual(state, "goal-state+intro α inst x")
