"""Frozen dataset rows through the loader, the replay adapters and real Lean."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import _dataset_row_from_sample
from maths_ai.lean_replay import (
    ReplayFailure,
    ReplayInputError,
    render_replay,
    seed_from_text,
)

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/dataset_replay/rows.json").read_text(encoding="utf-8")
)
ROWS = {row["theorem"] if row["origin"] == "synthetic" else row["row_index"]: row for row in FIXTURE["rows"]}
PI = "synthetic.printed_pi"


def dataset_row(key):
    row = ROWS[key]
    return _dataset_row_from_sample(
        row,
        split=row["split"],
        row_index=row["row_index"],
        dataset_name=FIXTURE["dataset"],
        dataset_revision=FIXTURE["revision"],
    )


def failure(call) -> ReplayFailure | None:
    try:
        call()
    except ReplayInputError as exc:
        return exc.reason
    return None


def test_fixture_rows_are_marked_original_or_synthetic():
    assert {row["origin"] for row in FIXTURE["rows"]} == {"original", "synthetic"}
    assert FIXTURE["revision"] == "f44841f98193c09b20c6092198a22e98ff2cd7a8"


@pytest.mark.parametrize(
    "key,structured,text",
    [
        (0, ReplayFailure.UNRESOLVED_UNIVERSE, ReplayFailure.UNRESOLVED_UNIVERSE),
        (5, None, None),
        (7, None, None),
        (27, ReplayFailure.MISSING_CONTEXT, None),
        (41, None, None),
        (PI, None, None),
    ],
)
def test_offline_classification(key, structured, text):
    row = dataset_row(key)
    assert failure(lambda: render_replay(row.structured_seed())) is structured
    assert failure(lambda: render_replay(seed_from_text(row.state))) is text


def test_structured_and_text_contexts_agree_once_multiline_types_parse():
    row = dataset_row(7)
    structured = row.structured_seed()
    text = seed_from_text(row.state)
    assert [local.name for local in text.locals] == [local.name for local in structured.locals]


def test_raw_constants_replace_environment_dependent_printed_names():
    assert "@Real.pi" in render_replay(dataset_row(PI).structured_seed()).expression
    assert "π" in render_replay(seed_from_text(dataset_row(PI).state)).expression
    assert "@Submodule.span" in render_replay(dataset_row(41).structured_seed()).expression


@pytest.fixture
def mathlib_env(request):
    from pantograph_async import PantographEnv

    source = os.environ.get("PANTOGRAPH_TEST_SOURCE_ROOT")
    repl = os.environ.get("PANTOGRAPH_TEST_REPL")
    if not source or not repl:
        if request.config.getoption("--run-pantograph-live"):
            pytest.fail(
                "Required live run needs PANTOGRAPH_TEST_SOURCE_ROOT and PANTOGRAPH_TEST_REPL."
            )
        pytest.skip("Explicit live paths not configured.")
    return PantographEnv(
        source_root=Path(source), pantograph_repl=Path(repl), imports=("Mathlib",)
    )


@pytest.mark.pantograph_live
def test_live_replay_repairs_supported_rows_and_rejects_the_rest(mathlib_env):
    from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
        create_model_sexpr_client,
        materialize_seed,
    )

    async def outcome(scope, seed):
        try:
            _, goal, validation = await materialize_seed(scope, seed)
        except ReplayInputError as exc:
            return exc.reason, None, None
        return None, goal, validation

    async def run():
        client = await create_model_sexpr_client(mathlib_env)
        async with client:
            baseline = (await client.stat()).n_goals
            async with client.state_scope() as scope:
                # Rows whose raw columns are complete replay exactly, names included.
                for key in (5, 7, 41, PI):
                    seed = dataset_row(key).structured_seed()
                    reason, goal, validation = await outcome(scope, seed)
                    assert reason is None, (key, reason)
                    # Lean prints exactly the names the replay predicted; any
                    # rename is listed in the replay's name map.
                    assert validation.name_drift == ()
                    replay = render_replay(seed)
                    assert [l.user_name for l in goal.locals] == list(replay.expected_names)
                    renamed = {source for source, _ in replay.name_map}
                    assert all(
                        l.user_name == s.name
                        for l, s in zip(goal.locals, seed.locals)
                        if s.name not in renamed
                    )
                    # The canonical goal replays through the same rules.
                    reason, again, _ = await outcome(scope, goal.replay_seed())
                    assert reason is None and again.state_fingerprint() == goal.state_fingerprint()

                # Printed names that need the source file's `open` fail in Lean.
                for key in (41, PI):
                    reason, _, _ = await outcome(scope, seed_from_text(dataset_row(key).state))
                    assert reason is ReplayFailure.LEAN_REJECTED, key

                # Named universes: shared and multiple levels, dependent binders.
                universes = seed_from_text(
                    "α : Type u_1\nβ : Type u_1\nγ : Sort (max u v)\nf : α → β\na : α\n⊢ f a = f a"
                )
                assert render_replay(universes).universe_levels == ("u_1", "u", "v")
                reason, goal, _ = await outcome(scope, universes)
                assert reason is None and [l.user_name for l in goal.locals][-1] == "a"

                # A canonical goal with a local definition and an instance.
                start = await scope.start_goal(
                    "∀ (α : Type) [inst : Inhabited α] (n : Nat), let x := n; x = x"
                )
                state = await scope.apply_tactic(start, "intro α inst n x")
                from maths_ai.hybrid_reasoner.pantograph_adapter import execution_goal_to_goal

                canonical = execution_goal_to_goal(state.goals[0])
                reason, replayed, _ = await outcome(scope, canonical.replay_seed())
                assert reason is None
                assert replayed.state_fingerprint() == canonical.state_fingerprint()
            assert (await client.stat()).n_goals == baseline
        assert client.phase.value == "CLOSED"

    asyncio.run(run())
