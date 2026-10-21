"""Typed state ownership against a real final-contract diagnostic subprocess.

The fake models independent snapshots, not Lean's ancestor deletion semantics.
Real ancestor/sibling lifetime verification remains the Stage 5 deployment gate.
"""

import asyncio
from dataclasses import replace

import pytest

from maths_ai.hybrid_reasoner.pantograph_client import PantographClient, SessionPhase
from maths_ai.hybrid_reasoner.pantograph_protocol import (
    PantographGoalRejected,
    PantographProtocolError,
    PantographRequestTimeout,
    PantographStateError,
    PantographTacticRejected,
    StartedGoal,
    TacticState,
)
from maths_ai.gnn_inference.tests.test_pantograph_client import (
    assert_reaped,
    records,
    run,
    setup,
    wait_event,
)


def requests(report, command):
    return [row for row in records(report) if row.get("command") == command]


def test_starts_materialization_branches_and_batched_exit(setup):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as client:
            baseline = (await client.stat()).n_goals
            async with client.state_scope() as scope:
                start = await scope.start_goal("True ∧ True")
                assert isinstance(start, StartedGoal)
                assert not hasattr(start, "goals")
                split = await scope.apply_tactic(start, "split")
                assert isinstance(split, TacticState) and len(split.goals) == 2
                candidate = await scope.apply_tactic(split, "trivial", goal_index=1)
                assert candidate.goals == ()
                await scope.release(candidate)
                alternative = await scope.apply_tactic(split, "skip", goal_index=0)
                await scope.release(start)
                assert (await client.stat()).n_goals == baseline + 2
                assert alternative.goals[0].goal.goal_model_sexp == "(:fv FV0)"
            assert (await client.stat()).n_goals == baseline
            assert not client._scopes and not client._allocated_ids
            deletes = requests(report, "goal.delete")
            assert len(deletes[-1]["args"]["stateIds"]) == 2
            assert requests(report, "goal.tactic")[1]["args"]["goalId"] == 1

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize("index", [-1, 1, True, 0.0, "0", None])
def test_invalid_initial_index_fails_before_io(setup, index):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as client:
            async with client.state_scope() as scope:
                start = await scope.start_goal("True")
                with pytest.raises(PantographStateError):
                    await scope.apply_tactic(start, "skip", goal_index=index)
                assert not requests(report, "goal.tactic")
                assert client.is_usable

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize("tactic", ["trivial", "conv"])
def test_empty_or_conversion_materialized_state_is_not_selectable(setup, tactic):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as client:
            async with client.state_scope() as scope:
                start = await scope.start_goal("True")
                state = await scope.apply_tactic(start, tactic)
                with pytest.raises(PantographStateError):
                    await scope.apply_tactic(state, "skip")
                assert len(requests(report, "goal.tactic")) == 1

    run(body)
    assert_reaped(report)


def test_release_is_idempotent_and_forged_state_is_rejected(setup):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as client:
            async with client.state_scope() as scope:
                start = await scope.start_goal("True")
                for forged in (replace(start), replace(start, root="another-root")):
                    with pytest.raises(PantographStateError):
                        await scope.release(forged)
                await asyncio.gather(scope.release(start), scope.release(start))
                await scope.release(start)
                with pytest.raises(PantographStateError):
                    await scope.apply_tactic(start, "skip")
                assert len(requests(report, "goal.delete")) == 1
            assert len(requests(report, "goal.delete")) == 1

    run(body)
    assert_reaped(report)


def test_foreign_scope_and_session_fail_before_io(setup):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as first:
            async with await PantographClient.create(env) as second:
                async with first.state_scope() as owner, first.state_scope() as other:
                    async with second.state_scope() as foreign:
                        start = await owner.start_goal("True")
                        for scope in (other, foreign):
                            with pytest.raises(PantographStateError):
                                await scope.release(start)
                            with pytest.raises(PantographStateError):
                                await scope.apply_tactic(start, "skip")
                        assert not requests(report, "goal.delete")
                        assert not requests(report, "goal.tactic")

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize(
    "tactic", ["parse_failure", "tactic_failure", "logged_error_failure"]
)
def test_domain_failure_preserves_input_and_healthy_session(setup, tactic):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as client:
            async with client.state_scope() as scope:
                start = await scope.start_goal("True")
                with pytest.raises(PantographTacticRejected) as error:
                    await scope.apply_tactic(start, tactic)
                assert error.value.command == "goal.tactic"
                assert client.is_usable and (await client.stat()).n_goals == 1
                await scope.apply_tactic(start, "trivial")
            assert (await client.stat()).n_goals == 0

    run(body)
    assert_reaped(report)


def test_rejected_goal_has_no_allocation(setup):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as client:
            async with client.state_scope() as scope:
                with pytest.raises(PantographGoalRejected):
                    await scope.start_goal("reject")
                assert not scope._states and (await client.stat()).n_goals == 0

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize("tactic", ["sorry", "unsafe"])
def test_proof_rejection_deletes_result_before_raising(setup, tactic):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as client:
            async with client.state_scope() as scope:
                start = await scope.start_goal("True")
                with pytest.raises(PantographTacticRejected):
                    await scope.apply_tactic(start, tactic)
                assert client.is_usable and (await client.stat()).n_goals == 1
                assert requests(report, "goal.delete")[-1]["args"]["stateIds"] == [1]
                await scope.apply_tactic(start, "trivial")
            assert (await client.stat()).n_goals == 0

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize("tactic", ["missing_validation", "validation_error"])
def test_invalid_evidence_retires_process_and_invalidates_states(setup, tactic):
    env, report = setup

    async def body():
        client = await PantographClient.create(env)
        async with client.state_scope() as scope:
            start = await scope.start_goal("True")
            with pytest.raises(PantographProtocolError):
                await scope.apply_tactic(start, tactic)
            assert client.phase is SessionPhase.CLOSED
            assert not scope._live_ids and not client._allocated_ids
        assert not requests(report, "goal.delete")
        await client.aclose()

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize("at_exit", [False, True])
def test_deletion_failure_is_infrastructure_not_domain_success(
    setup, monkeypatch, at_exit
):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "delete_error")

    async def body():
        client = await PantographClient.create(env)
        with pytest.raises(PantographProtocolError):
            async with client.state_scope() as scope:
                start = await scope.start_goal("True")
                if not at_exit:
                    await scope.apply_tactic(start, "sorry")
        assert client.phase is SessionPhase.CLOSED
        assert not client._scopes and not client._allocated_ids

    run(body)
    assert_reaped(report)


def test_exception_exit_batches_cleanup(setup):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as client:
            with pytest.raises(ValueError, match="search failed"):
                async with client.state_scope() as scope:
                    await scope.start_goal("True")
                    await scope.start_goal("True")
                    raise ValueError("search failed")
            assert (await client.stat()).n_goals == 0
            assert requests(report, "goal.delete")[-1]["args"]["stateIds"] == [0, 1]

    run(body)
    assert_reaped(report)


def test_scope_exit_joins_cleanup_despite_repeated_cancellation(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "delete_delay")

    async def body():
        async with await PantographClient.create(env) as client:
            entered = asyncio.Event()

            async def search():
                async with client.state_scope() as scope:
                    await scope.start_goal("True")
                    entered.set()
                    await asyncio.Event().wait()

            task = asyncio.create_task(search())
            await entered.wait()
            task.cancel("original cancellation")
            await wait_event(report, "request", command="goal.delete")
            task.cancel("second cancellation")
            with pytest.raises(asyncio.CancelledError) as error:
                await task
            assert error.value.args == ("original cancellation",)
            assert (await client.stat()).n_goals == 0 and client.is_usable
            assert not client._scopes

    run(body)
    assert_reaped(report)


def test_cancel_during_explicit_deletion_retires_session(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "delete_delay")

    async def body():
        client = await PantographClient.create(env)
        async with client.state_scope() as scope:
            start = await scope.start_goal("True")
            task = asyncio.create_task(scope.release(start))
            await wait_event(report, "request", command="goal.delete")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert client.phase is SessionPhase.CLOSED
        assert not client._scopes

    run(body)
    assert_reaped(report)


def test_release_validation_occurs_after_lock_acquisition(setup):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as client:
            async with client.state_scope() as scope:
                start = await scope.start_goal("True")
                await client._lock.acquire()
                release = asyncio.create_task(scope.release(start))
                tactic = asyncio.create_task(scope.apply_tactic(start, "skip"))
                await asyncio.sleep(0)
                client._lock.release()
                await release
                with pytest.raises(PantographStateError):
                    await tactic
                assert not requests(report, "goal.tactic")

    run(body)
    assert_reaped(report)


def test_closing_scope_registers_inflight_result_before_batching(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "tactic_delay")

    async def body():
        async with await PantographClient.create(env) as client:
            async with client.state_scope() as scope:
                start = await scope.start_goal("True")
                tactic = asyncio.create_task(scope.apply_tactic(start, "skip"))
                await wait_event(report, "request", command="goal.tactic")
                await scope.aclose()
                await tactic
                assert (await client.stat()).n_goals == 0
                assert requests(report, "goal.delete")[-1]["args"]["stateIds"] == [0, 1]
                with pytest.raises(PantographStateError):
                    await scope.start_goal("True")

    run(body)
    assert_reaped(report)


def test_replacement_reusing_zero_cannot_accept_old_handle(setup):
    env, report = setup

    async def body():
        first = await PantographClient.create(env)
        async with first.state_scope() as old_scope:
            old = await old_scope.start_goal("True")
            await first.aclose()
            async with await PantographClient.create(env) as replacement:
                async with replacement.state_scope() as scope:
                    new = await scope.start_goal("True")
                    assert old.handle.state_id == new.handle.state_id == 0
                    assert old.handle.session_id != new.handle.session_id
                    with pytest.raises(PantographStateError):
                        await scope.apply_tactic(old, "skip")
                    await scope.apply_tactic(new, "trivial")
        assert len(requests(report, "goal.delete")) == 1

    run(body)
    assert_reaped(report)


def test_scope_not_entered_reentry_and_closed_operations(setup):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as client:
            scope = client.state_scope()
            with pytest.raises(PantographStateError):
                await scope.start_goal("True")
            async with scope:
                with pytest.raises(PantographStateError):
                    await scope.__aenter__()
                start = await scope.start_goal("True")
            await scope.aclose()
            for call in (
                scope.release(start),
                scope.apply_tactic(start, "skip"),
                scope.__aenter__(),
            ):
                with pytest.raises(PantographStateError):
                    await call
        with pytest.raises(PantographStateError):
            client.state_scope()

    run(body)
    assert_reaped(report)


def test_scope_exit_timeout_before_write_still_retires_owned_allocations(setup):
    env, report = setup

    async def body():
        client = await PantographClient.create(replace(env, timeout=0.08))
        scope = client.state_scope()
        await scope.__aenter__()
        await scope.start_goal("True")
        await client._lock.acquire()
        try:
            with pytest.raises(PantographRequestTimeout):
                await scope.aclose()
        finally:
            client._lock.release()
        assert client.phase is SessionPhase.CLOSED
        assert not requests(report, "goal.delete")
        assert not client._scopes and not client._allocated_ids

    run(body)
    assert_reaped(report)


def test_cancel_rejection_cleanup_waiter_retires_instead_of_leaking(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "delete_delay")

    async def body():
        client = await PantographClient.create(env)
        async with client.state_scope() as scope:
            start = await scope.start_goal("True")
            task = asyncio.create_task(scope.apply_tactic(start, "sorry"))
            await wait_event(report, "request", command="goal.delete")
            assert len(scope._live_ids) == 2
            task.cancel("rejection cancelled")
            with pytest.raises(asyncio.CancelledError) as error:
                await task
            assert error.value.args == ("rejection cancelled",)
            assert client.phase is SessionPhase.CLOSED
        assert not client._scopes and not client._allocated_ids

    run(body)
    assert_reaped(report)


def test_original_search_cancellation_survives_cleanup_failure(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "delete_error")

    async def body():
        client = await PantographClient.create(env)
        entered = asyncio.Event()

        async def search():
            async with client.state_scope() as scope:
                await scope.start_goal("True")
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(search())
        await entered.wait()
        task.cancel("external abort")
        with pytest.raises(asyncio.CancelledError) as error:
            await task
        assert error.value.args == ("external abort",)
        assert any("deletion unavailable" in note for note in error.value.__notes__)
        assert client.phase is SessionPhase.CLOSED and not client._scopes

    run(body)
    assert_reaped(report)


def test_cancellation_waiting_for_tactic_lock_preserves_session(setup):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as client:
            async with client.state_scope() as scope:
                start = await scope.start_goal("True")
                await client._lock.acquire()
                task = asyncio.create_task(scope.apply_tactic(start, "skip"))
                await asyncio.sleep(0)
                task.cancel()
                try:
                    with pytest.raises(asyncio.CancelledError):
                        await task
                finally:
                    client._lock.release()
                assert client.is_usable and not requests(report, "goal.tactic")
                await scope.apply_tactic(start, "trivial")

    run(body)
    assert_reaped(report)


def test_scope_and_states_reject_other_event_loop_before_io(setup):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as client:
            async with client.state_scope() as scope:
                state = await scope.start_goal("True")

                async def foreign_calls():
                    for call in (
                        scope.release(state),
                        scope.apply_tactic(state, "skip"),
                        scope.aclose(),
                    ):
                        with pytest.raises(
                            PantographStateError, match="owning event loop"
                        ):
                            await call

                # A separate loop in another thread exercises public owner checks.
                await asyncio.to_thread(lambda: asyncio.run(foreign_calls()))
                assert not requests(report, "goal.tactic") and not requests(
                    report, "goal.delete"
                )

    run(body)
    assert_reaped(report)


def test_reused_released_id_does_not_revive_old_state(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "reuse_released_zero")

    async def body():
        async with await PantographClient.create(env) as client:
            async with client.state_scope() as scope:
                old = await scope.start_goal("True")
                await scope.release(old)
                new = await scope.start_goal("True")
                assert old.handle == new.handle
                await scope.release(old)
                assert (await client.stat()).n_goals == 1
                with pytest.raises(PantographStateError):
                    await scope.apply_tactic(old, "skip")
                await scope.apply_tactic(new, "trivial")
            assert (await client.stat()).n_goals == 0

    run(body)
    assert_reaped(report)


def test_bookkeeping_failure_after_owned_allocation_retires_session(setup, monkeypatch):
    env, report = setup

    async def body():
        client = await PantographClient.create(env)
        async with client.state_scope() as scope:
            register = scope._register

            def failing_register(response):
                register(response)
                raise RuntimeError("allocation bookkeeping failed")

            monkeypatch.setattr(scope, "_register", failing_register)
            with pytest.raises(PantographProtocolError, match="bookkeeping failed"):
                await scope.start_goal("True")
            assert client.phase is SessionPhase.CLOSED and not scope._live_ids
        assert not client._scopes and not requests(report, "goal.delete")

    run(body)
    assert_reaped(report)


def test_cancellation_after_registration_has_an_explicit_owner(setup, monkeypatch):
    env, report = setup

    async def body():
        async with await PantographClient.create(env) as client:
            captured = []
            with pytest.raises(asyncio.CancelledError, match="after allocation"):
                async with client.state_scope() as scope:
                    register = scope._register
                    task = asyncio.current_task()

                    def cancelling_register(response):
                        state = register(response)
                        captured.append((state.handle.state_id, set(scope._live_ids)))
                        client._loop.call_soon(task.cancel, "after allocation")
                        return state

                    monkeypatch.setattr(scope, "_register", cancelling_register)
                    await scope.start_goal("True")
                    await asyncio.sleep(0)
            assert captured == [(0, {0})]
            assert (await client.stat()).n_goals == 0 and not client._scopes

    run(body)
    assert_reaped(report)


def test_post_write_tactic_cancellation_clears_all_scope_ownership(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "tactic_delay")

    async def body():
        client = await PantographClient.create(env)
        async with client.state_scope() as scope:
            start = await scope.start_goal("True")
            task = asyncio.create_task(scope.apply_tactic(start, "skip"))
            await wait_event(report, "request", command="goal.tactic")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert client.phase is SessionPhase.CLOSED
            assert not scope._live_ids and not client._allocated_ids
        assert not client._scopes and not requests(report, "goal.delete")

    run(body)
    assert_reaped(report)
