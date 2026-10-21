"""Stage 2 process tests use real children, not mocked pipe responses."""

import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from maths_ai.hybrid_reasoner.pantograph_client import PantographClient, SessionPhase
from maths_ai.hybrid_reasoner.pantograph_env import PantographEnv
from maths_ai.hybrid_reasoner.pantograph_launch import build_launch_spec
from maths_ai.hybrid_reasoner.pantograph_protocol import (
    PantographConfigurationError,
    PantographGoalRejected,
    PantographProtocolError,
    PantographRequestTimeout,
    PantographStartupError,
    PantographStateError,
    PantographTransportError,
)


FAKE = Path(__file__).parent / "fixtures" / "pantograph" / "fake_repl.py"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "mathlib"
    project = tmp_path / "Pantograph"
    for root in (source, project):
        root.mkdir()
        (root / "lakefile.lean").write_text("import Lake\n")
        (root / "lean-toolchain").write_text("leanprover/lean4:v4.10.0-rc1\n")
    repl = project / ".lake" / "build" / "bin" / "repl"
    repl.parent.mkdir(parents=True)
    shutil.copyfile(FAKE, repl)
    repl.chmod(0o755)
    tools = tmp_path / "tools"
    tools.mkdir()
    lake = tools / "lake"
    shutil.copyfile(FAKE, lake)
    lake.chmod(0o755)
    report = tmp_path / "events.jsonl"
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PANTOGRAPH_FAKE_REPORT", str(report))
    monkeypatch.setenv(
        "PANTOGRAPH_FAKE_CONTRACT", str(FAKE.with_name("future_contract.json"))
    )
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "normal")
    monkeypatch.setenv("PANTOGRAPH_FAKE_RESOLVER", "normal")
    env = PantographEnv(
        source_root=source,
        pantograph_repl=repl,
        imports=("Init",),
        timeout=1,
        server_start_timeout_s=2,
        worker_shutdown_timeout_s=0.4,
    )
    return env, report


def records(report):
    return (
        [json.loads(line) for line in report.read_text().splitlines()]
        if report.exists()
        else []
    )


def assert_reaped(report):
    for pid in {entry["pid"] for entry in records(report)}:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


async def wait_event(report, kind, **fields):
    deadline = asyncio.get_running_loop().time() + 3
    while asyncio.get_running_loop().time() < deadline:
        if any(
            row["kind"] == kind
            and all(row.get(key) == value for key, value in fields.items())
            for row in records(report)
        ):
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"Child event missing: {kind} {fields}; {records(report)}")


def run(body):
    async def checked():
        errors = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))
        await body()
        await asyncio.sleep(0)
        assert not errors
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and task.get_name().startswith("pantograph-")
        ]

    asyncio.run(checked(), debug=True)


def test_imports_do_not_load_upstream_or_models():
    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pantograph', 'torch'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from maths_ai.hybrid_reasoner.pantograph_env import PantographEnv
from maths_ai.hybrid_reasoner.pantograph_client import PantographClient
assert not any(name.startswith(('pantograph', 'torch')) for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, timeout=10
    )


def test_normal_startup_launch_and_context_cleanup(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("LEAN_PATH", "parent-path-unchanged")

    async def body():
        client = await PantographClient.create(env)
        async with client:
            assert client.is_usable and client.phase is SessionPhase.READY
            assert (await client.stat()).n_goals == 0
            assert client.launch_spec.command[1:3] == ("-oL", str(env.pantograph_repl))
            assert client.effective_options.to_dict()["automaticMode"] is True
            assert os.environ["LEAN_PATH"] == "parent-path-unchanged"
        assert client.phase is SessionPhase.CLOSED
        assert not client.is_usable
        await client.aclose()
        with pytest.raises(PantographStateError):
            await client.stat()
        child = [
            entry
            for entry in records(report)
            if entry["kind"] == "spawn" and entry["executable"] == "repl"
        ][0]
        assert child["cwd"] == str(env.source_root)
        assert child["args"] == ["Init"]
        assert child["lean_path"] == "/fake/lean/path"

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize("scenario", ["split", "stderr_flood"])
def test_split_lines_and_stderr_flood_do_not_block(setup, monkeypatch, scenario):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", scenario)

    async def body():
        client = await PantographClient.create(replace(env, stderr_tail_limit=512))
        try:
            assert (await client.stat()).n_goals == 0
            if scenario == "stderr_flood":
                assert len(client.stderr_tail.encode()) <= 512
                assert client.stderr_tail.endswith("TAIL-MARKER")
        finally:
            await client.aclose()

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize(
    "scenario",
    [
        "bad_ready",
        "ready_eof",
        "bad_descriptor",
        "unextended",
        "options_error",
        "options_mismatch",
        "unsolicited",
    ],
)
def test_failed_startup_reaps_all_children(setup, monkeypatch, scenario):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", scenario)

    async def body():
        with pytest.raises(
            (PantographStartupError, PantographProtocolError, PantographTransportError)
        ):
            await PantographClient.create(env)

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize("scenario", ["ready_delay", "startup_total_delay"])
def test_startup_has_one_deadline_for_all_phases(setup, monkeypatch, scenario):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", scenario)

    async def body():
        start = asyncio.get_running_loop().time()
        with pytest.raises((PantographStartupError, PantographRequestTimeout)):
            await PantographClient.create(replace(env, server_start_timeout_s=0.2))
        assert asyncio.get_running_loop().time() - start < 1

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize("scenario", ["failure", "empty", "invalid_utf8", "oversized"])
def test_resolver_bad_output_is_configuration_error(setup, monkeypatch, scenario):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_RESOLVER", scenario)

    async def body():
        with pytest.raises(PantographConfigurationError):
            await PantographClient.create(replace(env, response_line_limit=1024))

    run(body)
    assert_reaped(report)
    assert not any(row.get("executable") == "repl" for row in records(report))


@pytest.mark.parametrize("phase", ["resolver", "readiness", "reply"])
def test_startup_and_request_cancellation_reap_children(setup, monkeypatch, phase):
    env, report = setup
    monkeypatch.setenv(
        "PANTOGRAPH_FAKE_RESOLVER", "ignore_term" if phase == "resolver" else "normal"
    )
    monkeypatch.setenv(
        "PANTOGRAPH_FAKE_SCENARIO",
        "ready_delay" if phase == "readiness" else "reply_delay",
    )

    async def body():
        client = None
        if phase == "reply":
            client = await PantographClient.create(env)
            task = asyncio.create_task(client.stat())
            await wait_event(report, "request", command="stat")
        else:
            task = asyncio.create_task(PantographClient.create(env))
            await wait_event(
                report, "spawn", executable="lake" if phase == "resolver" else "repl"
            )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        if client is not None:
            assert client.phase is SessionPhase.CLOSED
            with pytest.raises(PantographTransportError):
                await client.stat()

    run(body)
    assert_reaped(report)


def test_resolver_timeout_terminates_and_kills_if_necessary(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_RESOLVER", "ignore_term")

    async def body():
        with pytest.raises(PantographStartupError):
            await PantographClient.create(replace(env, server_start_timeout_s=0.15))

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize(
    "scenario",
    [
        "malformed",
        "invalid_utf8",
        "oversized",
        "partial",
        "reply_eof",
        "invalid_schema",
    ],
)
def test_bad_response_retires_session_before_raising(setup, monkeypatch, scenario):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", scenario)

    async def body():
        client = await PantographClient.create(replace(env, response_line_limit=1024))
        with pytest.raises((PantographTransportError, PantographProtocolError)):
            await client.stat()
        assert client.phase is SessionPhase.CLOSED and not client.is_usable
        with pytest.raises((PantographTransportError, PantographProtocolError)):
            await client.stat()
        await client.aclose()

    run(body)
    assert_reaped(report)


def test_request_deadline_prevents_late_reply_reuse(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "reply_delay")

    async def body():
        client = await PantographClient.create(replace(env, timeout=0.05))
        with pytest.raises(PantographRequestTimeout):
            await client.stat()
        assert client.phase is SessionPhase.CLOSED
        with pytest.raises(PantographRequestTimeout):
            await client.stat()
        assert (
            len([row for row in records(report) if row.get("command") == "stat"]) == 1
        )

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize("cancel", [True, False])
def test_lock_waiter_does_not_retire_another_transaction(setup, monkeypatch, cancel):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "reply_delay")

    async def body():
        client = await PantographClient.create(env)
        try:
            active = asyncio.create_task(client.stat())
            await wait_event(report, "request", command="stat")
            waiter = asyncio.create_task(
                client._exchange(
                    "stat", {}, deadline=asyncio.get_running_loop().time() + 0.04
                )
            )
            await asyncio.sleep(0)
            if cancel:
                waiter.cancel()
            with pytest.raises(
                asyncio.CancelledError if cancel else PantographRequestTimeout
            ):
                await waiter
            assert (await active).n_goals == 0
            assert client.is_usable
            assert (await client.stat()).n_goals == 0
        finally:
            await client.aclose()

    run(body)
    assert_reaped(report)


def test_concurrent_transactions_and_independent_clients(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "reply_delay")

    async def body():
        clients = await asyncio.gather(
            PantographClient.create(env), PantographClient.create(env)
        )
        try:
            start = asyncio.get_running_loop().time()
            results = await asyncio.gather(*(client.stat() for client in clients))
            assert [result.n_goals for result in results] == [0, 0]
            assert asyncio.get_running_loop().time() - start < 0.5
            starts = await asyncio.gather(
                *(
                    clients[0]._exchange("goal.start", {"expr": str(index)})
                    for index in range(12)
                )
            )
            assert sorted(result.state_id for result in starts) == list(range(12))
            assert (await clients[0].stat()).n_goals == 12
            assert clients[0].session_id != clients[1].session_id
        finally:
            await asyncio.gather(*(client.aclose() for client in clients))

    run(body)
    assert_reaped(report)


def test_concurrent_close_and_repeated_cancellation_join_cleanup(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "ignore_term")

    async def body():
        client = await PantographClient.create(env)
        first = asyncio.create_task(client.aclose())
        second = asyncio.create_task(client.aclose())
        await asyncio.sleep(0.02)
        first.cancel("original close cancellation")
        await asyncio.sleep(0.02)
        first.cancel("later cancellation")
        with pytest.raises(asyncio.CancelledError, match="original close cancellation"):
            await first
        await second
        await client.aclose()
        assert client.phase is SessionPhase.CLOSED

    run(body)
    assert_reaped(report)


def test_close_does_not_wait_for_request_lock(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "reply_delay")

    async def body():
        client = await PantographClient.create(env)
        active = asyncio.create_task(client.stat())
        await wait_event(report, "request", command="stat")
        await client.aclose()
        with pytest.raises(PantographTransportError):
            await active
        assert client.phase is SessionPhase.CLOSED

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize("cancel", [True, False])
def test_write_drain_failure_or_cancellation_retires_session(
    setup, monkeypatch, cancel
):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "no_read")

    async def body():
        client = await PantographClient.create(replace(env, timeout=0.1))
        task = asyncio.create_task(
            client._exchange("goal.start", {"expr": "x" * (2 * 1024 * 1024)})
        )
        await asyncio.sleep(0.02)
        if cancel:
            task.cancel()
        with pytest.raises(
            asyncio.CancelledError if cancel else PantographRequestTimeout
        ):
            await task
        assert client.phase is SessionPhase.CLOSED

    run(body)
    assert_reaped(report)


def test_bookkeeping_error_invalidates_session_before_return(setup):
    env, report = setup

    async def body():
        client = await PantographClient.create(env)

        def fail(_result):
            raise ValueError("ownership failed")

        with pytest.raises(PantographProtocolError, match="ownership"):
            await client._exchange("goal.start", {"expr": "True"}, record=fail)
        assert client.phase is SessionPhase.CLOSED

    run(body)
    assert_reaped(report)


def test_domain_rejection_and_prewrite_validation_leave_transport_usable(setup):
    env, report = setup

    async def body():
        client = await PantographClient.create(env)
        try:
            with pytest.raises(PantographProtocolError):
                await client._exchange("goal.tactic", {"stateId": 0, "tactic": "skip"})
            with pytest.raises(PantographGoalRejected):
                await client._exchange("goal.start", {"expr": "reject"})
            assert client.is_usable
            assert (await client.stat()).n_goals == 0
        finally:
            await client.aclose()

    run(body)
    assert_reaped(report)


def test_live_allocation_id_reuse_is_fatal(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "duplicate_allocation")

    async def body():
        client = await PantographClient.create(env)
        await client._exchange("goal.start", {"expr": "True"})
        with pytest.raises(PantographProtocolError, match="reused"):
            await client._exchange("goal.start", {"expr": "True"})
        assert not client.is_usable

    run(body)
    assert_reaped(report)


def test_cross_loop_use_fails_before_io(setup):
    env, report = setup

    async def body():
        client = await PantographClient.create(env)

        def foreign():
            async def calls():
                for action in (client.stat, client.aclose):
                    with pytest.raises(PantographStateError, match="owning event loop"):
                        await action()

            asyncio.run(calls())

        try:
            await asyncio.to_thread(foreign)
            assert (await client.stat()).n_goals == 0
        finally:
            await client.aclose()

    run(body)
    assert_reaped(report)


@pytest.mark.parametrize(
    "change",
    [
        {"source_root": None},
        {"pantograph_repl": None},
        {"imports": ()},
        {"imports": "Init"},
        {"imports": ("--flag",)},
        {"imports": ("Init\nstat",)},
        {"options": {"noRepeat": True}},
        {"options": {"unknown": True}},
        {"timeout": True},
        {"timeout": float("nan")},
        {"server_start_timeout_s": 0},
        {"worker_shutdown_timeout_s": -1},
        {"response_line_limit": True},
        {"stderr_tail_limit": 0},
    ],
)
def test_invalid_launch_settings_fail_without_spawning(setup, change):
    env, report = setup
    with pytest.raises(PantographConfigurationError):
        build_launch_spec(replace(env, **change))
    assert records(report) == []


@pytest.mark.parametrize(
    "missing",
    [
        "source_toolchain",
        "repl_toolchain",
        "source_lakefile",
        "repl_lakefile",
        "executable",
        "lake",
        "stdbuf",
    ],
)
def test_missing_launch_dependencies_are_not_silently_accepted(
    setup, monkeypatch, missing
):
    env, report = setup
    if missing == "source_toolchain":
        (env.source_root / "lean-toolchain").unlink()
    elif missing == "repl_toolchain":
        (env.pantograph_repl.parents[3] / "lean-toolchain").unlink()
    elif missing == "source_lakefile":
        (env.source_root / "lakefile.lean").unlink()
    elif missing == "repl_lakefile":
        (env.pantograph_repl.parents[3] / "lakefile.lean").unlink()
    elif missing == "executable":
        env.pantograph_repl.chmod(0o644)
    else:
        original = shutil.which
        monkeypatch.setattr(
            "maths_ai.hybrid_reasoner.pantograph_launch.shutil.which",
            lambda name: None if name == missing else original(name),
        )
    with pytest.raises(PantographConfigurationError):
        build_launch_spec(env)
    assert records(report) == []


def test_toolchain_mismatch_is_rejected(setup):
    env, _report = setup
    (env.pantograph_repl.parents[3] / "lean-toolchain").write_text(
        "leanprover/lean4:v4.29.1\n"
    )
    with pytest.raises(PantographConfigurationError, match="toolchain mismatch"):
        build_launch_spec(env)


def test_external_executable_requires_matching_explicit_manifest(setup, tmp_path):
    env, _report = setup
    external = tmp_path / "external-repl"
    shutil.copyfile(env.pantograph_repl, external)
    external.chmod(0o755)
    with pytest.raises(PantographConfigurationError, match="build_manifest"):
        build_launch_spec(replace(env, pantograph_repl=external))
    manifest = tmp_path / "manifest.json"
    value = {
        "toolchain": "leanprover/lean4:v4.10.0-rc1",
        "repl_sha256": hashlib.sha256(external.read_bytes()).hexdigest(),
    }
    manifest.write_text(json.dumps(value))
    assert (
        build_launch_spec(
            replace(env, pantograph_repl=external, build_manifest=manifest)
        ).repl
        == external
    )
    value["repl_sha256"] = "wrong"
    manifest.write_text(json.dumps(value))
    with pytest.raises(PantographConfigurationError, match="hash mismatch"):
        build_launch_spec(
            replace(env, pantograph_repl=external, build_manifest=manifest)
        )


@pytest.mark.parametrize("cancel", [True, False])
@pytest.mark.parametrize("flood", [True, False])
def test_spawn_cancellation_or_deadline_retains_process_ownership(
    setup, monkeypatch, cancel, flood
):
    from maths_ai.hybrid_reasoner.pantograph_client import _ProcessOwner

    env, report = setup
    if flood:
        monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "stderr_flood")
    original = _ProcessOwner.spawn

    async def paused(owner, *command, **kwargs):
        process = await original(owner, *command, **kwargs)
        if "-oL" in command:
            await asyncio.sleep(5)
        return process

    monkeypatch.setattr(_ProcessOwner, "spawn", paused)

    async def body():
        # A cancellation must arrive before the startup deadline, or the
        # deadline wins the race and the test checks the other path.
        timeout = 2 if cancel else 0.2
        task = asyncio.create_task(
            PantographClient.create(replace(env, server_start_timeout_s=timeout))
        )
        if cancel:
            await wait_event(report, "spawn", executable="repl")
            task.cancel("original cancellation")
        with pytest.raises(
            asyncio.CancelledError if cancel else PantographStartupError
        ):
            await task

    run(body)
    assert_reaped(report)


def test_helper_stderr_flood_is_drained_independently_of_stdout_limit(
    setup, monkeypatch
):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_RESOLVER", "stderr_flood")

    async def body():
        client = await PantographClient.create(
            replace(env, stderr_tail_limit=128, response_line_limit=1024)
        )
        await client.aclose()

    run(body)
    assert_reaped(report)


def test_unsolicited_response_is_not_returned_as_a_valid_transaction(
    setup, monkeypatch
):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "unsolicited_after_reply")

    async def body():
        client = await PantographClient.create(env)
        with pytest.raises(PantographProtocolError, match="Unsolicited"):
            await client.stat()
        assert client.phase is SessionPhase.CLOSED

    run(body)
    assert_reaped(report)


def test_idle_stdout_violation_starts_retained_cleanup_without_a_request(
    setup, monkeypatch
):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "unsolicited_idle")

    async def body():
        client = await PantographClient.create(env)
        for _attempt in range(100):
            if client.phase is SessionPhase.CLOSED:
                break
            await asyncio.sleep(0.01)
        assert client.phase is SessionPhase.CLOSED
        with pytest.raises(PantographProtocolError):
            await client.stat()
        await client.aclose()

    run(body)
    assert_reaped(report)


def test_expired_deadline_never_writes_even_when_lock_is_free(setup):
    env, report = setup

    async def body():
        client = await PantographClient.create(env)
        try:
            with pytest.raises(PantographRequestTimeout):
                await client._exchange(
                    "stat", {}, deadline=asyncio.get_running_loop().time() - 1
                )
            assert client.is_usable
            assert not any(row.get("command") == "stat" for row in records(report))
        finally:
            await client.aclose()

    run(body)
    assert_reaped(report)


def test_request_deadline_includes_synchronous_allocation_bookkeeping(setup):
    env, report = setup

    async def body():
        client = await PantographClient.create(env)

        def slow(_response):
            time.sleep(0.05)

        with pytest.raises(PantographRequestTimeout):
            await client._exchange(
                "goal.start",
                {"expr": "True"},
                deadline=asyncio.get_running_loop().time() + 0.025,
                record=slow,
            )
        assert client.phase is SessionPhase.CLOSED

    run(body)
    assert_reaped(report)


def test_close_reports_failure_to_confirm_reaping_instead_of_claiming_closed(
    setup, monkeypatch
):
    env, report = setup

    async def body():
        client = await PantographClient.create(env)

        async def never_reports_exit():
            await asyncio.Event().wait()

        monkeypatch.setattr(client._owner.process, "wait", never_reports_exit)
        with pytest.raises(
            PantographTransportError, match="reaping could not be confirmed"
        ):
            await client.aclose()
        assert not client.is_usable and client.phase is SessionPhase.CLOSING
        with pytest.raises(PantographTransportError):
            await client.aclose()

    run(body)
    assert_reaped(report)


def test_repeated_postwrite_cancellation_preserves_the_original_context(
    setup, monkeypatch
):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "ignore_term")

    async def body():
        client = await PantographClient.create(env)
        task = asyncio.create_task(
            client._exchange(
                "goal.start",
                {"expr": "x" * (2 * 1024 * 1024)},
                record=lambda _result: None,
            )
        )
        # Ensure a request is awaiting the pipe before cancelling it twice.
        await asyncio.sleep(0)
        task.cancel("original cancellation")
        await asyncio.sleep(0.02)
        task.cancel("later cancellation")
        with pytest.raises(asyncio.CancelledError, match="original cancellation"):
            await task
        assert client.phase is SessionPhase.CLOSED

    run(body)
    assert_reaped(report)


def test_request_deadline_includes_synchronous_decoding(setup, monkeypatch):
    import maths_ai.hybrid_reasoner.pantograph_client as transport

    env, report = setup
    original = transport.decode_response

    async def body():
        client = await PantographClient.create(env)

        def slow(command, payload):
            time.sleep(0.05)
            return original(command, payload)

        monkeypatch.setattr(transport, "decode_response", slow)
        with pytest.raises(PantographRequestTimeout):
            await client._exchange(
                "stat", {}, deadline=asyncio.get_running_loop().time() + 0.025
            )
        assert client.phase is SessionPhase.CLOSED

    run(body)
    assert_reaped(report)


def test_spawn_failure_before_readers_exist_is_cleaned_up(setup, monkeypatch):
    from maths_ai.hybrid_reasoner.pantograph_client import _ProcessOwner

    env, report = setup
    original = _ProcessOwner.spawn

    async def fail_repl(owner, *command, **kwargs):
        if "-oL" in command:
            raise FileNotFoundError("executable disappeared")
        return await original(owner, *command, **kwargs)

    monkeypatch.setattr(_ProcessOwner, "spawn", fail_repl)

    async def body():
        with pytest.raises(PantographStartupError, match="executable disappeared"):
            await PantographClient.create(env)

    run(body)
    assert_reaped(report)


def test_closed_child_stdin_is_a_typed_transport_failure(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_SCENARIO", "broken_pipe")

    async def body():
        client = await PantographClient.create(env)
        await asyncio.sleep(0.02)
        with pytest.raises(PantographTransportError):
            await client.stat()
        assert client.phase is SessionPhase.CLOSED

    run(body)
    assert_reaped(report)


def test_non_gnu_stdbuf_is_rejected_before_repl_spawn(setup):
    env, report = setup
    launcher = report.parent / "tools" / "stdbuf"
    shutil.copyfile(FAKE, launcher)
    launcher.chmod(0o755)

    async def body():
        with pytest.raises(PantographConfigurationError, match="not GNU"):
            await PantographClient.create(env)

    run(body)
    assert_reaped(report)
    assert not any(row.get("executable") == "repl" for row in records(report))


def test_repeated_resolver_cancellation_preserves_original_context(setup, monkeypatch):
    env, report = setup
    monkeypatch.setenv("PANTOGRAPH_FAKE_RESOLVER", "ignore_term")

    async def body():
        task = asyncio.create_task(PantographClient.create(env))
        await wait_event(report, "spawn", executable="lake")
        task.cancel("original resolver cancellation")
        await asyncio.sleep(0.02)
        task.cancel("later cancellation")
        with pytest.raises(
            asyncio.CancelledError, match="original resolver cancellation"
        ):
            await task

    run(body)
    assert_reaped(report)
