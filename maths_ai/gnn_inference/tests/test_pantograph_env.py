"""Strict shared environment and factory; no upstream package or bundled REPL."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from maths_ai.hybrid_reasoner.pantograph_env import PantographEnv
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import create_model_sexpr_client
from maths_ai.hybrid_reasoner.pantograph_protocol import (
    PantographConfigurationError,
    PantographProtocolError,
    PantographStartupError,
)
from maths_ai.gnn_inference.tests.test_pantograph_client import (
    setup,
    run,
    assert_reaped,
)


def test_explicit_paths_are_required():
    with pytest.raises(TypeError):
        PantographEnv()
    with pytest.raises(PantographConfigurationError):
        PantographEnv(source_root=None, pantograph_repl=None).verify()


def test_environment_uses_owned_launch_validation(setup):
    env, _report = setup
    env.verify()
    assert "stdbuf -oL" in env.describe()
    for changed in (
        replace(env, source_root=Path("/nonexistent")),
        replace(env, pantograph_repl=Path("/nonexistent/repl")),
        replace(env, options={"noRepeat": True}),
    ):
        with pytest.raises(PantographConfigurationError):
            changed.verify()


def test_factory_finishes_scoped_probe_or_cleans_up_failure(setup, monkeypatch):
    env, report = setup
    visited = []

    async def probe(client):
        visited.append(client)
        async with client.state_scope() as scope:
            await scope.start_goal("True")

    monkeypatch.setattr(
        "maths_ai.hybrid_reasoner.pantograph_model_sexpr.probe_model_client", probe
    )

    async def body():
        async with await create_model_sexpr_client(env) as client:
            assert visited == [client]
            assert "Mathlib" in client.launch_spec.imports
            assert (await client.stat()).n_goals == 0

    run(body)
    assert_reaped(report)


def test_factory_failure_reaps_started_client(setup, monkeypatch):
    env, report = setup

    async def reject(client):
        async with client.state_scope() as scope:
            await scope.start_goal("True")
            raise PantographProtocolError("model mismatch")

    monkeypatch.setattr(
        "maths_ai.hybrid_reasoner.pantograph_model_sexpr.probe_model_client", reject
    )

    async def body():
        with pytest.raises(PantographProtocolError, match="model mismatch"):
            await create_model_sexpr_client(env)

    run(body)
    assert_reaped(report)


def test_factory_cancellation_awaits_probe_and_process_cleanup(setup, monkeypatch):
    env, report = setup

    async def body():
        entered = asyncio.Event()

        async def probe(client):
            async with client.state_scope() as scope:
                await scope.start_goal("True")
                entered.set()
                await asyncio.Event().wait()

        monkeypatch.setattr(
            "maths_ai.hybrid_reasoner.pantograph_model_sexpr.probe_model_client", probe
        )
        task = asyncio.create_task(create_model_sexpr_client(env))
        await entered.wait()
        task.cancel("probe aborted")
        with pytest.raises(asyncio.CancelledError, match="probe aborted"):
            await task

    run(body)
    assert_reaped(report)


def test_factory_startup_budget_includes_model_probe(setup, monkeypatch):
    env, report = setup
    visited = []

    async def probe(client):
        visited.append(client)
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "maths_ai.hybrid_reasoner.pantograph_model_sexpr.probe_model_client", probe
    )

    async def body():
        with pytest.raises(PantographStartupError):
            await create_model_sexpr_client(replace(env, server_start_timeout_s=1))
        assert visited and visited[0].phase.value == "CLOSED"

    run(body)
    assert_reaped(report)
