"""Strict owned-client factory and scoped canonical-goal materialization."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from pantograph_async import (
    PantographClient,
    PantographEnv,
    PantographStartupError,
    PantographStateScope,
    TacticState,
)
from pantograph_async.probes import probe_model_client

from maths_ai.data_models.proof_components import GoalReplaySpec

from .pantograph_adapter import require_client_compatibility


async def create_model_sexpr_client(env: PantographEnv) -> PantographClient:
    """Finish descriptor, options, and scoped model probes under one deadline."""
    require_client_compatibility()
    started = asyncio.get_running_loop().time()
    strict_env = replace(env, imports=tuple(dict.fromkeys((*env.imports, "Mathlib"))))
    client = None
    try:
        async with asyncio.timeout_at(started + strict_env.server_start_timeout_s):
            client = await PantographClient.create(strict_env)
            await probe_model_client(client)
        return client
    except BaseException as exc:
        if client is not None:
            try:
                await client.aclose()
            except BaseException as cleanup:
                if isinstance(exc, asyncio.CancelledError):
                    exc.add_note(f"Model-session cleanup also reported: {cleanup}")
                    raise exc
                raise
        if isinstance(exc, TimeoutError):
            raise PantographStartupError(
                "Model-session startup deadline expired."
            ) from exc
        raise


async def materialize_replay(
    scope: PantographStateScope, replay: GoalReplaySpec
) -> TacticState:
    """Serialize replayed locals, then release the unmaterialized start."""
    start = await scope.start_goal(replay.expression)
    try:
        tactic = (
            f"intro {' '.join(replay.local_names)}" if replay.local_names else "skip"
        )
        return await scope.apply_tactic(start, tactic)
    finally:
        if scope.is_usable:
            await scope.release(start)
