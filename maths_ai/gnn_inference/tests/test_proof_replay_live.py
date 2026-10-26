"""Replay combination against the actual snapshot companion REPL.

Branches are explored independently in retained snapshots and combined by
sequential replay; only a kernel certificate counts. Run with
``--run-pantograph-live`` and the usual ``PANTOGRAPH_TEST_*`` paths.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path

import pytest

from maths_ai.hybrid_reasoner.execution_types import DeclarationKey
from maths_ai.hybrid_reasoner.proof_recipes import ArtifactArgument, ReplayArtifact
from maths_ai.hybrid_reasoner.proof_replay import (
    ArtifactMismatch,
    ReplayCoordinator,
    ReplayStatus,
    build_artifact,
    reverify,
)
from maths_ai.gnn_inference.tests.snapshot_explorer import Explorer
from maths_ai.lean_replay import render_replay, seed_from_text
from pantograph_async import PantographClient, PantographEnv


@pytest.fixture
def live_env(request):
    source = os.environ.get("PANTOGRAPH_TEST_SOURCE_ROOT")
    repl = os.environ.get("PANTOGRAPH_TEST_REPL")
    if not source or not repl:
        if request.config.getoption("--run-pantograph-live"):
            pytest.fail(
                "Required live run needs PANTOGRAPH_TEST_SOURCE_ROOT and PANTOGRAPH_TEST_REPL."
            )
        pytest.skip("Explicit live paths not configured.")
    imports = tuple(os.environ.get("PANTOGRAPH_TEST_IMPORTS", "Mathlib").split(","))
    return PantographEnv(source_root=Path(source), pantograph_repl=Path(repl), imports=imports)


def scoped(env, body):
    async def run():
        async with await PantographClient.create(env) as client:
            async with client.state_scope() as scope:
                result = await body(scope)
            assert (await client.stat()).n_goals == 0
            return result

    return asyncio.run(run())


async def combine(explorer):
    coordinator = ReplayCoordinator(explorer.registry)
    return await coordinator.combine(explorer, explorer.root, 0)


@pytest.mark.pantograph_live
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_a_conjunction_combines_from_either_exploration_order(live_env, order):
    async def body(scope):
        explorer = await Explorer.start(scope, "True ∧ True")
        sides = await explorer.expand(0, "constructor")
        for index in order:
            assert await explorer.expand(sides[index], "trivial") == []
        result = await combine(explorer)
        assert result.status is ReplayStatus.CERTIFIED, result.conflicts
        assert [step.command for step in result.witness] == ["constructor", "trivial", "trivial"]
        assert all(step.observation_matched for step in result.witness)
        await explorer.registry.close()

    scoped(live_env, body)


@pytest.mark.pantograph_live
def test_conflicting_witness_branches_never_certify(live_env):
    async def body(scope):
        explorer = await Explorer.start(scope, "∃ n : Nat, n = 0 ∧ n = 1")
        w, a, b = await explorer.expand(0, "refine ⟨?w, ?a, ?b⟩")
        # Each equality succeeds on its own by choosing its own witness.
        assert await explorer.expand(a, "exact (rfl : 0 = 0)") == []
        assert await explorer.expand(b, "exact (rfl : 1 = 1)") == []
        assert await explorer.expand(w, "exact 0") == []
        assert await explorer.expand(w, "exact 1") == []
        result = await combine(explorer)
        assert result.status is ReplayStatus.CONFLICT
        assert result.certificate is None
        assert {conflict.kind for conflict in result.conflicts} == {"rejected"}
        # The changed equality is kept for a fresh search of its actual state.
        assert sorted((c.node, c.occurrence.goal.expression) for c in result.continuations) == [
            (a, "1 = 0"),
            (b, "0 = 1"),
        ]
        for continuation in result.continuations:
            await explorer.registry.release(continuation.occurrence.lease)
        await explorer.registry.close()

    scoped(live_env, body)


@pytest.mark.pantograph_live
def test_another_alternative_certifies_after_a_conflict(live_env):
    async def body(scope):
        explorer = await Explorer.start(scope, "(∃ n : Nat, n = 0 ∧ n = 1) ∨ True")
        (exists,) = await explorer.expand(0, "left")
        w, a, b = await explorer.expand(exists, "refine ⟨?_, ?_, ?_⟩")
        await explorer.expand(a, "exact (rfl : 0 = 0)")
        await explorer.expand(b, "exact (rfl : 1 = 1)")
        await explorer.expand(w, "exact 0")
        (truth,) = await explorer.expand(0, "right")
        await explorer.expand(truth, "trivial")
        result = await combine(explorer)
        assert result.status is ReplayStatus.CERTIFIED
        assert result.conflicts and result.witness[0].command == "right"
        await explorer.registry.close()

    scoped(live_env, body)


@pytest.mark.pantograph_live
def test_a_sibling_solved_by_a_shared_assignment_needs_no_step(live_env):
    async def body(scope):
        explorer = await Explorer.start(scope, "∃ n : Nat, n = 0")
        children = await explorer.expand(0, "constructor")
        names = [explorer.observation(node).expression for node in children]
        assert names == ["?w = 0", "ℕ"]
        proof, witness = children
        await explorer.expand(proof, "rfl")
        # Explored on its own, the witness would be 1; replay never runs that.
        await explorer.expand(witness, "exact 1")
        result = await combine(explorer)
        assert result.status is ReplayStatus.CERTIFIED
        assert [step.command for step in result.witness] == ["constructor", "rfl"]
        await explorer.registry.close()

    scoped(live_env, body)


@pytest.mark.pantograph_live
def test_identical_targets_keep_their_slots_in_any_order(live_env):
    async def body(scope):
        explorer = await Explorer.start(scope, "True ∧ True ∧ True")
        parts = await explorer.expand(0, "refine ⟨?_, ?_, ?_⟩")
        assert len(parts) == 3 and len({explorer.observation(p).expression for p in parts}) == 1
        for index in (2, 0, 1):
            await explorer.expand(parts[index], "trivial")
        result = await combine(explorer)
        assert result.status is ReplayStatus.CERTIFIED
        assert [step.obligation.slot for step in result.witness[1:]] == [0, 1, 2]
        await explorer.registry.close()

    scoped(live_env, body)


async def _introduced_proof(scope):
    explorer = await Explorer.start(scope, "∀ (p q : Prop), p → q → p ∧ q")
    (body,) = await explorer.expand(0, "intro p q hp hq")
    left, right = await explorer.expand(body, "constructor")
    await explorer.expand(left, "exact", "hp")
    await explorer.expand(right, "exact", "hq")
    return explorer


@pytest.mark.pantograph_live
def test_introduced_locals_are_rebound_through_their_keys(live_env):
    async def body(scope):
        explorer = await _introduced_proof(scope)
        result = await combine(explorer)
        assert result.status is ReplayStatus.CERTIFIED
        # The replay's intro created new locals; arguments follow their keys.
        exact = [step for step in result.witness if step.command.startswith("exact")]
        assert [step.command for step in exact] == ["exact hp", "exact hq"]
        keys = [step.step.arguments[0].declaration for step in exact]
        assert keys == [DeclarationKey(0, 0, 2), DeclarationKey(0, 0, 3)]
        await explorer.registry.close()
        return result

    scoped(live_env, body)


@pytest.mark.pantograph_live
def test_an_exported_artifact_reverifies_in_a_fresh_session(live_env):
    statement = "∀ (p q : Prop), p → q → p ∧ q"

    async def first(scope):
        explorer = await _introduced_proof(scope)
        result = await combine(explorer)
        await explorer.registry.close()
        spec = render_replay(seed_from_text(goal=statement))
        return build_artifact(
            {"expression": spec.expression, "local_names": list(spec.local_names),
             "universe_levels": list(spec.universe_levels)},
            result,
            {"lean": result.certificate.receipt.lean},
        )

    artifact = ReplayArtifact.from_dict(scoped(live_env, first).as_dict())

    async def again(scope):
        return await reverify(scope, artifact)

    assert scoped(live_env, again).receipt.proof == artifact.receipt["proof"]

    altered = replace(artifact, statement={**artifact.statement, "expression": "∀ (p q : Prop), p → q → q ∧ p"})
    swapped = list(artifact.steps)
    index = next(i for i, step in enumerate(swapped) if step.arguments)
    argument = swapped[index].arguments[0]
    swapped[index] = replace(
        swapped[index],
        arguments=(ArtifactArgument(argument.kind, argument.text, DeclarationKey(0, 0, 3)),),
    )
    for broken in (altered, replace(artifact, steps=tuple(swapped))):
        async def check(scope, broken=broken):
            with pytest.raises(ArtifactMismatch):
                await reverify(scope, broken)

        scoped(live_env, check)
