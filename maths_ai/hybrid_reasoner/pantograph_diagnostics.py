"""CPU-only live verification of the owned execution and model contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys
import subprocess

from .pantograph_client import PantographClient
from .pantograph_env import PantographEnv
from .pantograph_protocol import (
    PROTOCOL_DESCRIPTION,
    PantographProtocolError,
    PantographTacticRejected,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PantographProtocolError(message)


async def probe_model_client(client: PantographClient) -> None:
    """Require actual normalized instance/let metadata and both proof outcomes."""
    baseline = (await client.stat()).n_goals
    async with client.state_scope() as scope:
        root = await scope.start_goal(
            "∀ (α : Type) [inst : Inhabited α] (x : α), x = x"
        )
        state = await scope.apply_tactic(root, "intro α inst x")
        goal = state.goals[0].goal.require_model_state()
        require(
            len(goal.locals) == 3 and goal.locals[1].is_instance,
            "Instance probe did not preserve its ordered local context.",
        )
        require(
            "FV0" in goal.locals[2].type_model_sexp and "FV2" in goal.goal_model_sexp,
            "Instance probe did not return normalized model expressions.",
        )
        require(
            not (await scope.apply_tactic(state, "rfl")).goals,
            "Valid probe did not close.",
        )
        root = await scope.start_goal("∀ (n : Nat), let x := n; x = x")
        state = await scope.apply_tactic(root, "intro n x")
        local = state.goals[0].locals[-1]
        require(
            local.is_let
            and local.value is not None
            and local.value.model_sexp == "(:fv FV0)",
            "Let probe lost its normalized value.",
        )
        require(
            not (await scope.apply_tactic(state, "rfl")).goals,
            "Local-let probe did not close.",
        )
        root = await scope.start_goal("True")
        try:
            await scope.apply_tactic(root, "exact (by sorry)")
        except PantographTacticRejected:
            pass
        else:
            raise PantographProtocolError("Admission probe was accepted.")
        require(
            not (await scope.apply_tactic(root, "trivial")).goals,
            "Source state was not reusable after admission rejection.",
        )
    require((await client.stat()).n_goals == baseline, "Model probe leaked states.")


async def verify_live_client(env: PantographEnv, *, proofs: int = 100) -> dict:
    """Exercise the actual client without a checkpoint, dataset, or GPU."""
    if type(proofs) is not int or proofs < 100:
        raise ValueError("The designated leak check requires at least 100 proofs.")
    from maths_ai.gnn_inference.atp_lean_gnn.graph import (
        dag_fingerprint,
        model_goal_to_dag,
    )

    client = await PantographClient.create(env)
    pid = client.pid
    outcomes = []
    try:
        baseline = (await client.stat()).n_goals
        await probe_model_client(client)
        outcomes.append("instance-let-valid-and-admission-probes")
        golden_path = (
            Path(__file__).parents[1]
            / "gnn_inference/tests/fixtures/pantograph/canonical_graph_expectations.json"
        )
        goldens = json.loads(golden_path.read_text())["goals"]
        cases = (
            ("identity_intro", "∀ (p : Prop), p → p", "intro p h"),
            ("conjunction_split", "True ∧ True", "constructor"),
            (
                "instance_intro",
                "∀ (α : Type) [inst : Inhabited α] (x : α), x = x",
                "intro α inst x",
            ),
            ("let_intro", "∀ (n : Nat), let x := n; x = x", "intro n x"),
        )
        async with client.state_scope() as scope:
            for label, expression, tactic in cases:
                root = await scope.start_goal(expression)
                state = await scope.apply_tactic(root, tactic)
                require(
                    len(state.goals) == len(goldens[label]),
                    f"Goal count changed: {label}",
                )
                for execution_goal, expected in zip(state.goals, goldens[label]):
                    goal = execution_goal.goal
                    require(
                        goal.state_fingerprint() == expected["state_fingerprint"],
                        f"Canonical model state changed: {label}",
                    )
                    dag = model_goal_to_dag(goal)
                    require(
                        dag_fingerprint(dag) == expected["dag_fingerprint"],
                        f"DAG changed: {label}",
                    )
                    require(
                        [node.as_dict() for node in dag.nodes] == expected["nodes"],
                        f"Features changed: {label}",
                    )
                    require(
                        [list(edge) for edge in dag.edges] == expected["edges"],
                        f"Edges changed: {label}",
                    )
                    require(
                        dag.expression_root_id == expected["expression_root_id"]
                        and dag.state_root_id == expected["state_root_id"],
                        f"Graph roots changed: {label}",
                    )
                    replay = goal.replay_spec()
                    replay_root = await scope.start_goal(replay.expression)
                    replayed = await scope.apply_tactic(
                        replay_root,
                        (
                            f"intro {' '.join(replay.local_names)}"
                            if replay.local_names
                            else "skip"
                        ),
                    )
                    require(
                        len(replayed.goals) == 1
                        and replayed.goals[0].goal.state_fingerprint()
                        == goal.state_fingerprint(),
                        f"Canonical replay round trip changed: {label}",
                    )
                outcomes.append(label)
        async with client.state_scope() as scope:
            root = await scope.start_goal("∀ (p : Prop), p → p")
            child = await scope.apply_tactic(root, "intro p h")
            await scope.release(root)
            first = await scope.apply_tactic(child, "exact h")
            await scope.release(first)
            require(
                not (await scope.apply_tactic(child, "assumption")).goals,
                "Independent sibling alternative was invalidated.",
            )
            root = await scope.start_goal("True ∧ True")
            split = await scope.apply_tactic(root, "constructor")
            remaining = await scope.apply_tactic(
                split, "all_goals trivial", goal_index=1
            )
            require(
                len(remaining.goals) == 1, "Focused tactic lost unresolved sibling."
            )
            require(
                not (await scope.apply_tactic(remaining, "trivial")).goals,
                "Automatic sibling resumption failed.",
            )
            root = await scope.start_goal("True")
            for tactic in (
                "exact False",
                "exact (",
                'run_tac Lean.logError "expected diagnostic"\ntrivial',
            ):
                try:
                    await scope.apply_tactic(root, tactic)
                except PantographTacticRejected:
                    pass
                else:
                    raise PantographProtocolError(f"Expected rejection: {tactic}")
            good = await scope.apply_tactic(
                root,
                'run_tac Lean.logWarning "warning evidence"\nrun_tac Lean.logInfo "information evidence"\ntrivial',
            )
            require(
                not good.goals
                and {m.severity for m in good.messages} == {"warning", "information"},
                "Diagnostics or multiline execution changed.",
            )
            unsafe = """run_tac
  let info : Lean.DefinitionVal := {
    name := `OwnedClientUnsafeEvidence, levelParams := [], type := Lean.mkConst ``True,
    value := Lean.mkConst ``True.intro, hints := .opaque, safety := .unsafe
  }
  Lean.addDecl (.defnDecl info)
  (← Lean.Elab.Tactic.getMainGoal).assign (Lean.mkConst `OwnedClientUnsafeEvidence)
  Lean.Elab.Tactic.setGoals []"""
            try:
                await scope.apply_tactic(root, unsafe)
            except PantographTacticRejected as exc:
                require(
                    getattr(exc.payload, "validation", None) is not None
                    and exc.payload.validation.has_unsafe,
                    "Unsafe example failed before reaching the proof scanner.",
                )
            else:
                raise PantographProtocolError("Unsafe proof was accepted.")
        outcomes.extend(
            (
                "parent-deletion-and-independent-alternatives",
                "focus-and-sibling-resumption",
                "failure-recovery-diagnostics-multiline",
                "positive-unsafe-rejection",
            )
        )
        async with client.state_scope() as scope:
            for expression, intro, closing in (
                ("let x : Nat := 1; x = x", "intro x", "rfl"),
                (
                    "∀ (n : Nat), let x := n; ∀ (h : x = n), x = n",
                    "intro n x h",
                    "exact h",
                ),
            ):
                start = await scope.start_goal(expression)
                state = await scope.apply_tactic(start, intro)
                goal = state.goals[0].goal
                require(
                    any(local.is_let for local in goal.locals),
                    "Local-let regression lost its definition.",
                )
                replay = goal.replay_spec()
                replayed = await scope.apply_tactic(
                    await scope.start_goal(replay.expression),
                    f"intro {' '.join(replay.local_names)}",
                )
                require(
                    replayed.goals[0].goal.state_fingerprint()
                    == goal.state_fingerprint(),
                    "Dependent local-let replay changed the model state.",
                )
                require(
                    not (await scope.apply_tactic(replayed, closing)).goals,
                    "Known local-let closing tactic failed after replay.",
                )
        outcomes.append("constant-and-dependent-let-replay")
        for _ in range(proofs):
            async with client.state_scope() as scope:
                root = await scope.start_goal("∀ (p : Prop), p → p")
                require(
                    not (await scope.apply_tactic(root, "intro p h\nexact h")).goals,
                    "Repeated identity proof failed.",
                )
            require(
                (await client.stat()).n_goals == baseline,
                "Repeated workload leaked a state.",
            )
        report = {
            "python": sys.executable,
            "python_version": sys.version,
            "repository_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parents[2],
                text=True,
                timeout=5,
            ).strip(),
            "python_sources_sha256": {
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in (
                    "pantograph_client.py",
                    "pantograph_protocol.py",
                    "pantograph_model_sexpr.py",
                    "pantograph_diagnostics.py",
                )
            },
            "repl": str(env.pantograph_repl),
            "repl_sha256": hashlib.sha256(
                Path(env.pantograph_repl).read_bytes()
            ).hexdigest(),
            "source_root": str(env.source_root),
            "toolchain": client.launch_spec.toolchain,
            "imports": list(env.imports),
            "options": client.effective_options.to_dict(),
            "descriptor": dict(PROTOCOL_DESCRIPTION),
            "checks": outcomes,
            "scoped_proofs": proofs,
            "initial_states": baseline,
            "final_states": (await client.stat()).n_goals,
            "pid": pid,
            "complete_theorem_certification": False,
        }
    finally:
        await client.aclose()
    # CLOSED is reached only after the transport confirms process reaping and
    # joins its readers; consumers do not inspect the private subprocess.
    report.update(
        phase=client.phase.value,
        stderr_tail=client.stderr_tail,
        subprocess_reaped=client.phase.value == "CLOSED",
    )
    return report
