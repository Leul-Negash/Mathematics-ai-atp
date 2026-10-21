"""CPU-only live verification of MCTS and best-first search over the owned client.

The client's own gate (``verify_live_client``) checks the protocol, canonical
graphs, admission rejection, and at least 100 scoped proofs. This module runs
the search itself against the actual executable: a genuine multi-leaf PUCT
batch, graph equivalence of search nodes, checked rejection of alternatives,
scope closure after a cooperative cutoff, repeated searches on one session,
two independent pool workers sharing one model device, and one bounded
on-policy plus HTPS update. Proposals are scripted, or drawn from a fresh small
model, so the evidence does not depend on a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from maths_ai.data_models.proof_components import TacticCandidate

from .hypergraph import EdgeStatus, NodeStatus, SearchEndReason
from .joint_inference import HybridReasoner, PantographExecutor
from .pantograph_client import PantographClient
from .pantograph_diagnostics import require, verify_live_client
from .pantograph_env import PantographEnv
from .pantograph_model_sexpr import create_model_sexpr_client
from .pantograph_protocol import PantographProtocolError


TARGET_BRANCH = "MCTS-AC-branch"
TARGET_BASE_COMMIT = "89a2417e522720d2cfe3f2e6218614e4d828c8de"
SOURCE_COMMIT = "e8e4b18e081d6dd84491433b00e8cfa596cdfbee"
GOLDEN_PATH = (
    Path(__file__).parents[1]
    / "gnn_inference/tests/fixtures/pantograph/canonical_graph_expectations.json"
)
SEARCH_SOURCES = (
    "maths_ai/hybrid_reasoner/joint_inference.py",
    "maths_ai/hybrid_reasoner/hypergraph.py",
    "maths_ai/hybrid_reasoner/mcts_diagnostics.py",
    "maths_ai/hybrid_reasoner/selection_policy.py",
    "maths_ai/gnn_inference/atp_lean_gnn/rl_reasoner.py",
    "maths_ai/gnn_inference/atp_lean_gnn/rl_live_collection.py",
    "maths_ai/gnn_inference/atp_lean_gnn/rl_training_driver.py",
)
# Each leaf tries an elaboration failure and an admission before the tactic
# that closes it, so every leaf runs several alternatives on one state.
CONJUNCTION_SCRIPT = {
    "True ∧ True": (("constructor", 1.0),),
    "True": (("exact False.elim", 0.5), ("exact (by sorry)", 0.3), ("trivial", 0.2)),
}
PUCT_SETTINGS = {
    "selection_policy": "puct",
    "num_simulations": 8,
    "sim_batch_size": 2,
    "max_depth": 4,
    "max_nodes": 20,
}
LEARNING_TACTICS = {"constructor": 0, "trivial": 1, "intro": 2, "exact": 3}
IDENTITY_SCRIPT = {
    "∀ (p : Prop), p → p": (("intro p h", 1.0),),
    "p": (("exact h", 1.0),),
}


class ScriptedSearchReasoner(HybridReasoner):
    """Deterministic proposals keyed by goal text; no checkpoint or PLN."""

    def __init__(self, server: PantographClient, script, **search) -> None:
        self.script = script
        self.proposal_batches: list[list[int]] = []
        super().__init__(
            None,
            None,
            None,
            executor=PantographExecutor(server),
            use_pln=False,
            **search,
        )

    def _build_gnn_engine(self, **_kwargs):
        return None

    def predict_next_tactics_for_nodes(self, nodes):
        self.proposal_batches.append([node.id for node in nodes])
        return [
            [
                TacticCandidate(tactic_name=tactic, arguments=[], probability=probability)
                for tactic, probability in self.script.get(node.goal.expression, ())
            ]
            for node in nodes
        ]


class _FirstBatchCutoff(ScriptedSearchReasoner):
    """A cooperative deadline that expires after the first simulation batch."""

    def __init__(self, server: PantographClient, script, **search) -> None:
        super().__init__(server, script, **search)
        self._deadline_checks = 0

    def _within_deadline(self, deadline) -> bool:
        self._deadline_checks += 1
        return self._deadline_checks == 1


def _puct_reasoner(client: PantographClient) -> ScriptedSearchReasoner:
    return ScriptedSearchReasoner(client, CONJUNCTION_SCRIPT, **PUCT_SETTINGS)


def _require_graph_equivalence(graph, edge, expected, label: str) -> None:
    from maths_ai.gnn_inference.atp_lean_gnn.graph import (
        dag_fingerprint,
        model_goal_to_dag,
    )

    require(len(edge.child_ids) == len(expected), f"Child count changed: {label}")
    for child_id, golden in zip(edge.child_ids, expected):
        goal = graph.nodes[child_id].goal
        require(
            goal.state_fingerprint() == golden["state_fingerprint"],
            f"Search node state differs from the canonical graph: {label}",
        )
        require(
            dag_fingerprint(model_goal_to_dag(goal)) == golden["dag_fingerprint"],
            f"Search node DAG differs from the canonical graph: {label}",
        )


async def verify_mcts_search(client: PantographClient, *, searches: int = 20) -> list[str]:
    """Run both search modes on one live session and check ownership afterwards."""
    if type(searches) is not int or searches < 2:
        raise ValueError("Repeated-search verification requires at least two searches.")
    goldens = json.loads(GOLDEN_PATH.read_text())["goals"]
    baseline = (await client.stat()).n_goals
    checks = []

    reasoner = _puct_reasoner(client)
    graph = await reasoner.prove("True ∧ True")
    require(graph.is_solved(), "PUCT search did not prove the conjunction.")
    require(
        any(len(batch) > 1 for batch in reasoner.proposal_batches),
        "PUCT search never proposed a multi-leaf batch.",
    )
    root_edges = [graph.edges[eid] for eid in graph.nodes[graph.root_id].outgoing_edge_ids]
    require(len(root_edges) == 1, "Conjunction split produced extra root edges.")
    for child_id in root_edges[0].child_ids:
        child = graph.nodes[child_id]
        require(child.status == NodeStatus.SOLVED, "A conjunction leaf stayed open.")
        tactics = [graph.edges[eid].tactic.tactic_name for eid in child.outgoing_edge_ids]
        # The elaboration failure and the admission were rejected by Lean.
        require(tactics == ["trivial"], f"Rejected alternatives became edges: {tactics}")
    require(
        all(edge.visit_stats.virtual_loss == 0 for edge in graph.edges.values()),
        "PUCT search left virtual losses.",
    )
    require(
        root_edges[0].visit_stats.N >= 1
        and root_edges[0].visit_stats.N_v == root_edges[0].visit_stats.N,
        "Completed PUCT simulations did not back up valid evidence.",
    )
    _require_graph_equivalence(
        graph, root_edges[0], goldens["conjunction_split"], "conjunction_split"
    )
    require((await client.stat()).n_goals == baseline, "PUCT search leaked states.")
    checks += [
        "puct-multi-leaf-batch",
        "checked-rejection-of-alternatives",
        "puct-graph-equivalence",
    ]

    best_first = ScriptedSearchReasoner(client, IDENTITY_SCRIPT, max_depth=4)
    graph = await best_first.prove("∀ (p : Prop), p → p")
    require(graph.is_solved(), "Best-first search did not prove the identity.")
    root_edge = graph.edges[graph.nodes[graph.root_id].outgoing_edge_ids[0]]
    require(root_edge.status == EdgeStatus.SOLVED, "Identity edge stayed unsolved.")
    _require_graph_equivalence(
        graph, root_edge, goldens["identity_intro"], "identity_intro"
    )
    require((await client.stat()).n_goals == baseline, "Best-first search leaked states.")
    checks.append("best-first-graph-equivalence")

    cutoff = _FirstBatchCutoff(client, CONJUNCTION_SCRIPT, **PUCT_SETTINGS)
    graph = await cutoff.prove("True ∧ True", deadline=0.0)
    require(
        graph.end_reason == SearchEndReason.DEADLINE and not graph.is_solved(),
        "Cooperative cutoff did not end the search with a partial graph.",
    )
    require(
        bool(graph.nodes[graph.root_id].outgoing_edge_ids),
        "Cooperative cutoff discarded the completed root expansion.",
    )
    require(
        (await client.stat()).n_goals == baseline,
        "Theorem scope did not close after a cooperative cutoff.",
    )
    checks.append("cooperative-deadline-scope-closure")

    for _ in range(searches):
        graph = await _puct_reasoner(client).prove("True ∧ True")
        require(graph.is_solved(), "A repeated PUCT search failed.")
        require(
            (await client.stat()).n_goals == baseline,
            "Repeated searches did not return allocations to baseline.",
        )
    checks.append(f"repeated-searches-{searches}")
    return checks


async def verify_shared_device_workers(env: PantographEnv) -> list[str]:
    """Two pool workers with independent clients share one model device."""
    import torch

    from maths_ai.gnn_inference.atp_lean_gnn.rl_live_collection import LiveReasonerPool
    from maths_ai.gnn_inference.atp_lean_gnn.rl_resources import ResourcePlan

    plan = ResourcePlan(
        reported_cpu_count=2,
        affinity_cpu_count=2,
        quota_cpu_count=None,
        usable_cpu_count=2,
        requested_worker_count=2,
        worker_count=2,
        collection_devices=(torch.device("cpu"),),
        update_device=torch.device("cpu"),
        gpu_resources=(),
    )

    async def server_factory(_worker_id):
        return await create_model_sexpr_client(env)

    def reasoner_factory(_worker_id, _device, _model, server):
        return _puct_reasoner(server)

    async def search_factory(reasoner, theorem, _greedy, deadline):
        return await reasoner.prove(theorem, deadline=deadline)

    pool = await LiveReasonerPool.create(
        model=torch.nn.Linear(1, 1),
        plan=plan,
        server_factory=server_factory,
        reasoner_factory=reasoner_factory,
        search_factory=search_factory,
        is_solved=lambda graph: graph.is_solved(),
        is_cooperative_deadline=lambda graph: False,
        startup_timeout_s=env.server_start_timeout_s,
        shutdown_timeout_s=env.worker_shutdown_timeout_s,
    )
    clients = [worker.server for worker in pool.workers]
    async with pool:
        require(
            len({client.session_id for client in clients}) == 2,
            "Pool workers share one client session.",
        )
        require(
            pool.workers[0].model is pool.workers[1].model,
            "Workers on one device did not share the model replica.",
        )
        pool.assert_generation(0)
        result = await pool.collect_round(
            ["True ∧ True"] * 4, timeout_s=120, model_generation=0
        )
        require(result.stats["solved"] == 4, "Pool workers did not prove every theorem.")
        for client in clients:
            require((await client.stat()).n_goals == 0, "A pool worker leaked states.")
    require(
        all(client.phase.value == "CLOSED" for client in clients),
        "A pool client was not closed and reaped.",
    )
    return ["two-workers-one-model-device"]


async def verify_learning_round(client: PantographClient) -> dict:
    """One bounded RL round: PUCT collection, one on-policy step, mining, HTPS.

    A fresh small actor-critic proposes the actions, so this checks the update
    path and its accounting, not model quality.
    """
    import math

    import torch
    from torch.optim import AdamW

    from maths_ai.data_models.proof_components import LeanGoalSeed, seed_replay_spec
    from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
    from maths_ai.gnn_inference.atp_lean_gnn.model_factory import build_actor_critic_model
    from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec
    from maths_ai.gnn_inference.atp_lean_gnn.pln_reward import RewardConfig
    from maths_ai.gnn_inference.atp_lean_gnn.pln_rl_training import (
        train_step_htps_style,
        train_step_onpolicy,
    )
    from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab_from_labels
    from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner
    from maths_ai.gnn_inference.atp_lean_gnn.search_harvest import (
        extract_critic_samples,
        extract_minimal_hypertree,
    )

    from .pantograph_model_sexpr import materialize_replay

    torch.manual_seed(0)
    async with client.state_scope() as scope:
        seed = seed_replay_spec(LeanGoalSeed(expression="True ∧ True"))
        root = (await materialize_replay(scope, seed)).goals[0].goal
    # Free-variable labels cover locals that sampled tactics may introduce.
    node_vocab = build_vocab_from_labels(
        [
            *(node.label for node in model_goal_to_dag(root).nodes),
            *(f"FV{index}" for index in range(8)),
        ]
    )
    model = build_actor_critic_model(
        model_spec=ModelSpec.from_dict(
            {
                "architecture": "graphsage",
                "hidden_dim": 32,
                "dropout": 0.1,
                "encoder": {"num_layers": 2},
                "use_node_type": True,
                "max_args": 1,
            }
        ),
        num_node_labels=len(node_vocab),
        num_tactics=len(LEARNING_TACTICS),
    )
    with torch.no_grad():
        # Favor the two tactics that close the seed so the round has a proof.
        model.actor.base.bias[LEARNING_TACTICS["constructor"]] += 4.0
        model.actor.base.bias[LEARNING_TACTICS["trivial"]] += 4.0
    reasoner = RLHybridReasoner(
        model,
        node_vocab,
        LEARNING_TACTICS,
        executor=PantographExecutor(client),
        top_k_tactics=3,
        use_pln=False,
        **PUCT_SETTINGS,
    )
    baseline = (await client.stat()).n_goals
    result = await reasoner.prove("True ∧ True")
    require((await client.stat()).n_goals == baseline, "Learning search leaked states.")
    require(
        bool(result.edge_actions or result.failure_actions),
        "The learning search produced no on-policy actions.",
    )

    onpolicy = train_step_onpolicy(
        model,
        AdamW(model.parameters(), lr=1e-3),
        [result],
        reasoner.dag_featurize_data,
        reward_cfg=RewardConfig(step_penalty=0.01),
        bc_weight=0.0,
    )
    critic_samples = extract_critic_samples(result.graph, visit_threshold=1)
    imitation_samples = extract_minimal_hypertree(result.graph, result.edge_actions)
    htps = train_step_htps_style(
        model,
        AdamW(model.parameters(), lr=1e-4),
        imitation_samples,
        critic_samples,
        reasoner.dag_featurize_data,
    )
    metrics = {**onpolicy, **htps}
    require(
        all(math.isfinite(float(value)) for value in metrics.values()),
        "A bounded learning round produced a non-finite metric.",
    )
    onpolicy_steps = int(onpolicy["onpolicy_optimizer_step"])
    htps_steps = int(htps["htps_optimizer_step"])
    require(onpolicy_steps == 1, "The on-policy update did not take its single step.")
    require(
        htps_steps == int(bool(imitation_samples or critic_samples)),
        "The HTPS update step count does not match the mined replay.",
    )
    return {
        "solved": result.graph.is_solved(),
        "edge_actions": len(result.edge_actions),
        "failure_actions": len(result.failure_actions),
        "imitation_samples": len(imitation_samples),
        "critic_samples": len(critic_samples),
        "onpolicy_optimizer_steps": onpolicy_steps,
        "htps_optimizer_steps": htps_steps,
        "model_generation": onpolicy_steps + htps_steps,
    }


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _target_identity() -> dict:
    root = Path(__file__).parents[2]

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, timeout=5
        ).strip()

    return {
        "target_branch": TARGET_BRANCH,
        "target_base_commit": TARGET_BASE_COMMIT,
        "client_source_commit": SOURCE_COMMIT,
        "repository_revision": git("rev-parse", "HEAD"),
        "worktree_clean": not git("status", "--porcelain", "--", *SEARCH_SOURCES),
        "search_sources_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SEARCH_SOURCES
        },
    }


async def verify_live_mcts(
    env: PantographEnv, *, proofs: int = 100, searches: int = 20
) -> dict:
    """The client gate, then the search gate, with the target's own identity."""
    report = {
        "target": _target_identity(),
        "configuration": {
            "imports": list(env.imports),
            "request_timeout_s": env.timeout,
            "server_start_timeout_s": env.server_start_timeout_s,
            "worker_shutdown_timeout_s": env.worker_shutdown_timeout_s,
            "source_root_revision": _git_revision(env.source_root),
            "scoped_proofs": proofs,
            "repeated_searches": searches,
            "search": PUCT_SETTINGS,
            "device": "cpu",
        },
        "unavailable_resources": {
            "gpu": "not used; this is a CPU ownership gate, not GPU deployment evidence",
            "checkpoint": "not used; proposals are scripted or from a fresh small model",
            "petta": "not used; searches run with use_pln=False",
            "complete_theorem_certification": False,
        },
    }
    report["client"] = await verify_live_client(env, proofs=proofs)
    client = await create_model_sexpr_client(env)
    try:
        initial = (await client.stat()).n_goals
        checks = await verify_mcts_search(client, searches=searches)
        learning = await verify_learning_round(client)
        final = (await client.stat()).n_goals
        if final != initial:
            raise PantographProtocolError("MCTS search checks leaked states.")
    finally:
        await client.aclose()
    report["search"] = {
        "checks": checks,
        "initial_states": initial,
        "final_states": final,
        "phase": client.phase.value,
        "subprocess_reaped": client.phase.value == "CLOSED",
    }
    report["search"]["checks"] += await verify_shared_device_workers(env)
    report["learning"] = learning
    return report
