from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.optim import AdamW

from maths_ai.data_models.proof_components import Goal, LeanGoalSeed, STV
from maths_ai.hybrid_reasoner.hypergraph import TacticOutcome
from maths_ai.pln_inference.model import PLNResult

from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import checkpoint_payload
from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import MODEL_SEXPR_GRAPH_SPEC
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_vocab
from maths_ai.gnn_inference.atp_lean_gnn.rl_reasoner import RLHybridReasoner
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver import (
    RLTrainingConfig,
    TheoremItem,
    TheoremPool,
    _create_live_reasoner_pool,
    bc_weight_at_round,
    build_theorem_pool,
    collect_round,
    evaluate_proof_rate,
    run_rl_training,
    save_checkpoint,
)
from maths_ai.gnn_inference.atp_lean_gnn.rl_live_collection import LiveReasonerPool
from maths_ai.gnn_inference.atp_lean_gnn import rl_training_driver
from maths_ai.gnn_inference.atp_lean_gnn.rl_resources import ResourcePlan
from pantograph_async import (
    PantographConfigurationError,
    PantographProtocolError,
    PantographRequestTimeout,
    PantographStateError,
    PantographTransportError,
)
from maths_ai.gnn_inference.tests.model_helpers import (
    ScopedFakeClient,
    actor_critic,
    pantograph_goal,
    structured_goal,
)


# ---------------------------------------------------------------------------
# Fakes (mirroring test_rl_reasoner.py — no Lean/petta)
# ---------------------------------------------------------------------------


class _FakeGoalState:
    def __init__(self, goals=None):
        self.goals = list(goals or [])


class _FakeServer(ScopedFakeClient):
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True
        self.is_usable = False

    async def _start_goal(self, expression):
        return _FakeGoalState()

    async def _apply_tactic(self, state, tactic):
        if tactic.startswith("intro "):
            return _FakeGoalState([pantograph_goal(structured_goal(GOAL_EXPR, HYPS))])
        # Any other tactic closes the replayed goal.
        return _FakeGoalState()


class _QEDExecutor:
    def __init__(self):
        self.server = _FakeServer()

    async def apply(self, server, state, tactic):
        return TacticOutcome(success=True, subgoals=[])


class _RejectExecutor:
    def __init__(self):
        self.server = _FakeServer()

    async def apply(self, server, state, tactic):
        return TacticOutcome(success=False, subgoals=[], error="rejected")


class _StubPLN:
    async def evaluate_async(self, expression, hypotheses=None, **kwargs):
        return PLNResult(stv=STV(strength=0.1, confidence=1.0), status="ok", is_fallback=False)


class _RaisingReasoner:
    """prove() raises on a chosen call index — tests per-theorem fault isolation."""

    def __init__(self, inner, raise_on: set[int]):
        self._inner = inner
        self._raise_on = raise_on
        self._calls = 0
        self.model = inner.model
        self.dag_featurize_data = inner.dag_featurize_data

    async def prove(self, goal, *, hypotheses=None, greedy=False, deadline=None):
        idx = self._calls
        self._calls += 1
        if idx in self._raise_on:
            raise RuntimeError("simulated Lean transport failure")
        return await self._inner.prove(goal, hypotheses=hypotheses, greedy=greedy, deadline=deadline)


TACTIC_VOCAB = {"trivial": 0, "intro": 1, "exact": 2, "<UNK_TACTIC>": 3}
GOAL_EXPR = "p → p"
HYPS = ["p : Prop"]


def _build_node_vocab():
    return build_vocab([model_goal_to_dag(structured_goal(GOAL_EXPR, HYPS))])


def _make_model(node_vocab):
    return actor_critic(len(node_vocab), len(TACTIC_VOCAB))


def _make_reasoner(model, node_vocab, executor, *, top_k=3):
    reasoner = RLHybridReasoner(
        model,
        node_vocab,
        TACTIC_VOCAB,
        executor=executor,
        top_k_tactics=top_k,
        max_depth=3,
        max_nodes=20,
    )
    reasoner.petta_chainer = _StubPLN()
    return reasoner


def _items(n: int) -> list[TheoremItem]:
    return [
        TheoremItem(
            goal=LeanGoalSeed(expression=GOAL_EXPR, hypotheses=HYPS),
            tactic_label="intro",
            size=10 + i,
        )
        for i in range(n)
    ]


def _write_config(tmp: Path, **overrides) -> RLTrainingConfig:
    """Config pointing at a synthetic prepared_root + warm-start checkpoint in tmp."""
    node_vocab = _build_node_vocab()
    vocab_dir = tmp / "prepared" / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    with open(vocab_dir / "node_vocab.json", "w") as f:
        json.dump(node_vocab, f)
    with open(vocab_dir / "tactic_vocab.json", "w") as f:
        json.dump(TACTIC_VOCAB, f)
    metadata_dir = tmp / "prepared" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "graph_representation.json", "w") as f:
        json.dump(MODEL_SEXPR_GRAPH_SPEC.to_dict(), f)
    manifests_dir = tmp / "prepared" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        pyg_dir = tmp / "prepared" / "pyg" / split
        pyg_dir.mkdir(parents=True, exist_ok=True)
        with open(manifests_dir / f"{split}.json", "w") as f:
            json.dump({"artifact_paths": {"pyg_dir": f"pyg/{split}"}}, f)

    torch.manual_seed(0)
    model = _make_model(node_vocab)
    ckpt = tmp / "warmstart.pt"
    torch.save(
        checkpoint_payload(
            model_kind="actor_critic_with_args",
            model_spec=model.model_spec,
            node_vocab=node_vocab,
            tactic_vocab=TACTIC_VOCAB,
            model=model,
            graph_representation=MODEL_SEXPR_GRAPH_SPEC,
        ),
        ckpt,
    )

    defaults = dict(
        warmstart_checkpoint=ckpt,
        prepared_root=tmp / "prepared",
        run_root=tmp / "runs",
        device="cpu",
        num_rounds=2,
        theorems_per_round=2,
        theorem_timeout_s=30.0,
        checkpoint_every=1,
        eval_every=0,
        eval_pool_size=0,
        bc_anneal_start=0.5,
        bc_anneal_end=0.0,
        bc_anneal_rounds=10,
        top_k_tactics=2,
        max_depth=3,
        max_nodes=20,
    )
    defaults.update(overrides)
    return RLTrainingConfig(**defaults)


def _pool(n_items: int = 6, eval_size: int = 0) -> TheoremPool:
    return TheoremPool(_items(n_items), eval_pool_size=eval_size, curriculum_size=4, seed=0)


def _qed_factory(model, node_vocab, tactic_vocab, cfg):
    return _make_reasoner(model, node_vocab, _QEDExecutor(), top_k=cfg.top_k_tactics)


def _reject_factory(model, node_vocab, tactic_vocab, cfg):
    return _make_reasoner(model, node_vocab, _RejectExecutor(), top_k=cfg.top_k_tactics)


class ConfigTests(unittest.TestCase):
    def test_config_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp)
            path = tmp / "cfg.json"
            with open(path, "w") as f:
                json.dump(cfg.to_dict(), f)
            loaded = RLTrainingConfig.from_json(path)
            self.assertEqual(loaded.to_dict(), cfg.to_dict())
            self.assertIsInstance(loaded.warmstart_checkpoint, Path)

    def test_live_workers_receive_complete_puct_configuration(self):
        class _Environment:
            def describe(self):
                return "test MCTS environment"

        class _RecordingReasoner:
            instances = []

            def __init__(self, **kwargs):
                self.model = kwargs["model"]
                self.kwargs = kwargs
                self._pending = {}
                self._result = None
                self.__class__.instances.append(self)

        async def create_server(_environment):
            return _FakeServer()

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(
                tmp,
                selection_policy="puct",
                num_simulations=12,
                sim_batch_size=3,
                puct_c=1.75,
                top_k_tactics=5,
                max_depth=7,
                max_nodes=33,
                use_pln=False,
            )
            node_vocab = _build_node_vocab()
            model = _make_model(node_vocab)
            plan = ResourcePlan(
                reported_cpu_count=8,
                affinity_cpu_count=8,
                quota_cpu_count=None,
                usable_cpu_count=8,
                requested_worker_count=2,
                worker_count=2,
                collection_devices=(torch.device("cpu"),),
                update_device=torch.device("cpu"),
                gpu_resources=(),
            )
            with (
                patch(
                    "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver."
                    "create_model_sexpr_client",
                    new=create_server,
                ),
                patch(
                    "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver."
                    "RLHybridReasoner",
                    new=_RecordingReasoner,
                ),
                patch(
                    "maths_ai.hybrid_reasoner.joint_inference.PantographExecutor",
                    new=lambda server: server,
                ),
            ):
                live_pool = asyncio.run(
                    _create_live_reasoner_pool(
                        model=model,
                        node_vocab=node_vocab,
                        tactic_vocab=TACTIC_VOCAB,
                        cfg=cfg,
                        env=_Environment(),
                        plan=plan,
                    )
                )
                try:
                    # A replacement worker is built by the same factory with the
                    # same model replica, so it keeps every search setting.
                    replacement = asyncio.run(
                        live_pool._replacement_worker(live_pool.workers[0])
                    )
                    self.assertIs(replacement.model, live_pool.workers[1].model)
                    live_pool.assert_generation(0)
                    _RecordingReasoner.instances.remove(replacement.reasoner)
                    settings = (
                        "selection_policy",
                        "num_simulations",
                        "sim_batch_size",
                        "puct_c",
                        "top_k_tactics",
                        "max_depth",
                        "max_nodes",
                        "use_pln",
                    )
                    self.assertEqual(
                        {key: replacement.reasoner.kwargs[key] for key in settings},
                        {
                            key: _RecordingReasoner.instances[0].kwargs[key]
                            for key in settings
                        },
                    )
                    self.assertEqual(len(_RecordingReasoner.instances), 2)
                    self.assertEqual(
                        len({id(reasoner) for reasoner in _RecordingReasoner.instances}),
                        2,
                    )
                    self.assertEqual(
                        len({id(reasoner._pending) for reasoner in _RecordingReasoner.instances}),
                        2,
                    )
                    for reasoner in _RecordingReasoner.instances:
                        self.assertEqual(reasoner.kwargs["selection_policy"], "puct")
                        self.assertEqual(reasoner.kwargs["num_simulations"], 12)
                        self.assertEqual(reasoner.kwargs["sim_batch_size"], 3)
                        self.assertEqual(reasoner.kwargs["puct_c"], 1.75)
                        self.assertEqual(reasoner.kwargs["top_k_tactics"], 5)
                        self.assertEqual(reasoner.kwargs["max_depth"], 7)
                        self.assertEqual(reasoner.kwargs["max_nodes"], 33)
                        self.assertFalse(reasoner.kwargs["use_pln"])
                finally:
                    asyncio.run(live_pool.close())

    def test_config_missing_required_field_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            with open(path, "w") as f:
                json.dump({"prepared_root": "x"}, f)  # no warmstart_checkpoint
            with self.assertRaises(TypeError):
                RLTrainingConfig.from_json(path)

    def test_config_rejects_budget_under_legacy_policy(self):
        # __post_init__ runs resolve_search_params: an explicit simulation
        # budget under selection_policy="legacy" fails at config construction,
        # before any Lean server spins up.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _write_config(Path(tmp), num_simulations=8)

    def test_config_puct_requires_num_simulations(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                _write_config(Path(tmp), selection_policy="puct")
            cfg = _write_config(Path(tmp), selection_policy="puct", num_simulations=8)
            self.assertEqual(cfg.num_simulations, 8)


class PoolTests(unittest.TestCase):
    @patch("maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver.iter_dataset_rows")
    def test_dataset_name_is_forwarded_to_stream(self, mock_iter_dataset_rows):
        mock_iter_dataset_rows.return_value = iter(
            [
                DatasetRow(
                    state="p : Prop\n|- p → p",
                    theorem="demo",
                    tactic="intro",
                    split="train",
                    row_index=0,
                    dataset_name="jajostrains/Mathlib-Normalized-Sexpr",
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(
                Path(tmp),
                dataset_name="jajostrains/Mathlib-Normalized-Sexpr",
                max_pool_size=1,
                eval_pool_size=0,
            )
            pool = build_theorem_pool(cfg)
        self.assertEqual(len(pool.train_items), 1)
        mock_iter_dataset_rows.assert_called_once_with(
            dataset_name="jajostrains/Mathlib-Normalized-Sexpr",
            split="train",
            sample_limit=2,
        )

    def test_file_mode_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            theorem_file = tmp / "theorems.jsonl"
            rows = [
                {"goal": "p → p", "hypotheses": ["p : Prop"], "tactic": "intro"},
                {"goal": "q ∨ p", "hypotheses": ["p : Prop", "q : Prop", "h : p ∨ q"]},
                {"goal": "x" * 500, "hypotheses": []},  # over max_state_chars → dropped
            ]
            with open(theorem_file, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            cfg = _write_config(
                tmp, data_source="file", theorem_file=theorem_file,
                max_state_chars=400, eval_pool_size=0,
            )
            pool = build_theorem_pool(cfg)
            total = len(pool.train_items) + len(pool.eval_items)
            self.assertEqual(total, 2)  # oversized row dropped
            self.assertEqual(pool.train_items[0].goal.expression, "p → p")  # size-sorted
            self.assertEqual(pool.train_items[0].tactic_label, "intro")

    def test_universe_placeholders_are_not_term_metavariables(self):
        from maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver import _has_metavariable

        goal = LeanGoalSeed(
            expression="x = x",
            hypotheses=["alpha : Type ?u.319125", "x : alpha"],
        )
        self.assertFalse(_has_metavariable(goal))

    def test_curriculum_window_and_growth(self):
        pool = _pool(n_items=10)
        pool.curriculum_size = 4
        batch = pool.sample_batch(3)
        self.assertEqual(len(batch), 3)
        window = pool.train_items[:4]
        for item in batch:
            self.assertIn(item, window)
        pool.grow(2.0)
        self.assertEqual(pool.curriculum_size, 8)
        pool.grow(10.0)  # capped at the pool size
        self.assertEqual(pool.curriculum_size, len(pool.train_items))


class BCAnnealTests(unittest.TestCase):
    def test_anneal_endpoints_and_monotonicity(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _write_config(Path(tmp), bc_anneal_start=0.5, bc_anneal_end=0.1, bc_anneal_rounds=10)
        self.assertAlmostEqual(bc_weight_at_round(0, cfg), 0.5)
        self.assertAlmostEqual(bc_weight_at_round(10, cfg), 0.1)
        self.assertAlmostEqual(bc_weight_at_round(100, cfg), 0.1)
        weights = [bc_weight_at_round(i, cfg) for i in range(11)]
        self.assertTrue(all(a >= b for a, b in zip(weights, weights[1:])))


class RoundLoopTests(unittest.TestCase):
    def test_happy_path_two_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=2)
            torch.manual_seed(0)
            metrics = asyncio.run(
                run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool())
            )
            self.assertEqual(metrics["round"], 1)
            run_dirs = list((tmp / "runs").iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertTrue((run_dir / "last.pt").exists())
            self.assertTrue((run_dir / "config.json").exists())
            rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertGreater(rows[0]["num_transitions"] + rows[0]["num_failures"], 0)

    def test_params_change_after_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1)
            node_vocab = _build_node_vocab()
            torch.manual_seed(0)
            before = _make_model(node_vocab)
            before.load_state_dict(
                torch.load(cfg.warmstart_checkpoint, weights_only=False)["model_state_dict"]
            )
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            after_sd = torch.load(run_dir / "last.pt", weights_only=False)["model_state_dict"]
            changed = any(
                not torch.equal(before.state_dict()[k], after_sd[k]) for k in after_sd
            )
            self.assertTrue(changed, "one training round did not update any parameters")

    def test_per_theorem_fault_isolation(self):
        node_vocab = _build_node_vocab()
        torch.manual_seed(0)
        model = _make_model(node_vocab)
        inner = _make_reasoner(model, node_vocab, _QEDExecutor(), top_k=2)
        reasoner = _RaisingReasoner(inner, raise_on={1})  # second theorem dies
        results, stats = asyncio.run(
            collect_round(reasoner, _items(3), timeout_s=10.0)
        )
        self.assertEqual(stats["attempted"], 3.0)
        self.assertEqual(stats["collected"], 2.0)
        self.assertEqual(stats["searches_failed"], 1.0)
        self.assertEqual(len(results), 2)

    def test_resume_continues_round_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=2)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())

            cfg.num_rounds = 3
            torch.manual_seed(0)
            asyncio.run(
                run_rl_training(
                    cfg, resume_run_dir=run_dir, reasoner_factory=_qed_factory, pool=_pool()
                )
            )
            rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
            train_rows = [r for r in rows if "num_transitions" in r]
            self.assertEqual([r["round"] for r in train_rows], [0, 1, 2])
            state = torch.load(run_dir / "last.pt", weights_only=False)
            self.assertEqual(state["round"], 2)


class EvalTests(unittest.TestCase):
    def test_greedy_eval_deterministic(self):
        node_vocab = _build_node_vocab()
        torch.manual_seed(0)
        model = _make_model(node_vocab)
        reasoner = _make_reasoner(model, node_vocab, _QEDExecutor(), top_k=2)
        items = _items(4)
        s1 = asyncio.run(evaluate_proof_rate(reasoner, items, timeout_s=10.0))
        s2 = asyncio.run(evaluate_proof_rate(reasoner, items, timeout_s=10.0))
        self.assertEqual(s1, s2)
        self.assertEqual(s1["attempted"], 4.0)

    def test_best_checkpoint_written_on_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1, eval_every=1)
            pool = TheoremPool(_items(20), eval_pool_size=2, curriculum_size=4, seed=0)
            self.assertGreater(len(pool.eval_items), 0)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=pool))
            run_dir = next((tmp / "runs").iterdir())
            # QED executor solves everything ⇒ proof rate 1.0 > initial -1 ⇒ best.pt written.
            self.assertTrue((run_dir / "best.pt").exists())

    def test_reject_run_writes_no_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1, eval_every=1)
            pool = TheoremPool(_items(20), eval_pool_size=2, curriculum_size=4, seed=0)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_reject_factory, pool=pool))
            run_dir = next((tmp / "runs").iterdir())
            # Reject executor proves nothing ⇒ proof rate 0.0 > -1 initial: best.pt IS
            # written once (first eval), but records rate 0.
            state = torch.load(run_dir / "best.pt", weights_only=False)
            self.assertEqual(state["best_proof_rate"], 0.0)


class HTPSDriverTests(unittest.TestCase):
    """Phase 2–3 driver wiring: queues, decoupled step, checkpoint round-trip."""

    def _metric_rows(self, run_dir: Path) -> list[dict]:
        rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
        return [r for r in rows if "num_transitions" in r]

    def test_default_off_runs_no_decoupled_step(self):
        # htps_steps_per_round defaults to 0: mining still fills the queues
        # (pure bookkeeping, no gradient), but no decoupled loss is ever logged.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1)
            self.assertEqual(cfg.htps_steps_per_round, 0)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            for row in self._metric_rows(run_dir):
                self.assertNotIn("tactic_imitation_loss", row)
                self.assertNotIn("critic_soft_loss", row)
                self.assertEqual(row["htps_optimizer_steps"], 0.0)
                self.assertEqual(row["parameter_updates_this_round"], 1.0)
                self.assertEqual(row["model_generation"], 1.0)

    def test_enabled_step_logs_losses_and_queues_grow(self):
        # QED executor ⇒ every root is SOLVED: the critic queue gets hard 1.0
        # labels and the tactic queue gets the minimal hypertree's QED edges.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1, htps_steps_per_round=2, htps_batch_size=4)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            rows = self._metric_rows(run_dir)
            self.assertGreater(rows[-1]["tactic_queue_len"], 0)
            self.assertGreater(rows[-1]["critic_queue_len"], 0)
            self.assertGreater(rows[-1]["imitation_samples_mined"], 0)
            self.assertIn("tactic_imitation_loss", rows[-1])
            self.assertIn("critic_soft_loss", rows[-1])
            self.assertEqual(rows[-1]["onpolicy_optimizer_step"], 1.0)
            self.assertEqual(rows[-1]["htps_optimizer_steps"], 2.0)
            self.assertEqual(rows[-1]["parameter_updates_this_round"], 3.0)
            self.assertEqual(rows[-1]["model_generation"], 3.0)
            self.assertEqual(rows[-1]["replica_generation"], 3.0)

    def test_checkpoint_roundtrips_optimizer_htps_and_queues(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1, htps_steps_per_round=1, htps_batch_size=4)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            state = torch.load(run_dir / "last.pt", weights_only=False)
            # The decoupled step ran ⇒ its optimizer carries Adam moments.
            self.assertIn("optimizer_htps_state_dict", state)
            self.assertGreater(len(state["optimizer_htps_state_dict"]["state"]), 0)
            # Queues serialize canonical goals and graph fingerprints as plain dicts.
            self.assertGreater(len(state["tactic_queue"]), 0)
            tactic_row = state["tactic_queue"][0]
            self.assertIsInstance(tactic_row["goal"], dict)
            self.assertTrue(tactic_row["graph_fingerprint"])
            self.assertIsInstance(tactic_row["tactic_id"], int)
            self.assertIsInstance(tactic_row["arg_indices"], list)
            self.assertGreater(len(state["critic_queue"]), 0)
            critic_row = state["critic_queue"][0]
            self.assertIsInstance(critic_row["goal"], dict)
            self.assertTrue(critic_row["graph_fingerprint"])
            self.assertEqual(critic_row["target"], 1.0)
            self.assertEqual(critic_row["source"], "lean_status")

            # Resume restores the queues: with num_rounds == start_round the loop
            # body never runs, so the final checkpoint's queues are exactly the
            # restored ones.
            cfg.num_rounds = int(state["round"]) + 1
            torch.manual_seed(0)
            asyncio.run(run_rl_training(
                cfg, resume_run_dir=run_dir, reasoner_factory=_qed_factory, pool=_pool()
            ))
            state2 = torch.load(run_dir / "last.pt", weights_only=False)
            self.assertEqual(state2["tactic_queue"], state["tactic_queue"])
            self.assertEqual(state2["critic_queue"], state["critic_queue"])

    def test_htps_only_update_advances_model_generation(self):
        no_onpolicy_step = {
            "num_transitions": 0.0,
            "num_failures": 0.0,
            "unknown_label_count": 0.0,
            "structural_unknown_label_count": 0.0,
            "semantic_unknown_label_count": 0.0,
            "total_label_count": 0.0,
            "onpolicy_optimizer_step": 0.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(
                tmp,
                num_rounds=1,
                htps_steps_per_round=1,
                htps_batch_size=4,
            )
            with patch(
                "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver."
                "train_step_onpolicy",
                return_value=no_onpolicy_step,
            ):
                asyncio.run(
                    run_rl_training(
                        cfg,
                        reasoner_factory=_qed_factory,
                        pool=_pool(),
                    )
                )
            run_dir = next((tmp / "runs").iterdir())
            row = self._metric_rows(run_dir)[-1]
            self.assertEqual(row["onpolicy_optimizer_step"], 0.0)
            self.assertEqual(row["htps_optimizer_steps"], 1.0)
            self.assertEqual(row["parameter_updates_this_round"], 1.0)
            self.assertEqual(row["model_generation"], 1.0)

    def test_no_update_round_keeps_model_generation_unchanged(self):
        no_onpolicy_step = {
            "num_transitions": 0.0,
            "num_failures": 0.0,
            "unknown_label_count": 0.0,
            "structural_unknown_label_count": 0.0,
            "semantic_unknown_label_count": 0.0,
            "total_label_count": 0.0,
            "onpolicy_optimizer_step": 0.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(
                tmp,
                num_rounds=1,
                htps_steps_per_round=0,
                max_dead_rounds=2,
            )
            with patch(
                "maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver."
                "train_step_onpolicy",
                return_value=no_onpolicy_step,
            ):
                asyncio.run(
                    run_rl_training(
                        cfg,
                        reasoner_factory=_qed_factory,
                        pool=_pool(),
                    )
                )
            run_dir = next((tmp / "runs").iterdir())
            row = self._metric_rows(run_dir)[-1]
            self.assertEqual(row["parameter_updates_this_round"], 0.0)
            self.assertEqual(row["model_generation"], 0.0)
            self.assertEqual(row["replica_generation"], 0.0)

    def test_pre_validity_checkpoint_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, num_rounds=1)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_qed_factory, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            state = torch.load(run_dir / "last.pt", weights_only=False)
            for key in (
                "rl_target_schema_version",
                "optimizer_htps_state_dict",
                "tactic_queue",
                "critic_queue",
            ):
                state.pop(key, None)  # strip back to the pre-HTPS format
            torch.save(state, run_dir / "last.pt")

            cfg.num_rounds = 2
            torch.manual_seed(0)
            with self.assertRaisesRegex(ValueError, "canonical model-S-expression"):
                asyncio.run(run_rl_training(
                    cfg, resume_run_dir=run_dir, reasoner_factory=_qed_factory, pool=_pool()
                ))


class PLNKillSwitchDriverTests(unittest.TestCase):
    """use_pln=False: config roundtrip and threading to the reasoner factory."""

    def test_use_pln_false_survives_config_roundtrip(self):
        # use_pln=False must survive to_dict() → from_json() intact.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, use_pln=False)
            self.assertFalse(cfg.use_pln)
            path = tmp / "cfg_pln.json"
            with open(path, "w") as f:
                json.dump(cfg.to_dict(), f)
            loaded = RLTrainingConfig.from_json(path)
            self.assertFalse(loaded.use_pln)

    def test_use_pln_false_reaches_reasoner_factory(self):
        # When use_pln=False is set in config, the value must reach the factory.
        # Inject a factory that records and asserts the flag, returning a mock
        # reasoner so no live Lean server is needed.
        received: list[bool] = []

        def _recording_factory(model, node_vocab, tactic_vocab, cfg):
            received.append(cfg.use_pln)
            return _make_reasoner(model, node_vocab, _QEDExecutor(), top_k=cfg.top_k_tactics)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(tmp, use_pln=False, num_rounds=1)
            torch.manual_seed(0)
            asyncio.run(run_rl_training(cfg, reasoner_factory=_recording_factory, pool=_pool()))
        self.assertEqual(len(received), 1)
        self.assertFalse(received[0])


class _ErrorReasoner:
    """prove() raises a chosen error on its first call."""

    def __init__(self, inner, error: BaseException, *, retire_client: bool = False):
        self._inner = inner
        self._error = error
        self._retire_client = retire_client
        self.model = inner.model
        self.server = inner.server
        self.calls = 0

    async def prove(self, goal, *, hypotheses=None, greedy=False, deadline=None):
        self.calls += 1
        if self.calls == 1:
            if self._retire_client:
                await self.server.aclose()
            raise self._error
        return await self._inner.prove(
            goal, hypotheses=hypotheses, greedy=greedy, deadline=deadline
        )


class SingleReasonerFailureTests(unittest.TestCase):
    """The single-reasoner path cannot replace its client, so session failures end it."""

    def _reasoner(self, error, **kwargs):
        node_vocab = _build_node_vocab()
        torch.manual_seed(0)
        inner = _make_reasoner(_make_model(node_vocab), node_vocab, _QEDExecutor(), top_k=2)
        return _ErrorReasoner(inner, error, **kwargs)

    def test_session_failures_propagate_instead_of_counting_as_failed_searches(self):
        for error in (
            PantographTransportError("process exited"),
            PantographProtocolError("bad contract"),
            PantographStateError("stale state"),
            PantographConfigurationError("bad executable"),
        ):
            reasoner = self._reasoner(error)
            with self.assertRaises(type(error)):
                asyncio.run(collect_round(reasoner, _items(2), timeout_s=10.0))
            self.assertEqual(reasoner.calls, 1)

    def test_timeout_that_retired_the_only_client_ends_collection(self):
        reasoner = self._reasoner(
            PantographRequestTimeout("deadline exceeded"), retire_client=True
        )
        with self.assertRaises(PantographTransportError) as error:
            asyncio.run(collect_round(reasoner, _items(2), timeout_s=10.0))
        self.assertIn("reasoner pool", str(error.exception))
        self.assertEqual(reasoner.calls, 1)


class LivePoolDriverTests(unittest.TestCase):
    """The round loop over a real ``LiveReasonerPool`` of scoped fake clients.

    Collection finishes before any parameter update; at most one on-policy step
    and the configured HTPS steps follow; the generation advances by the actual
    optimizer-step count; replicas synchronize once to the final generation
    before the next collection.
    """

    def _run(self, **overrides):
        events: list[tuple] = []
        clients: list[_FakeServer] = []
        original_collect = LiveReasonerPool.collect_round
        original_sync = LiveReasonerPool.synchronize
        original_onpolicy = rl_training_driver.train_step_onpolicy
        original_htps = rl_training_driver.train_step_htps_style

        class _Environment:
            def verify(self):
                return None

            def describe(self):
                return "test environment"

        async def create_client(_environment):
            client = _FakeServer()
            clients.append(client)
            return client

        async def collect(pool, batch, **kwargs):
            events.append(("collect", pool.model_generation))
            return await original_collect(pool, batch, **kwargs)

        def synchronize(pool, model, generation):
            events.append(("sync", generation))
            return original_sync(pool, model, generation)

        def onpolicy(*args, **kwargs):
            metrics = original_onpolicy(*args, **kwargs)
            events.append(("onpolicy", metrics["onpolicy_optimizer_step"]))
            return metrics

        def htps(*args, **kwargs):
            metrics = original_htps(*args, **kwargs)
            events.append(("htps", metrics["htps_optimizer_step"]))
            return metrics

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = _write_config(
                tmp,
                collection_workers=2,
                cpu_reserve=0,
                resource_check="warn",
                **overrides,
            )
            torch.manual_seed(0)
            with (
                patch.object(rl_training_driver, "pantograph_env", lambda _cfg: _Environment()),
                patch.object(rl_training_driver, "create_model_sexpr_client", create_client),
                patch.object(LiveReasonerPool, "collect_round", collect),
                patch.object(LiveReasonerPool, "synchronize", synchronize),
                patch.object(rl_training_driver, "train_step_onpolicy", onpolicy),
                patch.object(rl_training_driver, "train_step_htps_style", htps),
            ):
                asyncio.run(run_rl_training(cfg, pool=_pool()))
            run_dir = next((tmp / "runs").iterdir())
            rows = [
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            ]
        return events, rows, clients

    def _assert_round_order(self, events, rows, htps_steps):
        generation = 0
        rounds = [index for index, event in enumerate(events) if event[0] == "collect"]
        self.assertEqual(len(rounds), len(rows))
        for start, end, row in zip(rounds, [*rounds[1:], len(events)], rows):
            phase = events[start:end]
            self.assertEqual(phase[0], ("collect", generation))
            onpolicy = [event for event in phase if event[0] == "onpolicy"]
            htps = [event for event in phase if event[0] == "htps"]
            syncs = [event for event in phase if event[0] == "sync"]
            self.assertLessEqual(len(onpolicy), 1)
            self.assertEqual(len(htps), htps_steps)
            steps = int(sum(event[1] for event in [*onpolicy, *htps]))
            generation += steps
            # Updates follow collection; one synchronization ends the round.
            self.assertEqual(
                [event[0] for event in phase],
                ["collect", *(["onpolicy"] * len(onpolicy)), *(["htps"] * len(htps)),
                 *(["sync"] if steps else [])],
            )
            self.assertEqual(syncs, [("sync", generation)] if steps else [])
            self.assertEqual(row["parameter_updates_this_round"], float(steps))
            self.assertEqual(row["model_generation"], float(generation))
            self.assertEqual(row["replica_generation"], float(generation))
        return generation

    def test_htps_disabled_round_order_generation_and_replica_sync(self):
        events, rows, clients = self._run(num_rounds=2)
        generation = self._assert_round_order(events, rows, htps_steps=0)
        self.assertEqual(generation, 2)
        for row in rows:
            self.assertEqual(row["onpolicy_optimizer_step"], 1.0)
            self.assertEqual(row["htps_optimizer_steps"], 0.0)
            self.assertGreater(row["imitation_samples_mined"], 0.0)
            self.assertNotIn("tactic_imitation_loss", row)
        self.assertEqual(len(clients), 2)
        self.assertTrue(all(client.closed for client in clients))

    def test_htps_enabled_round_order_generation_and_replica_sync(self):
        events, rows, clients = self._run(
            num_rounds=2, htps_steps_per_round=2, htps_batch_size=4
        )
        generation = self._assert_round_order(events, rows, htps_steps=2)
        self.assertEqual(generation, 6)
        for row in rows:
            self.assertEqual(row["htps_optimizer_steps"], 2.0)
            self.assertGreater(row["tactic_queue_len"], 0.0)
            self.assertGreater(row["critic_queue_len"], 0.0)
            self.assertIn("tactic_imitation_loss", row)
        self.assertTrue(all(client.closed for client in clients))


if __name__ == "__main__":
    unittest.main()
