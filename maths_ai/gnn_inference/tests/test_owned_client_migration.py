"""Shared caller contracts after the atomic owned-client migration."""

import asyncio
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from maths_ai.data_models.proof_components import TacticCandidate
from maths_ai.gnn_inference.model import GNNPredictor
from maths_ai.gnn_inference.atp_lean_gnn.inference import InferencePipeline
from maths_ai.gnn_inference.tests.model_helpers import structured_goal, pantograph_goal
from maths_ai.hybrid_reasoner.joint_inference import HybridReasoner, PantographExecutor
from maths_ai.hybrid_reasoner.pantograph_protocol import (
    PantographStateError,
    execution_goal_to_translator,
)


def test_production_modules_import_with_upstream_client_forbidden():
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'pantograph':
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
import maths_ai.hybrid_reasoner.joint_inference
import maths_ai.gnn_inference.atp_lean_gnn.rl_training_driver
import maths_ai.gnn_inference.model
import maths_ai.pln_inference.metta.translator.translator_modules.cli
assert 'pantograph' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_prediction_is_synchronous_canonical_and_gradient_free_in_active_loop():
    goal = structured_goal("True")
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    seen = []
    pipeline = object.__new__(InferencePipeline)

    def predict(dag, *, top_k):
        seen.append(
            (torch.is_grad_enabled(), (parameter * 2).requires_grad, dag.state_root_id)
        )
        return SimpleNamespace(top_tactic_predictions=[{"tactic_name": "trivial"}])

    pipeline._predict_from_dag = predict
    predictor = object.__new__(GNNPredictor)
    predictor.pipeline = pipeline

    async def run():
        predictions = predictor.predict_tactics_with_arguments(goal)
        assert predictions == [{"tactic_name": "trivial"}]

    asyncio.run(run())
    assert seen and seen[0][:2] == (False, False)
    assert not hasattr(predictor, "close") and not hasattr(predictor, "pantograph_env")


def test_overlapping_prove_is_rejected_before_changing_search_state():
    reasoner = object.__new__(HybridReasoner)
    reasoner._proving = True

    async def run():
        with pytest.raises(PantographStateError):
            await reasoner.prove("True")

    asyncio.run(run())
    assert reasoner._proving is True


def test_translator_projection_uses_local_names_for_raw_free_variables():
    from maths_ai.pln_inference.metta.translator.translator_modules.extractor import (
        essentialize_subgoal,
    )
    from maths_ai.hybrid_reasoner.pantograph_protocol import decode_execution_goal
    import json

    fixture = Path(__file__).parent / "fixtures/pantograph/future_contract.json"
    wire = json.loads(fixture.read_text())["successes"]["identity_intro"]["goals"][0]
    projection = execution_goal_to_translator(decode_execution_goal(wire))
    script, _cleanup, _hyps, formula = essentialize_subgoal(projection, mode="dynamic")
    assert "_uniq" not in formula and "p" in formula
    assert "_uniq" not in script


def test_translator_validates_environment_before_touching_output(tmp_path):
    from maths_ai.pln_inference.metta.translator.translator_modules.cli import run_demo

    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("preserve")

    class InvalidEnvironment:
        def verify(self):
            raise ValueError("invalid explicit paths")

    async def run():
        with pytest.raises(ValueError, match="invalid explicit paths"):
            await run_demo(
                "unused.json",
                str(tmp_path),
                env=InvalidEnvironment(),
                axioms_path="axioms.metta",
            )

    asyncio.run(run())
    assert sentinel.read_text() == "preserve"
