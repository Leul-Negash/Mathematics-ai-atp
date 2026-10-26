"""Validated source-aware model and live retrieval for deterministic search."""

import json
from pathlib import Path
import torch
from .model import GNNPredictor
from .atp_lean_gnn.checkpointing import build_model_from_checkpoint
from .atp_lean_gnn.training import load_prepared_metadata
from .atp_lean_gnn.retrieval_artifacts import RetrievalConfig, LiveLemmaProvider
from maths_ai.data_models.proof_components import TacticCandidate


class GNNModelEngine:
    def __init__(
        self,
        config_path,
        tactic_predictor_model_path,
        retrieval_config_path,
        *,
        device="cpu",
    ):
        self.device = torch.device(device)
        config = json.loads(Path(config_path).read_text())
        metadata = load_prepared_metadata(Path(config["prepared_root"]), splits=())
        checkpoint = torch.load(
            tactic_predictor_model_path, map_location=self.device, weights_only=False
        )
        model, _, spec = build_model_from_checkpoint(
            checkpoint,
            node_vocab=metadata.node_vocab,
            tactic_vocab=metadata.tactic_vocab,
            graph_representation=metadata.graph_representation,
        )
        self.provider = LiveLemmaProvider(
            RetrievalConfig(**json.loads(Path(retrieval_config_path).read_text()))
        )
        if spec.retrieval_binding != self.provider.library.fingerprint:
            raise ValueError("Model and retrieval artifact differ.")
        self.gnn_inference = GNNPredictor(
            model.to(self.device).eval(),
            self.provider,
            metadata.node_vocab,
            metadata.tactic_vocab,
            self.device,
        )

    async def prepare_goal(self, goal):
        await self.gnn_inference.prepare_goal(goal)

    def inference(self, goal, top_k=3):
        return [
            TacticCandidate(
                tactic_name=x["tactic_name"],
                arguments=x["selected_arguments"],
                probability=x["probability"],
            )
            for x in self.gnn_inference.predict_tactics_with_arguments(goal, top_k)
        ]

    async def aclose(self):
        await self.provider.aclose()
