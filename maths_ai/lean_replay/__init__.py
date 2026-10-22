"""Lean goal reconstruction shared by dataset seeds and canonical replay.

Lightweight on purpose: no Torch, GNN or Pantograph client imports, so data
models and loaders can depend on it without import cycles.
"""

from .errors import ReplayFailure, ReplayInputError
from .text import Hypothesis, ProofState, parse_state, parse_states

__all__ = [
    "Hypothesis",
    "ProofState",
    "ReplayFailure",
    "ReplayInputError",
    "parse_state",
    "parse_states",
]
