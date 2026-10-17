"""Explicit immutable launch settings shared by every owned-client caller."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PantographEnv:
    source_root: Path
    pantograph_repl: Path
    imports: tuple[str, ...] = ("Init",)
    options: Mapping[str, Any] = field(default_factory=dict)
    timeout: float = 120
    server_start_timeout_s: float = 120.0
    worker_shutdown_timeout_s: float = 10.0
    response_line_limit: int = 8 * 1024 * 1024
    stderr_tail_limit: int = 64 * 1024
    build_manifest: Path | None = None

    def verify(self) -> None:
        """Perform the same strict synchronous validation used by client startup."""
        from .pantograph_launch import build_launch_spec

        build_launch_spec(self)

    def describe(self) -> str:
        return (
            f"source_root={self.source_root} repl={self.pantograph_repl} "
            f"launcher=stdbuf -oL imports={list(self.imports)}"
        )
