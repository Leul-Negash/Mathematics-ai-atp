"""Pure explicit-launch validation for the repository-owned Pantograph client.

The existing application factory is migrated in Stage 6. This specification
never discovers a bundled executable or imports PyPantograph.
"""

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING

from .pantograph_protocol import (
    PantographConfigurationError,
    PantographProtocolError,
    REQUIRED_OPTIONS,
    decode_json_response,
    encode_request,
)

if TYPE_CHECKING:
    from .pantograph_env import PantographEnv


@dataclass(frozen=True)
class PantographLaunchSpec:
    source_root: Path
    repl: Path
    lake: Path
    stdbuf: Path
    imports: tuple[str, ...]
    options: tuple[tuple[str, bool], ...]
    toolchain: str
    request_timeout_s: float
    startup_timeout_s: float
    shutdown_timeout_s: float
    response_line_limit: int
    stderr_tail_limit: int

    @property
    def command(self) -> tuple[str, ...]:
        return (str(self.stdbuf), "-oL", str(self.repl), *self.imports)


def build_launch_spec(env: "PantographEnv") -> PantographLaunchSpec:
    """Reject incomplete owned-client launches before spawning any process."""
    try:
        if env.source_root is None or env.pantograph_repl is None:
            raise PantographConfigurationError(
                "Owned clients require explicit source_root and pantograph_repl."
            )
        root = Path(env.source_root).resolve()
        repl = Path(env.pantograph_repl).resolve()
        if not root.is_dir() or not any(
            (root / name).is_file() for name in ("lakefile.lean", "lakefile.toml")
        ):
            raise PantographConfigurationError(
                f"source_root is not a Lake project: {root}"
            )
        toolchain = (root / "lean-toolchain").read_text().strip()
        if not toolchain:
            raise PantographConfigurationError(
                f"Empty lean-toolchain declaration: {root}"
            )
        if not repl.is_file() or not os.access(repl, os.X_OK):
            raise PantographConfigurationError(
                f"pantograph_repl is not executable: {repl}"
            )
        if repl.parts[-4:-1] == (".lake", "build", "bin"):
            repl_root = repl.parents[3]
            if not any(
                (repl_root / name).is_file()
                for name in ("lakefile.lean", "lakefile.toml")
            ):
                raise PantographConfigurationError(
                    f"REPL checkout has no Lake project: {repl_root}"
                )
            repl_toolchain = (repl_root / "lean-toolchain").read_text().strip()
        else:
            if env.build_manifest is None:
                raise PantographConfigurationError(
                    "An executable outside a checkout requires an explicit build_manifest."
                )
            manifest = decode_json_response(Path(env.build_manifest).read_bytes())
            repl_toolchain = manifest.get("toolchain")
            with repl.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if manifest.get("repl_sha256") != digest:
                raise PantographConfigurationError(
                    "Build manifest executable hash mismatch."
                )
        if not repl_toolchain or repl_toolchain != toolchain:
            raise PantographConfigurationError(
                f"Lean toolchain mismatch or missing declaration: source={toolchain!r}, repl={repl_toolchain!r}."
            )
        if (
            not isinstance(env.imports, (tuple, list))
            or not env.imports
            or any(
                not isinstance(name, str)
                or not re.fullmatch(r"[^\W\d]\w*(?:\.[^\W\d]\w*)*", name)
                for name in env.imports
            )
        ):
            raise PantographConfigurationError(
                "imports must contain Lean module names, not flags or paths."
            )
        options = {**REQUIRED_OPTIONS, **dict(env.options)}
        encode_request("options.set", options)
        timings = (
            env.timeout,
            env.server_start_timeout_s,
            env.worker_shutdown_timeout_s,
        )
        if any(
            type(value) not in (float, int) or not math.isfinite(value) or value <= 0
            for value in timings
        ):
            raise PantographConfigurationError(
                "Request, startup and shutdown timeouts must be finite and positive."
            )
        limits = (env.response_line_limit, env.stderr_tail_limit)
        if any(type(value) is not int or value <= 0 for value in limits):
            raise PantographConfigurationError(
                "Response and stderr limits must be positive integers."
            )
        lake, stdbuf = shutil.which("lake"), shutil.which("stdbuf")
        if lake is None or stdbuf is None:
            raise PantographConfigurationError(
                "Owned launches require Lake and GNU stdbuf on PATH."
            )
        return PantographLaunchSpec(
            root,
            repl,
            Path(lake).resolve(),
            Path(stdbuf).resolve(),
            tuple(env.imports),
            tuple(sorted(options.items())),
            toolchain,
            *map(float, timings),
            *limits,
        )
    except PantographConfigurationError:
        raise
    except (OSError, ValueError, TypeError, PantographProtocolError) as exc:
        raise PantographConfigurationError(
            f"Invalid explicit Pantograph environment: {exc}"
        ) from exc
