#!/usr/bin/env python3
"""Build the pinned Mathlib and companion Pantograph model-S-expression environment.

Pins and the companion patch come from the installed ``pantograph-async-client``
package, so the executable matches the contract the client decodes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from pantograph_async import PantographEnv, compatibility
from pantograph_async.diagnostics import distribution_version, verify_live_client


COMPANION = compatibility.companion_base()
MATHLIB_URL = "https://github.com/leanprover-community/mathlib4.git"
MATHLIB_COMMIT = COMPANION["mathlib_expected_pin"]
PANTOGRAPH_URL = COMPANION["source_url"]
PANTOGRAPH_COMMIT = COMPANION["base_commit"]
LEAN_TOOLCHAIN = COMPANION["toolchain"]


def _require_stdbuf() -> str:
    """Return the GNU line-buffering launcher required by the custom REPL."""
    executable = shutil.which("stdbuf")
    if executable is None:
        raise RuntimeError(
            "GNU stdbuf is required to run the pinned Pantograph REPL "
            "interactively. Install coreutils before provisioning the environment."
        )
    return str(Path(executable).resolve())


def _run(*command: str, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _output(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _status_porcelain(repository: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _ensure_toolchain(toolchain: str) -> None:
    installed = {
        line.split()[0]
        for line in _output("elan", "toolchain", "list").splitlines()
        if line.split()
    }
    if toolchain in installed:
        print(f"Lean toolchain is already installed: {toolchain}")
        return
    _run("elan", "toolchain", "install", toolchain)


def _changed_paths(repository: Path) -> set[str]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line[3:] for line in status.splitlines() if len(line) > 3}


def _checkout(url: str, commit: str, destination: Path) -> None:
    """Fetch and detach a clean checkout; never discard local modifications."""
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", "--filter=blob:none", "--no-checkout", url, str(destination))
    if not (destination / ".git").exists():
        raise RuntimeError(f"Refusing to reuse non-git directory: {destination}")
    current_origin = _output("git", "remote", "get-url", "origin", cwd=destination)
    if current_origin != url:
        _run("git", "remote", "set-url", "origin", url, cwd=destination)
    worktree_is_empty = not any(path.name != ".git" for path in destination.iterdir())
    if _status_porcelain(destination) and not worktree_is_empty:
        raise RuntimeError(
            f"Checkout has local modifications; refusing to overwrite: {destination}"
        )
    _run("git", "fetch", "--depth=1", "origin", commit, cwd=destination)
    _run("git", "checkout", "--detach", commit, cwd=destination)


def _prepare_companion(destination: Path) -> None:
    """Apply the client's companion patch, or reuse an exact earlier application."""
    if (
        (destination / ".git").exists()
        and _status_porcelain(destination)
        and _output("git", "rev-parse", "HEAD", cwd=destination) == PANTOGRAPH_COMMIT
        and _changed_paths(destination) == compatibility.companion_files()
    ):
        compatibility.verify_companion_sources(destination)
        print(f"Companion sources are already applied: {destination}")
        return
    _checkout(PANTOGRAPH_URL, PANTOGRAPH_COMMIT, destination)
    patch = compatibility.companion_patch()
    for check in (("--check",), ()):
        print("+ git apply", *check, "<companion patch>", flush=True)
        subprocess.run(
            ["git", "apply", *check, "-"], cwd=destination, input=patch, check=True
        )
    compatibility.verify_companion_sources(destination)


def _build_manifest(mathlib: Path, pantograph: Path, repl: Path) -> dict:
    """Identities of the reproduced build; also a valid client build_manifest."""
    with repl.open("rb") as stream:
        repl_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "kind": "companion-repl-build",
        "client_version": distribution_version(),
        "contract_revision": COMPANION["contract_revision"],
        "pantograph_source_url": PANTOGRAPH_URL,
        "pantograph_base_commit": _output("git", "rev-parse", "HEAD", cwd=pantograph),
        "patch_sha256": COMPANION["patch_sha256"],
        "patched_sources_verified": True,
        "toolchain": (pantograph / "lean-toolchain").read_text().strip(),
        "mathlib_commit": _output("git", "rev-parse", "HEAD", cwd=mathlib),
        "repl": str(repl),
        "repl_sha256": repl_sha256,
    }


def _smoke(mathlib: Path, repl: Path) -> dict:
    """Run the installed client's live gate against the newly built executable."""
    report = asyncio.run(
        verify_live_client(
            PantographEnv(source_root=mathlib, pantograph_repl=repl, imports=("Mathlib",))
        )
    )
    return {
        "checks": report["checks"],
        "scoped_proofs": report["scoped_proofs"],
        "final_states": report["final_states"],
        "subprocess_reaped": report["subprocess_reaped"],
    }


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "maths_ai" / "_support_files" / "sexpr_environment",
    )
    parser.add_argument("--skip-mathlib-cache", action="store_true")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Do not run the client's live checks against the new executable.",
    )
    args = parser.parse_args(argv)

    stdbuf = _require_stdbuf()
    output_root = args.output_root.resolve()
    mathlib = output_root / "mathlib4"
    pantograph = output_root / "Pantograph"
    _ensure_toolchain(LEAN_TOOLCHAIN)
    _checkout(MATHLIB_URL, MATHLIB_COMMIT, mathlib)
    if not args.skip_mathlib_cache:
        _run("lake", "exe", "cache", "get", cwd=mathlib)
    _prepare_companion(pantograph)
    _run("lake", "build", "repl", cwd=pantograph)

    repl = pantograph / ".lake" / "build" / "bin" / "repl"
    if not repl.is_file():
        raise RuntimeError(f"Pantograph build completed without REPL binary: {repl}")
    manifest = _build_manifest(mathlib, pantograph, repl)
    manifest["smoke"] = None if args.skip_smoke else _smoke(mathlib, repl)
    manifest_path = output_root / "companion_build.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("\nS-expression environment is ready:")
    print(f"  --source-root {mathlib}")
    print(f"  --pantograph-repl {repl}")
    print(f"  line-buffer launcher {stdbuf} -oL")
    print(f"  build manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
