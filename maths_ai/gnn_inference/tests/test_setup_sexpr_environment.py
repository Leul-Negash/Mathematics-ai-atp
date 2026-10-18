from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pantograph_async import PantographConfigurationError, compatibility

from maths_ai.gnn_inference.scripts import setup_sexpr_environment as setup
from maths_ai.gnn_inference.scripts.setup_sexpr_environment import (
    LEAN_TOOLCHAIN,
    _build_manifest,
    _ensure_toolchain,
    _prepare_companion,
    _require_stdbuf,
)


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


class SetupSExprEnvironmentTests(unittest.TestCase):
    @patch(
        "maths_ai.gnn_inference.scripts.setup_sexpr_environment.shutil.which",
        return_value="/usr/bin/stdbuf",
    )
    def test_stdbuf_dependency_resolves_from_runtime_path(self, which):
        self.assertEqual(_require_stdbuf(), "/usr/bin/stdbuf")
        which.assert_called_once_with("stdbuf")

    @patch(
        "maths_ai.gnn_inference.scripts.setup_sexpr_environment.shutil.which",
        return_value=None,
    )
    def test_missing_stdbuf_is_a_named_setup_error(self, _which):
        with self.assertRaisesRegex(RuntimeError, "Install coreutils"):
            _require_stdbuf()

    @patch("maths_ai.gnn_inference.scripts.setup_sexpr_environment._run")
    @patch("maths_ai.gnn_inference.scripts.setup_sexpr_environment._output")
    def test_already_installed_toolchain_is_not_reinstalled(self, output, run):
        output.return_value = f"{LEAN_TOOLCHAIN}\nleanprover/lean4:v4.29.1 (default)"
        _ensure_toolchain(LEAN_TOOLCHAIN)
        output.assert_called_once_with("elan", "toolchain", "list")
        run.assert_not_called()

    @patch("maths_ai.gnn_inference.scripts.setup_sexpr_environment._run")
    @patch("maths_ai.gnn_inference.scripts.setup_sexpr_environment._output")
    def test_missing_toolchain_is_installed(self, output, run):
        output.return_value = "leanprover/lean4:v4.29.1"
        _ensure_toolchain(LEAN_TOOLCHAIN)
        run.assert_called_once_with("elan", "toolchain", "install", LEAN_TOOLCHAIN)


class PinsTests(unittest.TestCase):
    def test_pins_come_from_the_installed_client(self):
        base = compatibility.companion_base()
        self.assertEqual(setup.PANTOGRAPH_COMMIT, base["base_commit"])
        self.assertEqual(setup.PANTOGRAPH_URL, base["source_url"])
        self.assertEqual(setup.MATHLIB_COMMIT, base["mathlib_expected_pin"])
        self.assertEqual(LEAN_TOOLCHAIN, base["toolchain"])


class CompanionCheckoutTests(unittest.TestCase):
    """Real git checkouts with a small stand-in for the companion patch."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        upstream = self.root / "upstream"
        upstream.mkdir()
        _git("init", "-q", cwd=upstream)
        for key, value in (("user.name", "test"), ("user.email", "t@example.com")):
            _git("config", key, value, cwd=upstream)
        (upstream / "lean-toolchain").write_text("leanprover/lean4:v4.10.0-rc1\n")
        (upstream / "Repl.lean").write_text("def main := 0\n")
        _git("add", ".", cwd=upstream)
        _git("commit", "-q", "-m", "base", cwd=upstream)
        self.commit = _git("rev-parse", "HEAD", cwd=upstream)
        (upstream / "Repl.lean").write_text("def main := 1\n")
        (upstream / "Validation.lean").write_text("-- companion\n")
        _git("add", "-N", "Validation.lean", cwd=upstream)
        patch_bytes = _git("diff", "--binary", cwd=upstream).encode() + b"\n"
        digests = {
            name: hashlib.sha256((upstream / name).read_bytes()).hexdigest()
            for name in ("Repl.lean", "Validation.lean")
        }
        _git("checkout", "-q", "--", "Repl.lean", cwd=upstream)
        _git("rm", "-q", "--cached", "Validation.lean", cwd=upstream)
        (upstream / "Validation.lean").unlink()

        def verify(destination):
            for name, digest in digests.items():
                path = Path(destination) / name
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise PantographConfigurationError(f"differ: {name}")

        fake = SimpleNamespace(
            companion_patch=lambda: patch_bytes,
            companion_files=lambda: frozenset(digests),
            verify_companion_sources=verify,
        )
        self.destination = self.root / "Pantograph"
        for target, value in (
            ("compatibility", fake),
            ("PANTOGRAPH_URL", str(upstream)),
            ("PANTOGRAPH_COMMIT", self.commit),
        ):
            patcher = patch.object(setup, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_fresh_checkout_applies_and_verifies_the_patch(self):
        _prepare_companion(self.destination)
        self.assertEqual(_git("rev-parse", "HEAD", cwd=self.destination), self.commit)
        self.assertEqual((self.destination / "Repl.lean").read_text(), "def main := 1\n")
        self.assertTrue((self.destination / "Validation.lean").is_file())

    def test_rerun_reuses_an_exact_application_without_touching_files(self):
        _prepare_companion(self.destination)
        before = (self.destination / "Repl.lean").stat().st_mtime_ns
        _prepare_companion(self.destination)
        self.assertEqual((self.destination / "Repl.lean").stat().st_mtime_ns, before)

    def test_user_changes_are_refused_and_preserved(self):
        _prepare_companion(self.destination)
        (self.destination / "Notes.lean").write_text("-- mine\n")
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            _prepare_companion(self.destination)
        self.assertEqual((self.destination / "Notes.lean").read_text(), "-- mine\n")
        self.assertEqual((self.destination / "Repl.lean").read_text(), "def main := 1\n")

    def test_build_manifest_records_source_and_binary_identities(self):
        _prepare_companion(self.destination)
        repl = self.destination / ".lake" / "build" / "bin" / "repl"
        repl.parent.mkdir(parents=True)
        repl.write_bytes(b"built")
        manifest = _build_manifest(self.destination, self.destination, repl)
        self.assertEqual(manifest["pantograph_base_commit"], self.commit)
        self.assertEqual(manifest["mathlib_commit"], self.commit)
        self.assertEqual(manifest["toolchain"], LEAN_TOOLCHAIN)
        self.assertEqual(manifest["repl_sha256"], hashlib.sha256(b"built").hexdigest())
        self.assertEqual(manifest["patch_sha256"], setup.COMPANION["patch_sha256"])

    def test_edited_companion_file_is_refused_and_preserved(self):
        _prepare_companion(self.destination)
        (self.destination / "Repl.lean").write_text("def main := 2\n")
        with self.assertRaises(PantographConfigurationError):
            _prepare_companion(self.destination)
        self.assertEqual((self.destination / "Repl.lean").read_text(), "def main := 2\n")


if __name__ == "__main__":
    unittest.main()
