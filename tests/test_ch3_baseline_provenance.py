"""Current baseline inventory protection; no training or simulator episodes."""
import copy
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.ch3_baselines import framework_provenance as provenance
from tools.ch3_baselines import provenance as source_gate
from chapter3_bser.experiments.hgr import provenance as hgr_provenance
from tools.ch3_baselines.provenance import digest, write_json

FINAL_SHA256 = "3bf6035001e28efbe5e5cd3db7b43bc43a11e0bf83ed2037a7ac29c5c518b4bd"


@contextmanager
def final_checkout():
    """An actual LF checkout fixture; no old manifests or checkpoint files."""
    with tempfile.TemporaryDirectory(prefix="ch3-final-checkout-") as temporary:
        root = Path(temporary)
        pin = json.loads(source_gate.PIN.read_text(encoding="utf-8"))
        paths = set(pin["production"]["files"]) | set(provenance.baseline_protected_identity()["files"])
        paths.update((source_gate.PIN.relative_to(provenance.ROOT).as_posix(),
                      provenance.PROTECTED_BASELINE.relative_to(provenance.ROOT).as_posix()))
        for name in paths:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = (provenance.ROOT / name).read_bytes()
            # Simulate Git's existing *.py text eol=lf checkout rule.
            if name in pin["production"]["files"]:
                data = data.replace(b"\r\n", b"\n")
                assert hashlib.sha256(data).hexdigest() == pin["production"]["files"][name]
            destination.write_bytes(data)
        with ExitStack() as stack:
            stack.enter_context(patch.object(source_gate, "PIN", root / source_gate.PIN.relative_to(provenance.ROOT)))
            stack.enter_context(patch.object(provenance, "PROTECTED_BASELINE", root / provenance.PROTECTED_BASELINE.relative_to(provenance.ROOT)))
            stack.enter_context(patch.object(source_gate, "ROOT", root))
            stack.enter_context(patch.object(provenance, "ROOT", root))
            stack.enter_context(patch.object(hgr_provenance, "__file__", str(root / "chapter3_bser/experiments/hgr/provenance.py")))
            yield root


class BaselineProvenanceTests(unittest.TestCase):
    def test_linux_inventory_excludes_windows_bytes_but_launchers_still_run(self):
        sources = provenance.framework_sources()
        current = sources["protected_baseline"]
        self.assertEqual(current, provenance.baseline_protected_identity())
        self.assertEqual(len(current["files"]), 36)
        for prefix, count in (("tools/ch3_baselines/", 9), ("configs/chapter3/baselines/", 6), ("scripts/", 21)):
            self.assertEqual(sum(name.startswith(prefix) for name in current["files"]), count)
        self.assertIn("tools/ch3_baselines/framework_provenance.py", current["files"])
        self.assertIn("tools/ch3_baselines/run_baseline.py", current["files"])
        self.assertIn("scripts/linux/run_ch3_basic_prior_eval.sh", current["files"])
        self.assertEqual(set(sources["framework_entry_scripts"]["files"]), set(provenance.SCRIPT_FILES))
        for inventory in (current, sources["baseline"], sources["framework_entry_scripts"]):
            self.assertFalse(any(Path(name).suffix in (".bat", ".cmd", ".ps1", ".psm1") for name in inventory["files"]))
        self.assertEqual(sources["production_source_sha256"], FINAL_SHA256)
        provenance.verify_framework_sources(sources)
        original_read = Path.read_bytes
        def reject_windows_reads(path):
            if path.suffix.lower() in (".bat", ".cmd", ".ps1", ".psm1"):
                raise AssertionError(f"Linux provenance read a Windows launcher: {path}")
            return original_read(path)
        # Exercise the entire capture_sources + framework chain, not only
        # the top-level scanner. No Windows launcher may be read at all.
        with patch.object(Path, "read_bytes", reject_windows_reads):
            self.assertEqual(sources, provenance.framework_sources())
            provenance.verify_framework_sources(sources)

        windows = {
            "run_ch3_basic_prior_eval.bat": "tools.ch3_baselines.evaluate",
            "run_ch3_baseline_eval.bat": "tools.ch3_baselines.run_baseline",
            "train_ch3_direct_mc.bat": "tools.ch3_baselines.run_training",
            "train_ch3_direct_boundary.bat": "tools.ch3_baselines.run_training",
        }
        for name, module in windows.items():
            path = provenance.ROOT / "scripts" / name
            self.assertTrue(path.is_file())
            self.assertIn("python -B -m " + module, path.read_text(encoding="utf-8"))
            if os.name == "nt":
                # Real Windows entrypoints, real Python help only; no training.
                environment = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
                process = subprocess.run(["cmd.exe", "/d", "/c", str(path), "--help"],
                                         cwd=provenance.ROOT, env=environment, capture_output=True, text=True, timeout=90)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn("usage:", process.stdout)
        learning = provenance.ROOT / "scripts/run_ch3_learning.ps1"
        self.assertTrue(learning.is_file())
        self.assertIn("chapter3_bser.experiments.hgr.cli", learning.read_text(encoding="utf-8"))
        if os.name == "nt":
            shell = shutil.which("powershell") or shutil.which("pwsh")
            self.assertIsNotNone(shell)
            with tempfile.TemporaryDirectory(prefix="baseline-windows-launcher-") as temporary:
                fake_conda = Path(temporary) / "conda-probe.ps1"
                fake_conda.write_text('ConvertTo-Json -InputObject @($args) -Compress\n$global:LASTEXITCODE = 37\n', encoding="utf-8")
                # Run the unchanged PowerShell launcher end to end with a conda
                # stub. Check arguments and exit status without environment setup.
                process = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-File", str(learning), "--help"],
                                         env=dict(os.environ, CRK_CONDA_EXE=str(fake_conda)),
                                         capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 37, process.stderr)
                forwarded = json.loads(process.stdout)
                self.assertEqual(forwarded[:3], ["run", "--no-capture-output", "-n"])
                self.assertEqual(forwarded[4:], ["python", "-B", "-m", "chapter3_bser.experiments.hgr.cli", "--help"])

    def test_changed_removed_and_added_baseline_inputs_are_rejected(self):
        current = provenance.baseline_protected_identity()
        for existing in ("tools/ch3_baselines/run_baseline.py", "configs/chapter3/baselines/search_prior_eval.json",
                         "scripts/linux/run_ch3_baseline_eval.sh"):
            for change in ("changed", "removed", "added"):
                altered = copy.deepcopy(current)
                if change == "changed":
                    altered["files"][existing] = "0" * 64
                elif change == "removed":
                    del altered["files"][existing]
                else:
                    altered["files"]["scripts/linux/new_baseline_entry.sh"] = "0" * 64
                altered["sha256"] = digest(altered["files"])
                with self.subTest(path=existing, change=change), patch.object(provenance, "baseline_protected_identity", return_value=altered):
                    with self.assertRaisesRegex(ValueError, "protected baseline inventory changed"):
                        provenance.framework_sources()
        # Also mutate actual reads of the four baseline Linux launchers.
        original_read = Path.read_bytes
        for name in provenance.SCRIPT_FILES:
            target = provenance.ROOT / name
            def changed_read(path):
                data = original_read(path)
                return data + b"\n# unreviewed launcher change\n" if path == target else data
            with self.subTest(script=name), patch.object(Path, "read_bytes", changed_read):
                with self.assertRaisesRegex(ValueError, "protected baseline inventory changed"):
                    provenance.framework_sources()

    def test_invalid_manifest_or_changed_production_binding_is_rejected(self):
        saved = json.loads(provenance.PROTECTED_BASELINE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="ch3-baseline-hash-validation-") as temporary:
            path = Path(temporary)/"protected.json"
            for field, error in (("sha256", "invalid protected baseline manifest"),
                                 ("production_source_sha256", "CH3-final production source hash")):
                write_json(path, {**saved, field: "0" * 64})
                with self.subTest(field=field), patch.object(provenance, "PROTECTED_BASELINE", path):
                    with self.assertRaisesRegex(ValueError, error):
                        provenance.framework_sources()

    def test_scanner_keeps_linux_byte_protection_and_ignores_windows_eol(self):
        with tempfile.TemporaryDirectory(prefix="ch3-baseline-hash-scan-") as temporary:
            root = Path(temporary)
            inputs = {
                "tools/ch3_baselines/example.py": b"# source\n",
                "configs/chapter3/baselines/example.json": b"{}\n",
                "scripts/dot.sh": b"python -m tools.ch3_baselines.run_baseline\n",
                "scripts/linux/slash.sh": b"python tools/ch3_baselines/run_baseline.py\n",
                "scripts/unrelated.sh": b"echo unrelated\n",
            }
            windows = {
                "scripts/slash.ps1": b"python tools/ch3_baselines/run_baseline.py\r\n",
                "scripts/backslash.bat": b"python tools\\ch3_baselines\\run_baseline.py\r\n",
                "scripts/other.cmd": b"python -m tools.ch3_baselines.run_baseline\r\n",
            }
            for name, content in {**inputs, **windows}.items():
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            current = provenance.baseline_protected_identity(root)
            self.assertEqual(current["files"], {name: hashlib.sha256(content).hexdigest() for name, content in inputs.items()})
            for name, content in windows.items():
                for replacement in (content.replace(b"\r\n", b"\n"), b"changed Windows-only command\r\n"):
                    (root/name).write_bytes(replacement)
                    self.assertEqual(current, provenance.baseline_protected_identity(root))
            added = root/"scripts/linux/new_baseline.sh"
            added.write_bytes(b"python -m tools.ch3_baselines.run_baseline\n")
            self.assertIn(added.relative_to(root).as_posix(), provenance.baseline_protected_identity(root)["files"])
            self.assertNotEqual(current, provenance.baseline_protected_identity(root))
            added.unlink()
            self.assertEqual(current, provenance.baseline_protected_identity(root))
            linux = root/"scripts/dot.sh"
            for replacement in (inputs["scripts/dot.sh"].replace(b"\n", b"\r\n"), b"echo removed baseline dispatch\n"):
                linux.write_bytes(replacement)
                self.assertNotEqual(current, provenance.baseline_protected_identity(root))

    def test_linux_provenance_platform_independent(self):
        with final_checkout() as root:
            before = provenance.framework_sources()
            self.assertEqual(before["production"]["sha256"], FINAL_SHA256)
            self.assertEqual(before["production_checkout_profile"], "git_lf")
            self.assertEqual(before, source_gate.capture_sources())
            for ending in (b"\r\n", b"\n"):
                for suffix in (".bat", ".ps1"):
                    (root / ("scripts/windows" + suffix)).write_bytes(b"python --help" + ending)
                self.assertEqual(before, provenance.framework_sources())
            # A fresh process imports only the copied checkout. Neither the old
            # frozen manifest nor any checkpoint is present or needed.
            code = ("from tools.ch3_baselines.framework_provenance import framework_sources; "
                    "s=framework_sources(); print(s['production']['sha256']); print('PASS')")
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [FINAL_SHA256, "PASS"])

    def test_real_source_inventory_changes_fail_for_every_protected_directory(self):
        with final_checkout() as root:
            examples = (
                "core/runtime/engine.py", "chapter3_bser/controllers/action_adapter.py",
                "tools/ch3_baselines/basic_search_prior.py", "configs/chapter3/baselines/search_prior_eval.json",
                "scripts/linux/run_ch3_learning.sh",
            )
            before = provenance.framework_sources()
            for name in examples:
                path = root / name
                original = path.read_bytes()
                added = path.with_name("new_unreviewed" + path.suffix)
                for change in ("changed", "removed", "added"):
                    with self.subTest(path=name, change=change):
                        try:
                            if change == "changed":
                                path.write_bytes(original + b"\n")
                            elif change == "removed":
                                path.unlink()
                            else:
                                added.write_bytes(original)
                            with self.assertRaisesRegex(ValueError, "source mismatch|protected baseline inventory changed"):
                                provenance.framework_sources()
                            with self.assertRaises(ValueError):
                                source_gate.capture_sources()
                        finally:
                            path.write_bytes(original)
                            added.unlink(missing_ok=True)
                self.assertEqual(before, provenance.framework_sources())
            # No general CRLF allowance: an unreviewed source representation fails.
            path = root / "core/runtime/engine.py"
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                provenance.framework_sources()

    def test_documentation_and_historical_metadata_are_not_runtime_inputs(self):
        with final_checkout() as root:
            before = provenance.framework_sources()
            (root / "README.md").write_text("documentation only\n", encoding="utf-8")
            (root / "docs/notes.md").write_text("documentation only\n", encoding="utf-8")
            for path in (source_gate.PIN, provenance.PROTECTED_BASELINE):
                record = json.loads(path.read_text(encoding="utf-8"))
                record["review_note"] = "Editorial metadata does not change source identity."
                path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(before, provenance.framework_sources())
            self.assertFalse(any("manifest_sha256" in key or key == "pin_sha256" for key in before))

    def test_final_pin_integrity_and_native_checkpoint_identity_are_retained(self):
        pin = json.loads(source_gate.PIN.read_text(encoding="utf-8"))
        source = provenance.framework_sources()
        self.assertEqual(source["production"], hgr_provenance.fresh_source_identity())
        self.assertEqual(pin["production"]["sha256"], FINAL_SHA256)
        self.assertEqual(len(pin["production"]["files"]), 196)
        with final_checkout():
            current_pin = json.loads(source_gate.PIN.read_text(encoding="utf-8"))
            current_pin["production"]["files"]["core/runtime/engine.py"] = "0" * 64
            write_json(source_gate.PIN, current_pin)
            with self.assertRaisesRegex(ValueError, "inventory/aggregate hash mismatch"):
                provenance.framework_sources()


if __name__ == "__main__":
    unittest.main()
