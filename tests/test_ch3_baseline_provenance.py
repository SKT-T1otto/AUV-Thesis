"""Current baseline inventory protection; no training or simulator episodes."""
import copy
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
from tools.ch3_baselines.provenance import digest, write_json


class BaselineProvenanceTests(unittest.TestCase):
    def test_linux_inventory_excludes_windows_bytes_but_launchers_still_run(self):
        sources = provenance.framework_sources()
        current = sources["protected_baseline"]
        self.assertEqual(current, provenance.baseline_protected_identity())
        self.assertEqual(len(current["files"]), 19)
        for prefix, count in (("tools/ch3_baselines/", 9), ("configs/chapter3/baselines/", 6), ("scripts/", 4)):
            self.assertEqual(sum(name.startswith(prefix) for name in current["files"]), count)
        self.assertIn("tools/ch3_baselines/framework_provenance.py", current["files"])
        self.assertIn("tools/ch3_baselines/run_baseline.py", current["files"])
        self.assertIn("scripts/linux/run_ch3_basic_prior_eval.sh", current["files"])
        self.assertEqual(set(sources["framework_entry_scripts"]["files"]), set(provenance.SCRIPT_FILES))
        for inventory in (current, sources["baseline"], sources["framework_entry_scripts"]):
            self.assertFalse(any(Path(name).suffix in (".bat", ".cmd", ".ps1", ".psm1") for name in inventory["files"]))
        self.assertEqual(sources["production"]["sha256"], "7a8dda612fb68c0a63bcc38b04ee0973ac80488bddd6c9fffe1fbf9b2f23f491")
        provenance.verify_framework_sources(sources)
        original_read = Path.read_bytes
        def reject_windows_reads(path):
            if path.suffix.lower() in (".bat", ".cmd", ".ps1", ".psm1"):
                raise AssertionError(f"Linux provenance read a Windows launcher: {path}")
            return original_read(path)
        # Exercise the entire B0 + capture_sources + framework chain, not only
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
        # Also mutate actual reads of all four Linux launchers. The historical
        # B0 gate or current framework gate must reject every changed script.
        original_read = Path.read_bytes
        for name in provenance.SCRIPT_FILES:
            target = provenance.ROOT / name
            def changed_read(path):
                data = original_read(path)
                return data + b"\n# unreviewed launcher change\n" if path == target else data
            with self.subTest(script=name), patch.object(Path, "read_bytes", changed_read):
                with self.assertRaisesRegex(ValueError, "protected B0 implementation changed|protected baseline inventory changed"):
                    provenance.framework_sources()

    def test_invalid_manifest_or_changed_production_binding_is_rejected(self):
        saved = json.loads(provenance.PROTECTED_BASELINE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="ch3-baseline-hash-validation-") as temporary:
            path = Path(temporary)/"protected.json"
            for field, error in (("sha256", "invalid protected baseline manifest"),
                                 ("production_source_sha256", "frozen production source hash")):
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
            }
            windows = {
                "scripts/slash.ps1": b"python tools/ch3_baselines/run_baseline.py\r\n",
                "scripts/backslash.bat": b"python tools\\ch3_baselines\\run_baseline.py\r\n",
                "scripts/other.cmd": b"python -m tools.ch3_baselines.run_baseline\r\n",
            }
            for name, content in {**inputs, **windows, "scripts/unrelated.sh": b"echo unrelated\n"}.items():
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


if __name__ == "__main__":
    unittest.main()
