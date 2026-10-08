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
    """Current sealed Git checkout, including every frozen historical seal."""
    from tests.test_safe_search_provenance import checked_copy
    from chapter3_bser.experiments.d2_v1 import provenance as d2
    with checked_copy() as root:
        # Reproduce Git's text conversion for sealed reference metadata too.
        preserved = {line.split()[0] for line in (root / ".gitattributes").read_text().splitlines()
                     if line.endswith("-text !eol")}
        for path in (root / "docs").rglob("*"):
            if (path.is_file() and path.suffix in (".md", ".json")
                    and path.relative_to(root).as_posix() not in preserved):
                path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        from chapter3_bser.experiments.d2_performance.provenance import MANIFEST as CURRENT_MANIFEST
        record = d2.old.read_json(root / CURRENT_MANIFEST)["profiles"]["git_clone_preserved"]
        for name, expected in record["files"].items():
            path = root / name
            data = path.read_bytes()
            alternatives = (data, data.replace(b"\r\n", b"\n"))
            selected = next((v for v in alternatives if hashlib.sha256(v).hexdigest() == expected), None)
            assert selected is not None, name
            path.write_bytes(selected)
        with ExitStack() as stack:
            stack.enter_context(patch.object(source_gate, "PIN", root / source_gate.PIN.relative_to(provenance.ROOT)))
            stack.enter_context(patch.object(provenance, "PROTECTED_BASELINE", root / provenance.PROTECTED_BASELINE.relative_to(provenance.ROOT)))
            stack.enter_context(patch.object(source_gate, "ROOT", root))
            stack.enter_context(patch.object(provenance, "ROOT", root))
            stack.enter_context(patch.object(d2.old, "ROOT", root))
            stack.enter_context(patch.object(hgr_provenance, "__file__", str(root / "chapter3_bser/experiments/hgr/provenance.py")))
            yield root


class BaselineProvenanceTests(unittest.TestCase):
    def test_baseline_maddpg_evolution_exists_and_records_independent_methods(self):
        path = provenance.ROOT / "docs/provenance/baseline_maddpg_evolution.json"
        self.assertTrue(path.is_file(), "B2/B3 provenance evolution must be included in the repository")
        self.assertEqual(path, source_gate.ROOT / source_gate.EVOLUTION_RELATIVE)
        evolution = source_gate.baseline_evolution()
        expected = {
            "B2_direct_mc": dict(algorithm="maddpg", architecture_version="ch3.baseline.maddpg.v1",
                                 independent_from="hgr", trainer="DirectMCTrainer"),
            "B3_direct_boundary": dict(algorithm="direct_boundary_maddpg",
                architecture_version="ch3.baseline.boundary_maddpg.v1", independent_from="hgr",
                trainer="DirectBoundaryTrainer", boundary_encoder=dict(schema="ch3.baseline.boundary_encoder.v1")),
        }
        self.assertEqual(evolution["baselines"], expected)
        for baseline, config_name in (("B2_direct_mc", "direct_mc_train.json"),
                                      ("B3_direct_boundary", "direct_boundary_train.json")):
            config = json.loads((provenance.ROOT / "configs/chapter3/baselines" / config_name).read_text(encoding="utf-8"))
            self.assertEqual(config["baseline"], baseline)
            for field in ("algorithm", "architecture_version"):
                self.assertEqual(evolution["baselines"][baseline][field], config[field])

    def test_linux_inventory_excludes_windows_bytes_but_launchers_still_run(self):
        sources = provenance.framework_sources()
        current = sources["protected_baseline"]
        self.assertEqual(current, provenance.baseline_protected_identity())
        self.assertEqual(current["files"], {k:v for k,v in sources["inventory"]["files"].items()
                         if k.startswith(("tools/ch3_baselines/", "configs/chapter3/baselines/")) or k.endswith(".sh")})
        for prefix, count in (("tools/ch3_baselines/", 10), ("configs/chapter3/baselines/", 6)):
            self.assertEqual(sum(name.startswith(prefix) for name in current["files"]), count)
        self.assertIn("tools/ch3_baselines/framework_provenance.py", current["files"])
        self.assertIn("tools/ch3_baselines/run_baseline.py", current["files"])
        self.assertIn("scripts/linux/run_ch3_basic_prior_eval.sh", current["files"])
        self.assertEqual(set(sources["framework_entry_scripts"]["files"]), set(provenance.SCRIPT_FILES))
        for inventory in (current, sources["baseline"], sources["framework_entry_scripts"]):
            self.assertFalse(any(Path(name).suffix in (".bat", ".cmd", ".ps1", ".psm1") for name in inventory["files"]))
        self.assertEqual(sources["historical_production_sha256"], FINAL_SHA256)
        self.assertEqual(sources["production_source_sha256"], sources["inventory"]["sha256"])
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
        with final_checkout() as root:
            for name in ("tools/ch3_baselines/run_baseline.py", "configs/chapter3/baselines/search_prior_eval.json",
                         *provenance.SCRIPT_FILES):
                path = root / name
                data = path.read_bytes()
                added = path.with_name("unreviewed_new" + path.suffix)
                for mode in ("changed", "removed", "added"):
                    try:
                        if mode == "changed": path.write_bytes(data + b"\n")
                        elif mode == "removed": path.unlink()
                        else: added.write_bytes(data)
                        with self.subTest(path=name, mode=mode), self.assertRaises((ValueError, FileNotFoundError)):
                            provenance.framework_sources()
                    finally:
                        path.write_bytes(data)
                        added.unlink(missing_ok=True)

    def test_invalid_manifest_or_changed_production_binding_is_rejected(self):
        with final_checkout():
            path = provenance.PROTECTED_BASELINE
            raw = path.read_bytes()
            saved = json.loads(raw)
            for field in ("sha256", "production_source_sha256"):
                try:
                    write_json(path, {**saved, field: "0" * 64})
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        provenance.framework_sources()
                finally:
                    path.write_bytes(raw)

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
            self.assertEqual(before["historical_production_sha256"], FINAL_SHA256)
            self.assertEqual(set(before["production"]["files"]),
                             {name for name in before["inventory"]["files"] if name.startswith(("core/", "chapter3_bser/")) and name.endswith(".py")})
            self.assertEqual(before["production_checkout_profile"], "git_clone_preserved")
            self.assertEqual(before, source_gate.capture_sources())
            for ending in (b"\r\n", b"\n"):
                for suffix in (".bat", ".ps1"):
                    (root / ("scripts/windows" + suffix)).write_bytes(b"python --help" + ending)
                self.assertEqual(before, provenance.framework_sources())
            # A fresh process imports the sealed checkout and all historical
            # metadata. No checkpoint or Git history is needed.
            code = ("from tools.ch3_baselines.framework_provenance import framework_sources; "
                    "s=framework_sources(); print(s['production']['sha256']); print('PASS')")
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [before["production"]["sha256"], "PASS"])

    def test_real_source_inventory_changes_fail_for_every_protected_directory(self):
        with final_checkout() as root:
            examples = (
                "core/runtime/engine.py", "chapter3_bser/controllers/action_adapter.py",
                "tools/ch3_baselines/basic_search_prior.py", "configs/chapter3/baselines/search_prior_eval.json",
                "scripts/linux/run_ch3_learning.sh",
                "chapter3_bser/experiments/baselines/common/train.py",
                "chapter3_bser/experiments/baselines/common/model.py",
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
                            with self.assertRaisesRegex(ValueError, "source mismatch|source inventory mismatch|protected baseline inventory changed|frozen D2 reference changed"):
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
            with self.assertRaisesRegex(ValueError, "source inventory mismatch"):
                provenance.framework_sources()

    def test_unsealed_notes_are_ignored_but_historical_metadata_is_frozen(self):
        with final_checkout() as root:
            before = provenance.framework_sources()
            (root / "README.md").write_text("documentation only\n", encoding="utf-8")
            (root / "docs/notes.md").write_text("documentation only\n", encoding="utf-8")
            self.assertEqual(before, provenance.framework_sources())
            for path in (source_gate.PIN, provenance.PROTECTED_BASELINE):
                raw = path.read_bytes()
                try:
                    record = json.loads(raw)
                    record["review_note"] = "Unreviewed metadata change"
                    path.write_text(json.dumps(record), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "frozen historical manifest changed"):
                        provenance.framework_sources()
                finally:
                    path.write_bytes(raw)
            self.assertEqual(before, provenance.framework_sources())

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
            with self.assertRaisesRegex(ValueError, "frozen historical manifest changed"):
                provenance.framework_sources()

    def test_evolution_preserves_all_historical_records_and_exact_new_hashes(self):
        evolution = source_gate.baseline_evolution()
        pin = json.loads(source_gate.PIN.read_text(encoding="utf-8"))
        historical = pin["production"]["files"]
        self.assertEqual(len(historical), 196)
        self.assertFalse(set(historical) & set(evolution["added_production"]))
        self.assertEqual(evolution["historical_production_sha256"], FINAL_SHA256)
        self.assertTrue(evolution["added_production"])
        from chapter3_bser.experiments.d2_v1 import provenance as d2
        current = provenance.framework_sources()
        transition = d2.old.read_json(d2.old.ROOT / d2.MANIFEST)["profiles"][current["checkout_profile"]]
        from chapter3_bser.experiments.d2_suite_v1.provenance import MANIFEST as SUITE_MANIFEST
        successor = d2.old.read_json(d2.old.ROOT / SUITE_MANIFEST)["profiles"][current["checkout_profile"]]
        from chapter3_bser.experiments.d2_performance.provenance import MANIFEST as PERFORMANCE_MANIFEST
        performance = d2.old.read_json(d2.old.ROOT / PERFORMANCE_MANIFEST)["profiles"][current["checkout_profile"]]
        for name, expected in evolution["added_production"].items():
            self.assertEqual(transition["changes"].get(name, {}).get("before", expected), expected)
            d2_after = transition["changes"].get(name, {}).get("after", expected)
            self.assertEqual(successor["changes"].get(name, {}).get("before", d2_after), d2_after)
            suite_after = successor["changes"].get(name, {}).get("after", d2_after)
            self.assertEqual(performance["changes"].get(name, {}).get("before", suite_after), suite_after)
            self.assertEqual(source_gate.file_sha256(provenance.ROOT / name),
                             performance["changes"].get(name, {}).get("after", suite_after))
        with final_checkout() as root:
            path = root / source_gate.EVOLUTION_RELATIVE
            value = json.loads(path.read_text())
            value["added_production"]["core/runtime/engine.py"] = "0" * 64
            value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
            write_json(path, value)
            with self.assertRaisesRegex(ValueError, "frozen historical manifest changed"):
                provenance.framework_sources()


if __name__ == "__main__":
    unittest.main()
