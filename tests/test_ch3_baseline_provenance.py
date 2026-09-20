"""Current baseline inventory protection; no training or simulator episodes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.ch3_baselines import framework_provenance as provenance
from tools.ch3_baselines.provenance import digest, write_json


class BaselineProvenanceTests(unittest.TestCase):
    def test_current_inventory_covers_tools_configs_and_all_eight_entry_scripts(self):
        sources = provenance.framework_sources()
        current = sources["protected_baseline"]
        self.assertEqual(current, provenance.baseline_protected_identity())
        self.assertEqual(len(current["files"]), 23)
        for prefix, count in (("tools/ch3_baselines/", 9), ("configs/chapter3/baselines/", 6), ("scripts/", 8)):
            self.assertEqual(sum(name.startswith(prefix) for name in current["files"]), count)
        self.assertIn("tools/ch3_baselines/framework_provenance.py", current["files"])
        self.assertIn("tools/ch3_baselines/run_baseline.py", current["files"])
        self.assertIn("scripts/run_ch3_basic_prior_eval.bat", current["files"])
        self.assertEqual(sources["production"]["sha256"], "7a8dda612fb68c0a63bcc38b04ee0973ac80488bddd6c9fffe1fbf9b2f23f491")
        provenance.verify_framework_sources(sources)

    def test_changed_removed_and_added_baseline_inputs_are_rejected(self):
        current = provenance.baseline_protected_identity()
        existing = "tools/ch3_baselines/run_baseline.py"
        for change in ("changed", "removed", "added"):
            altered = copy.deepcopy(current)
            if change == "changed":
                altered["files"][existing] = "0" * 64
            elif change == "removed":
                del altered["files"][existing]
            else:
                altered["files"]["scripts/new_baseline_entry.sh"] = "0" * 64
            altered["sha256"] = digest(altered["files"])
            with self.subTest(change=change), patch.object(provenance, "baseline_protected_identity", return_value=altered):
                with self.assertRaisesRegex(ValueError, "protected baseline inventory changed"):
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

    def test_scanner_selects_namespace_scripts_and_hashes_raw_bytes(self):
        with tempfile.TemporaryDirectory(prefix="ch3-baseline-hash-scan-") as temporary:
            root = Path(temporary)
            inputs = {
                "tools/ch3_baselines/example.py": b"# source\n",
                "configs/chapter3/baselines/example.json": b"{}\n",
                "scripts/dot.sh": b"python -m tools.ch3_baselines.run_baseline\n",
                "scripts/slash.ps1": b"python tools/ch3_baselines/run_baseline.py\n",
                "scripts/backslash.bat": b"python tools\\ch3_baselines\\run_baseline.py\r\n",
            }
            for name, content in {**inputs, "scripts/unrelated.py": b"print('unrelated')\n"}.items():
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            current = provenance.baseline_protected_identity(root)
            self.assertEqual(current["files"], {name: hashlib.sha256(content).hexdigest() for name, content in inputs.items()})
            (root/"scripts/backslash.bat").write_bytes(inputs["scripts/backslash.bat"].replace(b"\r\n", b"\n"))
            self.assertNotEqual(current, provenance.baseline_protected_identity(root))


if __name__ == "__main__":
    unittest.main()
