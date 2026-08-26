import hashlib,json
from pathlib import Path
import unittest

PERMITTED_EVOLUTION = {
    "core/registry/experiment_registry.py": {
        "historical_sha256": "769dad9c900af98bc0cb067632d2343db573fcd36c52bd176cba6966351f2b61",
        "current_sha256": "8c735bdbe3e6bff0a56a8e4c120f9e65c236987721ad4ba7ec7b74e01d41a87c",
    }
}

def _matches_exact_or_one_historical_eof_blank(data: bytes, expected: str) -> bool:
    return expected in {
        hashlib.sha256(data).hexdigest(),
        hashlib.sha256(data + b"\n").hexdigest(),
    }

class Phase1A1OriginalCoreFreezeTest(unittest.TestCase):
    def test_original_40_core_python_files_match(self):
        root=Path(__file__).resolve().parents[1]; manifest=json.loads((root/"docs/chapter3_bser/phase1a1/core_freeze_before.json").read_text()); self.assertEqual(len(manifest["files"]),40)
        for record in manifest["files"]:
            relative = record["path"]
            blob = (root / relative).read_bytes()
            expected = record["git_blob_sha256"]
            if relative in PERMITTED_EVOLUTION:
                permitted = PERMITTED_EVOLUTION[relative]
                self.assertEqual(expected, permitted["historical_sha256"])
                expected = permitted["current_sha256"]
            self.assertTrue(
                _matches_exact_or_one_historical_eof_blank(blob, expected),
                relative,
            )
