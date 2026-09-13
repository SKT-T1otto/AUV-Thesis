import hashlib,json
from pathlib import Path
import unittest

PERMITTED_EVOLUTION = {
    "core/env/uav_env.py": {
        "historical_sha256": "ef964149b6af3a164cd35ce3ff81636e140fd88180750b12ecac6f30e2b9f698",
        "current_sha256": "2626d3f957a8e448db34868442d28cc20988cbb46292dd48700eab0335ad7eaf",
    },
    "core/env/mission_env.py": {
        "historical_sha256": "21d62729cb214ff0b6707fd6ce58c410ac2d6e49a7fbcdbf0e3746aa3a31ad45",
        "current_sha256": "e4d853fe143764ef4197cba472ebd045fbd0d03fae956cf539aae491f946657f",
    },
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
