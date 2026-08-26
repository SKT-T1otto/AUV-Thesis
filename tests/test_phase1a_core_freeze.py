import ast
import hashlib
import json
from pathlib import Path
import unittest


PERMITTED_EVOLUTION = {
    "core/registry/experiment_registry.py": {
        "historical_sha256": "769dad9c900af98bc0cb067632d2343db573fcd36c52bd176cba6966351f2b61",
        "current_sha256": "8c735bdbe3e6bff0a56a8e4c120f9e65c236987721ad4ba7ec7b74e01d41a87c",
        "current_ast_dump_sha256": "441dd053e553e88ebfd18e4138af0f6331dfe5a9be3cb3ec029150a964188a36",
    }
}


def _matches_exact_or_one_historical_eof_blank(data: bytes, expected: str) -> bool:
    return expected in {
        hashlib.sha256(data).hexdigest(),
        hashlib.sha256(data + b"\n").hexdigest(),
    }


class Phase1ACoreFreezeTest(unittest.TestCase):
    def test_preexisting_core_python_files_unchanged(self):
        root = Path(__file__).resolve().parents[1]; manifest = json.loads((root / "docs/chapter3_bser/phase1a/core_freeze_before.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["existing_core_python_count"], 40)
        for record in manifest["files"]:
            relative = record["path"]
            data = (root / relative).read_bytes()
            expected_sha = record["sha256"]
            expected_ast = record["ast_dump_sha256"]
            if relative in PERMITTED_EVOLUTION:
                permitted = PERMITTED_EVOLUTION[relative]
                self.assertEqual(expected_sha, permitted["historical_sha256"])
                expected_sha = permitted["current_sha256"]
                expected_ast = permitted["current_ast_dump_sha256"]
            self.assertTrue(
                _matches_exact_or_one_historical_eof_blank(data, expected_sha),
                relative,
            )
            dump = ast.dump(
                ast.parse(data.decode("utf-8")),
                annotate_fields=True,
                include_attributes=True,
            )
            self.assertEqual(hashlib.sha256(dump.encode()).hexdigest(), expected_ast, relative)


if __name__ == "__main__": unittest.main()
