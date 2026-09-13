import ast
import hashlib
import json
from pathlib import Path
import unittest


PERMITTED_EVOLUTION = {
    "core/env/uav_env.py": {
        "historical_sha256": "ef964149b6af3a164cd35ce3ff81636e140fd88180750b12ecac6f30e2b9f698",
        "current_sha256": "2626d3f957a8e448db34868442d28cc20988cbb46292dd48700eab0335ad7eaf",
        "current_ast_dump_sha256": "1aa66c607da597a33dd9bb6f129d493b92421d07f1f4ae382b705320e9258282",
    },
    "core/env/mission_env.py": {
        "historical_sha256": "21d62729cb214ff0b6707fd6ce58c410ac2d6e49a7fbcdbf0e3746aa3a31ad45",
        "current_sha256": "e4d853fe143764ef4197cba472ebd045fbd0d03fae956cf539aae491f946657f",
        "current_ast_dump_sha256": "daec82bf738afbae9312b226ecee56acd345998dcb7ce8cffcb4860ddde6e81c",
    },
    "core/registry/experiment_registry.py": {
        "historical_sha256": "769dad9c900af98bc0cb067632d2343db573fcd36c52bd176cba6966351f2b61",
        "current_sha256": "f5cd0ee57ac83ba9c1525bcda43594f4639b19b7c3502c95d963a8726775c94b",
        "current_ast_dump_sha256": "b0a5aa025d5b0dc5e06945ccd5c126fe4d27bf81bd8be340bb41fdb339ed09a6",
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
