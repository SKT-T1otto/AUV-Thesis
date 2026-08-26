import hashlib,json,subprocess
from pathlib import Path
import unittest

class Phase1AV1FrozenTest(unittest.TestCase):
    def test_all_v1_files_match_before_manifest(self):
        root=Path(__file__).resolve().parents[1]; manifest=json.loads((root/"docs/chapter3_bser/phase1a1/phase1a_v1_freeze_before.json").read_text())
        for record in manifest["files"]:
            blob=subprocess.check_output(["git","cat-file","blob",f"HEAD:{record['path']}"],cwd=root)
            self.assertEqual(hashlib.sha256(blob).hexdigest(),record["git_blob_sha256"],record["path"])
