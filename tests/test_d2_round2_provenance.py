"""Second-round scope is exact; every historical predecessor still validates."""
import copy
import json
import unittest

from tests.test_safe_search_provenance import checked_copy
from chapter3_bser.experiments.d2_performance import round2_provenance as gate


class RoundTwoProvenanceTests(unittest.TestCase):
    def test_each_path_review_and_parent_is_protected(self):
        with checked_copy() as root:
            expected = gate.old.framework_sources(root)
            for name in gate.ADDED | gate.CHANGED | {gate.REVIEW, gate.PARENT}:
                path = root / name
                data = path.read_bytes()
                try:
                    path.write_bytes(data + b"\n")
                    with self.subTest(path=name), self.assertRaises(ValueError):
                        gate.old.framework_sources(root)
                finally:
                    path.write_bytes(data)
            self.assertEqual(gate.old.framework_sources(root), expected)

    def test_scope_before_hashes_and_unused_profiles_cannot_be_resealed(self):
        with checked_copy() as root:
            path = root / gate.MANIFEST
            original = json.loads(path.read_text())
            for kind in ("scope", "before", "unused_profile", "parent"):
                value = copy.deepcopy(original)
                if kind == "scope":
                    value["changed_paths"].append("core/env/uav_env.py")
                elif kind == "before":
                    value["profiles"]["git_lf"]["changes"][sorted(gate.CHANGED)[0]]["before"] = "0" * 64
                elif kind == "unused_profile":
                    value["profiles"]["git_clone_preserved"]["sha256"] = "0" * 64
                else:
                    value["parent_manifest_sha256"] = "0" * 64
                value.pop("sha256")
                value["sha256"] = gate.old.digest(value)
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    gate.old.framework_sources(root)


if __name__ == "__main__":
    unittest.main()
