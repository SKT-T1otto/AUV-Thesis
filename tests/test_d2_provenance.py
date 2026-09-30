"""Exact D2 source review tests, including frozen historical dependencies."""
import copy
import json
import unittest

from chapter3_bser.experiments.d2_v1 import provenance as d2
from tests.test_safe_search_provenance import checked_copy


class D2ProvenanceTests(unittest.TestCase):
    def test_exact_git_reference_profile_and_mixed_reference_rejection(self):
        with checked_copy() as root:
            frozen = d2.old.read_json(root / d2.REFERENCE)
            changed = [n for n in frozen["frozen_files"]
                       if frozen["frozen_files"][n] != frozen["frozen_git_lf_files"][n]]
            self.assertGreater(len(changed), 1)
            for i, name in enumerate(changed):
                path = root / name
                path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
                if i < len(changed) - 1:
                    with self.assertRaisesRegex(ValueError, "mixed reference profiles"):
                        d2.old.framework_sources(root)
            d2.old.framework_sources(root)

    def test_every_integration_path_and_frozen_reference_is_protected(self):
        with checked_copy() as root:
            before = d2.old.framework_sources(root)
            frozen = d2.old.read_json(root / d2.REFERENCE)
            paths = d2.CHANGED | d2.ADDED | {d2.REVIEW, d2.REFERENCE} | set(frozen["frozen_files"])
            for name in sorted(paths):
                path = root / name
                data = path.read_bytes()
                try:
                    path.write_bytes(data + b"\n")
                    with self.subTest(path=name), self.assertRaises(ValueError):
                        d2.old.framework_sources(root)
                finally:
                    path.write_bytes(data)
            self.assertEqual(before, d2.old.framework_sources(root))

    def test_no_reseal_can_expand_scope_or_change_before_hash(self):
        with checked_copy() as root:
            path = root / d2.MANIFEST
            data = path.read_bytes()
            original = json.loads(data)
            for mode in ("scope", "before", "unused_profile", "reference", "parent"):
                value = copy.deepcopy(original)
                if mode == "scope": value["changed_paths"].append("core/runtime/engine.py")
                elif mode == "before":
                    value["profiles"]["git_lf"]["changes"][sorted(d2.CHANGED)[0]]["before"] = "0" * 64
                elif mode == "unused_profile": value["profiles"]["git_clone_preserved"]["sha256"] = "0" * 64
                elif mode == "reference": value["frozen_reference_sha256"] = "0" * 64
                else: value["parent_manifest_sha256"] = "0" * 64
                value.pop("sha256")
                value["sha256"] = d2.old.digest(value)
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    d2.old.framework_sources(root)
            path.write_bytes(data)
            d2.old.framework_sources(root)


if __name__ == "__main__":
    unittest.main()
