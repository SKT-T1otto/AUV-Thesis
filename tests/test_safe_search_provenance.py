"""Real-file inventory corruption checks; no simulator or policy construction."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from chapter3_bser.experiments.safe_search_v1 import provenance as gate


@contextmanager
def checked_copy():
    with tempfile.TemporaryDirectory(prefix="safe-search-source-") as temporary:
        root = Path(temporary)
        names = set(gate.inventory()["files"]) | {gate.MANIFEST}
        # The separately sealed successor is part of the current source gate;
        # historical manifests remain byte-for-byte frozen.
        successor = "docs/provenance/safe_search_v2_evolution.json"
        if (gate.ROOT / successor).exists():
            names.update((successor, "docs/chapter3/search_diagnostics/safe_search_v2/source_review.md"))
        effect = "docs/provenance/bser_effect_v1_evolution.json"
        if (gate.ROOT / effect).exists():
            names.update((effect, "docs/chapter3/search_diagnostics/bser_effect_v1/source_review.md"))
        final = "docs/provenance/bser_final_v1_evolution.json"
        if (gate.ROOT / final).exists():
            names.update((final, "docs/chapter3/search_diagnostics/bser_final_v1/source_review.md"))
        d2 = "docs/provenance/d2_v1_evolution.json"
        if (gate.ROOT / d2).exists():
            from chapter3_bser.experiments.d2_v1.provenance import REFERENCE, REVIEW
            names.update((d2, REFERENCE, REVIEW))
            names.update(gate.read_json(gate.ROOT / REFERENCE)["frozen_files"])
        suite = "docs/provenance/d2_suite_v1_evolution.json"
        if (gate.ROOT / suite).exists():
            names.update((suite, "docs/chapter3/d2_suite_v1/source_review.md"))
        linux = "docs/provenance/d2_suite_linux_v1_evolution.json"
        if (gate.ROOT / linux).exists():
            names.update((linux, "docs/chapter3/d2_suite_v1/linux/source_review.md"))
        evaluation = "docs/provenance/d2_suite_linux_evaluation_v1_evolution.json"
        if (gate.ROOT / evaluation).exists():
            names.update((evaluation, "docs/chapter3/d2_suite_v1/linux_evaluation/source_review.md"))
        pipeline = "docs/provenance/d2_suite_pipeline_v1_evolution.json"
        if (gate.ROOT / pipeline).exists():
            names.update((pipeline, "docs/chapter3/d2_suite_v1/pipeline/source_review.md"))
        for name in names:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((gate.ROOT / name).read_bytes())
        yield root


def write_manifest(root, value):
    value.pop("sha256", None)
    value["sha256"] = gate.digest(value)
    (root / gate.MANIFEST).write_text(json.dumps(value), encoding="utf-8")


class SafeSearchProvenanceTests(unittest.TestCase):
    def test_current_exact_inventory_and_no_source_normalization(self):
        result = gate.framework_sources()
        self.assertEqual(result["inventory"], gate.inventory())
        self.assertEqual(result["historical_record_count"], 27)
        self.assertIn("scripts/search_diagnostic_observer.py", result["inventory"]["files"])
        self.assertIn("scripts/linux/run_hgr_phase1_acceptance.sh", result["inventory"]["files"])
        gate.verify_sources(result)

    def test_changed_removed_added_files_rejected_in_every_scope(self):
        samples = ("core/runtime/engine.py", "chapter3_bser/online/allocator.py",
                   "chapter3_bser/experiments/safe_search_v1/provenance.py",
                   "tools/ch3_baselines/basic_search_prior.py",
                   "configs/chapter3/baselines/search_prior_eval.json",
                   "scripts/search_diagnostic_observer.py", "scripts/linux/run_ch3_baseline_eval.sh")
        with checked_copy() as root:
            before = gate.framework_sources(root)
            for name in samples:
                path = root / name
                original = path.read_bytes()
                added = path.with_name("unreviewed_new" + path.suffix)
                for mode in ("changed", "removed", "added"):
                    with self.subTest(name=name, mode=mode):
                        try:
                            if mode == "changed":
                                path.write_bytes(original + b"\n")
                            elif mode == "removed":
                                path.unlink()
                            else:
                                added.write_bytes(original)
                            with self.assertRaises((ValueError, FileNotFoundError)):
                                gate.framework_sources(root)
                        finally:
                            path.write_bytes(original)
                            added.unlink(missing_ok=True)
            self.assertEqual(before, gate.framework_sources(root))

    def test_all_complete_reviewed_byte_profiles_and_mixed_profile_rejection(self):
        with checked_copy() as root:
            manifest = gate.read_json(root / gate.MANIFEST)
            for profile, record in manifest["profiles"].items():
                expected_files = record["files"]
                if (root / "docs/provenance/safe_search_v2_evolution.json").exists():
                    from chapter3_bser.experiments.safe_search_v2.provenance import extend_profile
                    expected_files = extend_profile(root, profile, expected_files)
                if (root / "docs/provenance/bser_effect_v1_evolution.json").exists():
                    from chapter3_bser.experiments.bser_effect_v1.provenance import extend_profile as extend_effect
                    expected_files = extend_effect(root, profile, expected_files)
                if (root / "docs/provenance/bser_final_v1_evolution.json").exists():
                    from chapter3_bser.experiments.bser_final_v1.provenance import extend_profile as extend_final
                    expected_files = extend_final(root, profile, expected_files)
                if (root / "docs/provenance/d2_v1_evolution.json").exists():
                    from chapter3_bser.experiments.d2_v1.provenance import extend_profile as extend_d2
                    expected_files = extend_d2(root, profile, expected_files)
                if (root / "docs/provenance/d2_suite_v1_evolution.json").exists():
                    from chapter3_bser.experiments.d2_suite_v1.provenance import extend_profile as extend_suite
                    expected_files = extend_suite(root, profile, expected_files)
                if (root / "docs/provenance/d2_suite_linux_v1_evolution.json").exists():
                    from chapter3_bser.experiments.d2_suite_v1.linux_provenance import extend_profile as extend_linux
                    expected_files = extend_linux(root, profile, expected_files)
                if (root / "docs/provenance/d2_suite_linux_evaluation_v1_evolution.json").exists():
                    from chapter3_bser.experiments.d2_suite_v1.evaluation_provenance import extend_profile as extend_evaluation
                    expected_files = extend_evaluation(root, profile, expected_files)
                if (root / "docs/provenance/d2_suite_pipeline_v1_evolution.json").exists():
                    from chapter3_bser.experiments.d2_suite_v1.pipeline_provenance import extend_profile as extend_pipeline
                    expected_files = extend_pipeline(root, profile, expected_files)
                for name, expected in expected_files.items():
                    path = root / name
                    original = (gate.ROOT / name).read_bytes()
                    alternatives = (original, original.replace(b"\r\n", b"\n"))
                    selected = next((data for data in alternatives if hashlib.sha256(data).hexdigest() == expected), None)
                    self.assertIsNotNone(selected, (profile, name))
                    path.write_bytes(selected)
                self.assertEqual(gate.framework_sources(root)["checkout_profile"], profile)
            # An arbitrary partial conversion never becomes an allowed profile.
            mixed = root / "chapter3_bser/controllers/action_adapter.py"
            mixed.write_bytes(mixed.read_bytes().replace(b"\r\n", b"\n"))
            with self.assertRaisesRegex(ValueError, "source inventory mismatch"):
                gate.framework_sources(root)

    def test_historical_manifests_and_all_27_records_are_frozen(self):
        migration = gate.read_json(gate.ROOT / gate.MIGRATION)
        self.assertEqual(len(migration["records"]), 27)
        self.assertEqual(len({r["new_core_path"] for r in migration["records"]}), 27)
        with checked_copy() as root:
            for name in gate.FROZEN_RAW:
                path = root / name
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b" ")
                    with self.assertRaisesRegex(ValueError, "frozen historical manifest changed"):
                        gate.framework_sources(root)
                finally:
                    path.write_bytes(original)

    def test_manifest_hash_scope_before_records_and_unused_profiles_checked(self):
        with checked_copy() as root:
            saved = gate.read_json(root / gate.MANIFEST)
            saved_bytes = (root / gate.MANIFEST).read_bytes()
            cases = []
            changed = copy.deepcopy(saved)
            changed["changed_paths"].append("core/runtime/engine.py")
            cases.append(changed)
            changed = copy.deepcopy(saved)
            changed["profiles"]["git_lf"]["safe_changes"]["chapter3_bser/experiments/hgr/runtime.py"]["before"] = "0" * 64
            cases.append(changed)
            changed = copy.deepcopy(saved)
            changed["profiles"]["git_clone_preserved"]["sha256"] = "0" * 64
            cases.append(changed)
            changed = copy.deepcopy(saved)
            changed["review_status"] = "candidate_requires_review"
            cases.append(changed)
            for value in cases:
                write_manifest(root, value)
                with self.assertRaises(ValueError):
                    gate.framework_sources(root)
            # The successor binds exact historical bytes, not just parsed JSON.
            (root / gate.MANIFEST).write_bytes(saved_bytes)
            gate.framework_sources(root)

    def test_d2_mainline_uses_exact_successor_without_rewriting_old_gate(self):
        from tools.ch3_baselines.framework_provenance import framework_sources
        from tools.ch3_baselines.provenance import production_sources
        # The historical production-only gate still rejects later evolution.
        with self.assertRaisesRegex(ValueError, "production source mismatch"):
            production_sources()
        self.assertIn("d2_evolution_sha256", framework_sources())
        before = gate.framework_sources()
        with checked_copy() as root:
            path = root / "scripts/search_diagnostic_observer.py"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                gate.verify_sources(before, root)


if __name__ == "__main__":
    unittest.main()
