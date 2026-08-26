import hashlib
import json
import subprocess
import unittest
from pathlib import Path

from core.registry.experiment_registry import (
    ACTIVE_CH3_FINAL_EXPERIMENT_MODES,
    INDEPENDENT_CH3_EXPERIMENT_MODES,
    REGISTERED_CH3_EXPERIMENT_MODES,
    assert_ch3_method,
    assert_registered_ch3_method,
)


ROOT = Path(__file__).resolve().parents[1]

# Phase 0B-2 provenance is a historical baseline and must not be rewritten
# merely because the repository legitimately evolves afterward.
#
# Every provenance target must still match the frozen Phase 0B-2 hash unless
# it is listed here as an explicitly reviewed post-baseline evolution.
#
# For each permitted evolution we pin BOTH:
#   1) the historical Phase 0B-2 hash recorded in the provenance manifest; and
#   2) the reviewed current hash after the legitimate post-baseline change.
#
# A dedicated semantic-contract test below additionally verifies that the
# legacy Chapter-3 active method tuple was not changed when the independent
# Phase 1C method was registered.
PERMITTED_POST_BASELINE_EVOLUTIONS = {
    "core/registry/experiment_registry.py": {
        "phase0b2_sha256": (
            "769dad9c900af98bc0cb067632d2343db573fcd36c52bd176cba6966351f2b61"
        ),
        "current_sha256": (
            "8c735bdbe3e6bff0a56a8e4c120f9e65c236987721ad4ba7ec7b74e01d41a87c"
        ),
        "reason": (
            "Post-Phase-0B-2 registration of the independent "
            "ch3_bser_rmaddpg_phase1c runtime without changing the frozen "
            "legacy Chapter-3 active-method tuple."
        ),
    }
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RepositoryMetadataTests(unittest.TestCase):
    def test_metadata_matches_chapter3_phase1c_wip(self):
        attributes = ROOT / ".gitattributes"
        ignore = ROOT / ".gitignore"
        self.assertTrue(attributes.is_file())
        self.assertTrue(ignore.is_file())

        attributes_text = attributes.read_text(encoding="utf-8")
        for rule in (
            "*.py   text eol=lf",
            "*.json text eol=lf",
            "*.md   text eol=lf",
            "*.txt  text eol=lf",
            "*.csv  text eol=lf",
            "*.yml  text eol=lf",
            "*.yaml text eol=lf",
        ):
            self.assertIn(rule, attributes_text)

        ignore_text = ignore.read_text(encoding="utf-8")
        self.assertIn("outputs/", ignore_text)
        self.assertIn("*.pt", ignore_text)
        self.assertIn("*.pth", ignore_text)
        self.assertIn("*.ckpt", ignore_text)
        self.assertIn("codex_*.diff", ignore_text)
        self.assertNotIn("\n*.diff\n", f"\n{ignore_text}\n")

        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        chapter3_readme = (ROOT / "chapter3_bser/README.md").read_text(
            encoding="utf-8"
        )
        phase1c_status = (
            ROOT / "docs2/phase1c_design/implementation_status.md"
        ).read_text(encoding="utf-8")

        stale_root_claim = "BSER, RCAG, and VSGC are " + "not implemented yet"
        stale_agent_claim = (
            "Phase 1A is the next " + "permitted algorithm phase"
        )
        stale_chapter_claim = (
            "BSER is not implemented as an " + "online controller"
        )
        self.assertNotIn(stale_root_claim, root_readme)
        self.assertNotIn(stale_agent_claim, agents)
        self.assertNotIn(stale_chapter_claim, chapter3_readme)

        for document in (
            root_readme,
            agents,
            chapter3_readme,
            phase1c_status,
        ):
            self.assertIn("Phase 1C", document)
            self.assertNotIn("Phase 1C is complete", document)
            self.assertNotIn("Phase 1C 已完成", document)
            self.assertNotIn("1000 episodes 已完成", document)

        self.assertIn("ch3_bser_rmaddpg_phase1c", root_readme)
        self.assertIn("ch3_bser_rmaddpg_phase1c", chapter3_readme)
        self.assertIn("PRRAC architecture", root_readme)
        self.assertIn("post-rebuild source repair", root_readme)
        self.assertIn("PRRAC dry-run: not run", root_readme)
        self.assertIn("PRRAC 100-episode pilot: not run", root_readme)
        self.assertIn("Chapter 4 RCAG: not begun", root_readme)
        self.assertIn("historical", phase1c_status.lower())
        self.assertIn("not the current experiment", root_readme)
        self.assertIn("episode 128", phase1c_status)
        self.assertIn("episode 100", phase1c_status)
        self.assertIn("not GitHub CI results", root_readme)

        source_path = ROOT / "SOURCE_MANIFEST.json"
        source_text = source_path.read_text(encoding="utf-8")
        source_manifest = json.loads(source_text)
        self.assertEqual(
            source_manifest["schema"],
            "crk.thesis.source_manifest.v2",
        )
        self.assertEqual(
            source_manifest["phase"],
            "chapter3_phase1c_wip",
        )
        self.assertIs(source_manifest["formal_code_migrated"], True)
        self.assertIs(source_manifest["core_self_contained"], True)
        self.assertIs(source_manifest["runtime_legacy_dependency"], False)
        self.assertEqual(
            source_manifest["phase0b2_validation"]["status"],
            "historical_baseline",
        )
        self.assertNotEqual(
            source_manifest["chapter_status"]["chapter3_bser"],
            "not_implemented",
        )

        self.assertIsNone(source_manifest["current_experiment"])
        engineering = source_manifest["current_engineering_status"]
        self.assertEqual(
            engineering["state"],
            "post_rebuild_source_repair_and_verification",
        )
        self.assertIs(engineering["prrac_architecture_implemented"], True)
        self.assertIs(engineering["prrac_dry_run_completed"], False)
        self.assertIs(engineering["prrac_pilot_100_completed"], False)
        self.assertIsNone(engineering["performance_passed"])
        self.assertIs(engineering["chapter4_rcag_begun"], False)

        current = source_manifest["historical_phase1c_experiment"]
        self.assertEqual(
            current["method"],
            "ch3_bser_rmaddpg_phase1c",
        )
        self.assertEqual(
            current["status"],
            "historical_interrupted_previously_resume_ready",
        )
        self.assertEqual(current["last_recorded_episode"], 128)
        self.assertEqual(current["resume_checkpoint_episode"], 100)
        self.assertIs(current["is_current_experiment"], False)
        self.assertIs(current["is_current_resume_target"], False)
        self.assertEqual(
            source_manifest["latest_recorded_verification"]["context"],
            "local_not_github_ci",
        )
        self.assertNotIn(
            r"E:\gym\code\WORKSPACE".lower(),
            source_text.lower(),
        )

    def test_formal_evidence_is_not_ignored(self):
        for relative in (
            "docs/phase0b2/delivery_validation.json",
            "docs/provenance/ch3_to_core_migration_manifest.json",
            "experiments/chapter3/e0_core_migration/core_without_legacy_summary.json",
            "configs/scenarios/e0_equivalence/M20_MOVING_UNKNOWN_MULTI.json",
        ):
            completed = subprocess.run(
                ["git", "check-ignore", "--quiet", relative],
                cwd=ROOT,
                check=False,
            )
            self.assertEqual(completed.returncode, 1, relative)

        checkpoint = subprocess.run(
            [
                "git",
                "check-ignore",
                "--quiet",
                "--no-index",
                "phase0b2_1_probe.pt",
            ],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(checkpoint.returncode, 0)

    def test_phase0b2_provenance_and_permitted_post_baseline_evolution(self):
        manifest = json.loads(
            (
                ROOT
                / "docs/provenance/ch3_to_core_migration_manifest.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["authority_record_count"], 27)
        self.assertEqual(len(manifest["records"]), 27)

        seen_permitted = set()

        for record in manifest["records"]:
            self.assertIs(record["semantic_changes"], False)

            relative = str(record["new_core_path"])
            target = ROOT / relative
            self.assertTrue(target.is_file(), relative)

            historical_sha = str(record["new_core_sha256"])
            current_sha = _sha256(target)

            evolution = PERMITTED_POST_BASELINE_EVOLUTIONS.get(relative)
            if evolution is None:
                self.assertEqual(
                    current_sha,
                    historical_sha,
                    f"unexpected post-Phase-0B-2 drift: {relative}",
                )
                continue

            seen_permitted.add(relative)

            # The historical manifest itself must remain frozen.
            self.assertEqual(
                historical_sha,
                evolution["phase0b2_sha256"],
                f"historical provenance was rewritten: {relative}",
            )

            # The reviewed post-baseline state is pinned independently.
            self.assertNotEqual(
                current_sha,
                historical_sha,
                f"expected reviewed post-baseline evolution is absent: {relative}",
            )
            self.assertEqual(
                current_sha,
                evolution["current_sha256"],
                f"unreviewed drift after permitted evolution: {relative}",
            )

        self.assertEqual(
            seen_permitted,
            set(PERMITTED_POST_BASELINE_EVOLUTIONS),
        )

    def test_phase1c_registry_evolution_preserves_legacy_active_modes(self):
        legacy_active_modes = (
            "ch3_pheromone_prior",
            "ch3_pheromone_rmaddpg",
            "ch3_pse_rmaddpg",
            "ch3_pse_no_belief",
            "ch3_pse_no_exec_cost",
            "ch3_pse_no_standby",
            "ch3_pse_no_residual",
        )
        independent_modes = ("ch3_bser_rmaddpg_phase1c",)

        self.assertEqual(
            ACTIVE_CH3_FINAL_EXPERIMENT_MODES,
            legacy_active_modes,
        )
        self.assertEqual(
            INDEPENDENT_CH3_EXPERIMENT_MODES,
            independent_modes,
        )
        self.assertEqual(
            REGISTERED_CH3_EXPERIMENT_MODES,
            legacy_active_modes + independent_modes,
        )

        self.assertEqual(
            assert_registered_ch3_method("ch3_bser_rmaddpg_phase1c"),
            "ch3_bser_rmaddpg_phase1c",
        )
        with self.assertRaises(ValueError):
            assert_ch3_method("ch3_bser_rmaddpg_phase1c")


if __name__ == "__main__":
    unittest.main()
