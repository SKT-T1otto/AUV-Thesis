"""Only CSV/config fixtures and mocked commands; no simulator or trained model."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from chapter3_bser.experiments.phase1c_prrac import paired_evaluation as pair


class ParsingTests(unittest.TestCase):
    def test_explicit_boolean_and_missing_values(self):
        for text in ("False", "false", "0", " false "):
            self.assertIs(pair.parse_optional_bool(text), False)
        for text in ("True", "true", "1"):
            self.assertIs(pair.parse_optional_bool(text), True)
        for value in (None, "", "  "):
            self.assertIsNone(pair.parse_optional_bool(value))
        with self.assertRaises(ValueError):
            pair.parse_optional_bool("not false")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.csv"
            path.write_text('found,success,found_step,scenario_id\nFalse,False,,0001\n', encoding="utf-8")
            row = pair.read_episode_csv(path)[0]
        self.assertIs(row["found"], False)
        self.assertIsNone(row["found_step"])
        self.assertEqual(row["scenario_id"], "0001")

    def test_conditional_success_uses_found_denominator_and_null(self):
        summary = pair.outcome_summary([{"found": False, "success": False},
            {"found": True, "success": True}, {"found": True, "success": False}])
        self.assertEqual(summary["success_if_found_numerator"], 1)
        self.assertEqual(summary["success_if_found_denominator"], 2)
        self.assertEqual(summary["success_if_found_rate"], .5)
        for rows in ([], [{"found": False, "success": False}], [{"found": None, "success": False}],
                     [{"found": True, "success": None}]):
            self.assertIsNone(pair.outcome_summary(rows)["success_if_found_rate"])


class ManualPairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "user_checkpoint.pt"
        # Opaque bytes for preparation/analysis fixtures only, never loaded.
        self.checkpoint.write_bytes(b"synthetic path fixture; not model weights")

    def prepare(self, episodes=1):
        return pair.prepare_pair(checkpoint=self.checkpoint, episodes=episodes, output_root=self.root)

    def outputs(self):
        root, plan = self.prepare()
        for mode in pair.CONTROLLERS:
            output = root / mode
            output.mkdir()
            config = pair.read_json(plan["shared_config"])
            config.update(controller=mode, output_dir=str(output))
            pair.write_new_json(output / "resolved_evaluation_config.json", config)
            pair.write_new_json(output / "evaluation_manifest.json", {"manifest_sha256": "synthetic-manifest",
                "scenarios": [{"scenario_id": "synthetic", "scenario_seed": 7}]})
            pair.write_new_json(output / "evaluation_summary.json", {"controller_mode": mode})
            row = dict(controller_mode=mode, checkpoint=str(self.checkpoint.resolve()), checkpoint_config_hash="synthetic-analysis-only",
                checkpoint_episode=2, manifest_sha256="synthetic-manifest", scenario_id="synthetic", scenario_seed=7,
                found="False", success="False")
            with (output / "episode_evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
        return root, plan

    def test_prepare_1_and_10_share_config_and_have_new_output_dirs(self):
        roots = []
        with mock.patch.object(pair.subprocess, "run", side_effect=AssertionError("must not execute")):
            for count in (1, 10, 1):
                root, plan = self.prepare(count)
                roots.append(root)
                self.assertEqual(plan["total_planned_episodes"], count * 2)
                config = pair.read_json(plan["shared_config"])
                self.assertEqual(config["checkpoints"], [str(self.checkpoint.resolve())])
                self.assertEqual(config["evaluation_episodes"], count)
                commands = plan["commands"]
                left, right = (commands[mode] for mode in pair.CONTROLLERS)
                self.assertEqual(left[:left.index("--controller")], right[:right.index("--controller")])
                self.assertNotEqual(left[-1], right[-1])
                self.assertFalse((root / "prior_only").exists())
        self.assertEqual(len(set(roots)), 3)

    def test_prepare_only_cli_never_loads_checkpoint_or_launches(self):
        with mock.patch.object(pair, "validate_checkpoint_for_pair", side_effect=AssertionError("no load")), \
             mock.patch.object(pair.subprocess, "run", side_effect=AssertionError("no simulation")):
            self.assertEqual(pair.main(["run", "--checkpoint", str(self.checkpoint), "--episodes", "1",
                                        "--output-root", str(self.root), "--prepare-only"]), 0)

    def test_checkpoint_is_required_and_fixture_rejected_even_if_renamed(self):
        with self.assertRaises(SystemExit):
            pair.main(["run", "--episodes", "1", "--prepare-only"])
        with self.assertRaisesRegex(ValueError, "untrained_test_fixture"):
            pair.reject_fixture("untrained_test_fixture.pt")
        with self.assertRaisesRegex(ValueError, "test/untrained"):
            pair.reject_fixture("renamed.pt", {"config_hash": "test-config-hash", "completed_episode": 12})

    def test_analysis_is_read_only_and_no_found_is_unavailable(self):
        root, _ = self.outputs()
        with mock.patch.object(pair.subprocess, "run", side_effect=AssertionError("no simulation")):
            result = pair.analyze_pair(root)
        self.assertEqual(result["paired_scenarios"], 1)
        self.assertIsNone(result["controllers"]["prior_only"]["success_if_found_rate"])
        self.assertIsNone(result["differences"]["success_if_found_rate"])
        self.assertFalse((root / "paired_analysis.json").exists())

    def test_analysis_rejects_config_or_checkpoint_drift(self):
        root, plan = self.outputs()
        config_file = root / "full_prrac/resolved_evaluation_config.json"
        config = pair.read_json(config_file)
        config["scenario_seed"] += 1
        config_file.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "scenario/configuration mismatch"):
            pair.analyze_pair(root)
        self.checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint bytes changed"):
            pair.analyze_pair(root)

    def test_analysis_rejects_csv_controller_mismatch(self):
        root, _ = self.outputs()
        path = root / "prior_only/episode_evaluation.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("prior_only", "full_prrac"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "episode controller"):
            pair.analyze_pair(root)

    def test_analysis_rejects_matching_but_wrong_scenarios(self):
        root, _ = self.outputs()
        for mode in pair.CONTROLLERS:
            path = root / mode / "episode_evaluation.csv"
            path.write_text(path.read_text(encoding="utf-8").replace(",7,", ",8,"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "do not match manifest"):
            pair.analyze_pair(root)

    def test_failed_first_command_stops_second_without_real_execution(self):
        with mock.patch.object(pair, "validate_checkpoint_for_pair"), \
             mock.patch.object(pair.subprocess, "run", side_effect=pair.subprocess.CalledProcessError(1, "synthetic")) as execute:
            with self.assertRaises(pair.subprocess.CalledProcessError):
                pair.main(["run", "--checkpoint", str(self.checkpoint), "--episodes", "1",
                           "--output-root", str(self.root)])
        self.assertEqual(execute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
