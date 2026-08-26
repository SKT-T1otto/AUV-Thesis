import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _matches_exact_or_one_historical_eof_blank(data: bytes, expected: str) -> bool:
    return expected in {
        hashlib.sha256(data).hexdigest(),
        hashlib.sha256(data + b"\n").hexdigest(),
    }


class PRRACIsolationTests(unittest.TestCase):
    def test_frozen_files_hashes_and_import_direction(self):
        hashes = {
            "core/algorithms/agents.py": "b507fc584689c4da3ce517cf35d423c73a1fa9e7e51470aa61aed35daa3a572d",
            "core/algorithms/maddpg.py": "a574285b33d9a82971cf8016ce93e79f1c90b07cca5236edf5deb7fcb9b6d907",
            "core/algorithms/networks.py": "fe28d54150c245f698418c57b850db44225a8fd65f8119ca38469270d5be54cc",
            "core/replay/ch3_buffer.py": "e22412e43e589962582c2a3ce727b167a6689e60223f6d64fd71ef1e9e5cfa42",
            "core/env/uav_env.py": "ef964149b6af3a164cd35ce3ff81636e140fd88180750b12ecac6f30e2b9f698",
            "core/env/mission_env.py": "21d62729cb214ff0b6707fd6ce58c410ac2d6e49a7fbcdbf0e3746aa3a31ad45",
            "core/env/observation_contract.py": "0bc4d78e09c9228744451bf07fdaaaf5c5eb275a2bb772603656cd4c444dd510",
            "core/registry/experiment_registry.py": "8c735bdbe3e6bff0a56a8e4c120f9e65c236987721ad4ba7ec7b74e01d41a87c",
            "core/config/ch3_config.py": "445ab698c37ac698ca2b291a25427d9f5647cacc979167d1deaa0657aa9c6085",
            "chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/phase_aware_replay.py": "7734f8c7330cdf04137b09e500d85438e23715007225198f6db09fb55adbd26c",
            "chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/training_env.py": "20378574acc6b3acbcba6a9359bc74021b1ff152a728ba5b015f2f13d1544cb2",
        }
        for relative, expected in hashes.items():
            data = (ROOT / relative).read_bytes()
            self.assertTrue(
                _matches_exact_or_one_historical_eof_blank(data, expected),
                relative,
            )
        for path in (ROOT / "core").rglob("*.py"):
            self.assertNotIn("chapter3_bser.models.prrac", path.read_text(encoding="utf-8"), str(path))
        for path in (ROOT / "chapter3_bser/experiments/phase1c_bser_rmaddpg_v2").glob("*.py"):
            self.assertNotIn("prrac", path.read_text(encoding="utf-8").lower(), str(path))

    def test_legacy_configs_and_dimensions_remain_isolated(self):
        for relative, schema in (
            ("configs/chapter3/bser_phase1c_v2_train.json", "bser.phase1c.training.v2"),
            ("configs/chapter3/bser_phase1c_v2_1_train.json", "bser.phase1c.training.v2"),
        ):
            config = json.loads((ROOT / relative).read_text(encoding="utf-8"))
            self.assertEqual(config["schema"], schema)
            self.assertNotIn("architecture", config)
            self.assertEqual((config["observation_dim"], config["action_dim"], config["critic_dim"]), (28, 3, 124))


if __name__ == "__main__":
    unittest.main()
