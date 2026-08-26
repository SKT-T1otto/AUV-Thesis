import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CleanCloneStandaloneTests(unittest.TestCase):
    def test_core_imports_in_isolated_minimal_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            isolated = Path(temporary) / "repo"
            isolated.mkdir()
            shutil.copytree(ROOT / "core", isolated / "core")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(isolated)
            completed = subprocess.run(
                [sys.executable, "-B", "-c", "from core.env import MissionCoreEnv; from core.scenarios.generator import build_e0_manifests; assert MissionCoreEnv.implementation_source == 'core'; assert len(build_e0_manifests()) == 4"],
                cwd=isolated, env=environment, capture_output=True, text=True, timeout=180,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
