from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class S1LauncherTests(unittest.TestCase):
    def test_launchers_exist_and_linux_contract_is_safe(self):
        paths = (
            ROOT / "scripts/run_phase1c_prrac_s1_train.ps1",
            ROOT / "scripts/run_phase1c_prrac_s1_train.bat",
            ROOT / "scripts/linux/run_phase1c_prrac_s1_train.sh",
            ROOT / "scripts/run_phase1c_prrac_s1_search_diag.ps1",
            ROOT / "scripts/run_phase1c_prrac_s1_search_diag.bat",
            ROOT / "scripts/linux/run_phase1c_prrac_s1_search_diag.sh",
        )
        self.assertTrue(all(path.is_file() for path in paths))
        for path in (paths[2], paths[5]):
            data = path.read_bytes()
            self.assertNotIn(b"\r\n", data)
            text = data.decode("utf-8")
            self.assertIn("set -euo pipefail", text)
            self.assertIn("OMP_NUM_THREADS", text)
            self.assertIn("MKL_NUM_THREADS", text)
        training = paths[2].read_text(encoding="utf-8")
        self.assertIn("--formal", training)
        self.assertIn("--dry-run", training)
        search = paths[5].read_text(encoding="utf-8")
        for option in ("--checkpoint", "--output-dir", "--runtime-origin", "--workers", "--episodes"):
            self.assertIn(option, search)
        self.assertIn("full_prrac searcher_residual_off", search)


if __name__ == "__main__":
    unittest.main()
