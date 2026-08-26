from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LINUX_SCRIPTS = ROOT / "scripts" / "linux"


class LinuxRuntimeAssetTests(unittest.TestCase):
    def test_linux_shell_assets_exist_use_lf_and_have_strict_entrypoints(self) -> None:
        expected = {
            "run_phase1c_v2_train.sh",
            "run_phase1c_v2_diagnostic_eval.sh",
            "env_preflight.sh",
        }
        self.assertEqual({path.name for path in LINUX_SCRIPTS.glob("*.sh")}, expected)
        for name in expected:
            payload = (LINUX_SCRIPTS / name).read_bytes()
            self.assertTrue(payload.startswith(b"#!/bin/bash\n"), name)
            self.assertNotIn(b"\r\n", payload, name)
            self.assertIn(b"set -e", payload, name)

    def test_launchers_call_shared_python_modules(self) -> None:
        train = (LINUX_SCRIPTS / "run_phase1c_v2_train.sh").read_text(encoding="utf-8")
        diagnostic = (LINUX_SCRIPTS / "run_phase1c_v2_diagnostic_eval.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "python -B -m chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.train_phase1c_v2",
            train,
        )
        self.assertIn(
            "python -B -m chapter3_bser.experiments.phase1c_bser_rmaddpg.evaluate_phase1c_checkpoints",
            diagnostic,
        )
        self.assertIn('"$@"', train)
        self.assertIn('"$@"', diagnostic)
        self.assertIn("cuda_requested", diagnostic)

    def test_linux_assets_activate_configured_conda_environment(self) -> None:
        for path in LINUX_SCRIPTS.glob("*.sh"):
            source = path.read_text(encoding="utf-8")
            self.assertIn('CRK_CONDA_ENV="${CRK_CONDA_ENV:-AUV}"', source, path.name)
            self.assertIn('CONDA_BASE="$(conda info --base)"', source, path.name)
            self.assertIn('source "${CONDA_SH}"', source, path.name)
            self.assertIn('conda activate "${CRK_CONDA_ENV}"', source, path.name)
            self.assertIn('OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"', source, path.name)
            self.assertIn('MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"', source, path.name)
            self.assertIn('MPLBACKEND="${MPLBACKEND:-Agg}"', source, path.name)

    def test_preflight_writes_required_json_report(self) -> None:
        source = (LINUX_SCRIPTS / "env_preflight.sh").read_text(encoding="utf-8")
        self.assertIn("outputs/runtime/linux_preflight.json", source)
        self.assertIn("nvidia-smi", source)
        for field in (
            "hostname",
            "python_version",
            "torch_version",
            "cuda_runtime",
            "gpu_name",
            "device_count",
        ):
            self.assertIn(f'"{field}"', source)

    def test_linux_cuda_environment_is_platform_specific_without_windows_packages(self) -> None:
        path = ROOT / "configs" / "environment_lock" / "environment_linux_cuda.yml"
        self.assertTrue(path.is_file())
        source = path.read_text(encoding="utf-8").lower()
        self.assertIn("name: auv", source)
        for required in (
            "linux-64",
            "python=3.10",
            "torch==2.11.0+cu126",
            "torchvision==0.26.0+cu126",
            "torchaudio==2.11.0+cu126",
            "cuda-version=12.6",
        ):
            self.assertIn(required, source)
        for forbidden in ("ucrt", "vs2015_runtime", "prefix:", "d:\\"):
            self.assertNotIn(forbidden, source)

    def test_gitattributes_forces_linux_shell_lf(self) -> None:
        source = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("*.sh   text eol=lf", source)

    def test_powershell_conda_fallback_is_environment_driven(self) -> None:
        for path in (ROOT / "scripts").glob("*.ps1"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("D:\\anaconda", source, path.name)
            self.assertIn("Get-Command conda", source, path.name)
            self.assertIn("CRK_CONDA_EXE", source, path.name)
            self.assertIn("[IO.Path]::PathSeparator", source, path.name)

    def test_all_pyplot_experiment_modules_select_agg_first(self) -> None:
        for path in (ROOT / "chapter3_bser" / "experiments").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "import matplotlib.pyplot" not in source:
                continue
            self.assertIn('matplotlib.use("Agg")', source, path.as_posix())
            self.assertLess(
                source.index('matplotlib.use("Agg")'),
                source.index("import matplotlib.pyplot"),
                path.as_posix(),
            )


if __name__ == "__main__":
    unittest.main()
