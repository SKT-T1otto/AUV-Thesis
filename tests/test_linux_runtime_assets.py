from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LINUX_SCRIPTS = ROOT / "scripts" / "linux"
AUDIT_LAUNCHERS = {"run_search_value_d1_audit.sh", "run_search_value_d2_audit.sh"}
SOURCED_HELPERS = {"_search_value_audit_common.sh"}
PACKAGERS = {"bundle_search_value_audits.sh"}


class LinuxRuntimeAssetTests(unittest.TestCase):
    def test_bash_syntax_and_all_launchers_forward_arguments_and_exit_codes(self):
        bash = str(Path('E:/git/Git/bin/bash.exe')) if os.name == 'nt' else shutil.which('bash')
        self.assertTrue(bash and Path(bash).is_file(), 'Bash required for launcher acceptance')
        def unix(path):
            value = Path(path).resolve().as_posix()
            return '/'+value[0].lower()+value[2:] if os.name == 'nt' else value
        for path in LINUX_SCRIPTS.glob('*.sh'):
            result = subprocess.run([bash, '-n', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory(prefix='auv-launcher-test-') as directory:
            base = Path(directory); bin_dir = base/'bin'; bin_dir.mkdir()
            profile = base/'etc/profile.d'; profile.mkdir(parents=True)
            (bin_dir/'conda').write_text('#!/usr/bin/env bash\nif [[ $1 == info ]]; then printf "%s\\n" "$STUB_BASE"; fi\n', encoding='utf-8', newline='\n')
            (profile/'conda.sh').write_text('conda(){ printf "%s\\n" "$*" >> "$STUB_ACTIVATION"; }\n', encoding='utf-8', newline='\n')
            (bin_dir/'python').write_text('#!/usr/bin/env bash\nfor arg in "$@"; do if [[ $arg == -m ]]; then printf "%s\\n" "$@" > "$STUB_ARGS"; exit 37; fi; done\nexit 0\n', encoding='utf-8', newline='\n')
            for path in bin_dir.iterdir(): path.chmod(0o755)
            for name in ('checkpoint.pt', 'training.json', 'resolved_evaluation_config.json', 'evaluation_manifest.json', 'episode_evaluation.csv', 'search_value_guidance_metrics.json'):
                (base/name).write_text('test only\n', encoding='utf-8')
            env = {**os.environ, 'STUB_BASE': unix(base), 'STUB_BIN': unix(bin_dir),
                'STUB_ACTIVATION': unix(base/'activation.txt'), 'STUB_ARGS': unix(base/'args.txt'),
                'CRK_CONDA_ENV': 'TEST_ONLY_ENV', 'CRK_CONDA_EXE': unix(bin_dir/'conda'),
                'AUV_AUDIT_PYTHON': unix(bin_dir/'python'), 'AUV_AUDIT_CHECKPOINT': unix(base/'checkpoint.pt'),
                'AUV_AUDIT_TRAINING_CONFIG': unix(base/'training.json'), 'AUV_AUDIT_TRAINING_MANIFEST': '',
                'AUV_AUDIT_OFF_OUTPUT': unix(base), 'AUV_AUDIT_ON_OUTPUT': unix(base),
                'AUV_AUDIT_OUTPUT_DIR': unix(base/'new-test-output')}
            for path in sorted(LINUX_SCRIPTS.glob('*.sh')):
                if path.name in SOURCED_HELPERS or path.name == 'env_preflight.sh': continue
                args = ['smoke'] if path.name in AUDIT_LAUNCHERS else []
                constrained = path.name in {'run_phase1c_prrac_s1_search_diag.sh', 'run_phase1c_prrac_s2a_collision_ablation.sh', 'run_phase1c_prrac_s2a1_local_connector_ablation.sh'}
                forwarded = ['--checkpoint', unix(base/'checkpoint.pt'), '--output-dir', 'two words', '--episodes', '1'] if constrained else ['--test-value', 'two words']
                command = [bash, '-c', 'export PATH="$STUB_BIN:$PATH"; exec bash "$@"', 'launcher-test', unix(path), *args, *forwarded]
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
                with self.subTest(script=path.name):
                    self.assertEqual(result.returncode, 37, result.stderr+result.stdout)
                    received = (base/'args.txt').read_text().splitlines()
                    marker = '--output-dir' if constrained else '--test-value'
                    self.assertEqual(received[received.index(marker)+1], 'two words')
            self.assertIn('activate TEST_ONLY_ENV', (base/'activation.txt').read_text())

    @unittest.skipUnless(os.name == 'nt', 'Windows PowerShell/BAT entry on Windows only')
    def test_windows_collision_launcher_syntax_arguments_and_exit_code(self):
        powershell = shutil.which('powershell')
        self.assertIsNotNone(powershell)
        with tempfile.TemporaryDirectory(prefix='auv-ps-launcher-test-') as directory:
            base = Path(directory); stub = base/'conda.ps1'; captured = base/'args.txt'
            stub.write_text('[IO.File]::WriteAllLines($env:STUB_ARGS, [string[]]$args)\nexit 37\n', encoding='utf-8')
            env = {**os.environ, 'CRK_CONDA_EXE': str(stub), 'CRK_CONDA_ENV': 'TEST_ONLY_ENV', 'STUB_ARGS': str(captured)}
            for name in ('run_collision_terminal.ps1', 'run_collision_terminal.bat'):
                command = ([powershell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT/'scripts'/name)]
                           if name.endswith('.ps1') else [str(ROOT/'scripts'/name)])
                result = subprocess.run([*command, '-CondaEnv', 'TEST_ONLY_ENV', 'evaluate', '--test-value', 'two words'], env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 37, result.stdout+result.stderr)
                args = captured.read_text(encoding='utf-8-sig').splitlines()
                self.assertEqual(args[:4], ['run', '--no-capture-output', '-n', 'TEST_ONLY_ENV'])
                self.assertEqual(args[-3:], ['evaluate', '--test-value', 'two words'])

    def test_linux_shell_assets_exist_use_lf_and_have_strict_entrypoints(self) -> None:
        expected = {
            "run_collision_terminal.sh",
            "env_preflight.sh",
            "run_phase1c_prrac_eval.sh",
            "run_phase1c_prrac_execution_ablation.sh",
            "run_phase1c_prrac_s1_search_diag.sh",
            "run_phase1c_prrac_s1_train.sh",
            "run_phase1c_prrac_s2a_collision_ablation.sh",
            "run_phase1c_prrac_s2a1_local_connector_ablation.sh",
            "run_phase1c_prrac_train.sh",
            "run_phase1c_v2_1_train.sh",
            "run_phase1c_v2_train.sh",
            "run_phase1c_v2_diagnostic_eval.sh",
        } | AUDIT_LAUNCHERS | SOURCED_HELPERS | PACKAGERS
        self.assertEqual({path.name for path in LINUX_SCRIPTS.glob("*.sh")}, expected)
        for name in expected:
            payload = (LINUX_SCRIPTS / name).read_bytes()
            self.assertTrue(
                payload.startswith(b"#!/bin/bash\n")
                or payload.startswith(b"#!/usr/bin/env bash\n"),
                name,
            )
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
            if path.name in PACKAGERS:
                self.assertIn('exec "$audit_python" -m chapter3_bser.experiments.phase1c_prrac.search_value_audit.analysis_bundle "$@"', source)
                continue
            if path.name in SOURCED_HELPERS:
                self.assertIn('AUV_AUDIT_PYTHON:-python', source)
                self.assertIn('command -v "$audit_python"', source)
                continue
            if path.name in AUDIT_LAUNCHERS:
                self.assertIn('_search_value_audit_common.sh', source)
                self.assertIn('exec "$audit_python"', source)
                self.assertIn('"$@"', source)
                continue
            self.assertTrue(
                'CRK_CONDA_ENV="${CRK_CONDA_ENV:-AUV}"' in source
                or '${CRK_CONDA_ENV:-AUV}' in source,
                path.name,
            )
            self.assertTrue(
                'CONDA_BASE="$(conda info --base)"' in source
                or 'CONDA_BASE="$("${COLLISION_CONDA_EXE}" info --base)"' in source
                or '$(conda info --base)/etc/profile.d/conda.sh' in source,
                path.name,
            )
            self.assertTrue(
                'source "${CONDA_SH}"' in source
                or 'source "$(conda info --base)/etc/profile.d/conda.sh"' in source
                or 'source "${CONDA_BASE}/etc/profile.d/conda.sh"' in source,
                path.name,
            )
            self.assertTrue('conda activate "${CRK_CONDA_ENV}"' in source or 'conda activate "${CRK_CONDA_ENV:-AUV}"' in source, path.name)
            self.assertTrue('OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"' in source or 'OMP_NUM_THREADS=1' in source, path.name)
            self.assertTrue('MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"' in source or 'MKL_NUM_THREADS=1' in source, path.name)
            self.assertTrue('MPLBACKEND="${MPLBACKEND:-Agg}"' in source or 'MPLBACKEND=Agg' in source, path.name)

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
            has_fallback = "Get-Command conda" in source and "CRK_CONDA_EXE" in source
            has_direct_conda = "& conda run" in source or "& conda @Arguments" in source
            has_wrapper = "& $Launcher @Arguments" in source
            self.assertTrue(has_fallback or has_direct_conda or has_wrapper, path.name)
            if has_fallback and "$env:PATH" in source:
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
