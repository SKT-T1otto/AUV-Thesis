"""Explicit shell opt-in and an exact frozen legacy exception; no training."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
VARIABLE = "KMP_DUPLICATE_LIB_OK"
LEGACY_PATH = "core/runtime/engine.py"
LEGACY_SHA256 = "1998051dc1651ea9413a4e18124d40dce0e56cc58aeb5e345c4a552a46e698cc"
LEGACY_STATEMENT = 'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"'
RECORD = ROOT / "docs/openmp_runtime_compatibility.json"
START = "OpenMP runtime compatibility (explicit user opt-in only)."
END = "End OpenMP runtime compatibility."
BASELINE_ENTRIES = {
    "run_ch3_basic_prior_eval": ("tools.ch3_baselines.evaluate", []),
    "run_ch3_baseline_eval": ("tools.ch3_baselines.run_baseline", []),
    "train_ch3_direct_mc": ("tools.ch3_baselines.run_training", ["--baseline", "B2_direct_mc"]),
    "train_ch3_direct_boundary": ("tools.ch3_baselines.run_training", ["--baseline", "B3_direct_boundary"]),
}


def git(*args):
    return subprocess.check_output(["git", "-c", "safe.directory=" + ROOT.as_posix(), *args], cwd=ROOT)


def bash_executable():
    git_path = shutil.which("git")
    if os.name == "nt" and git_path:
        candidate = Path(git_path).resolve().parents[1] / "bin/bash.exe"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("bash")


def reporting_block(source):
    lines = source.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if START in line)
    end = next(i for i, line in enumerate(lines) if END in line)
    return "".join(lines[start:end + 1])


def python_openmp_writes(source):
    """Detect literal/constant-key writes and mutators, not example strings.

    This regression guard does not claim to detect arbitrary computed Python.
    """
    tree = ast.parse(source)
    key_names = set()
    def is_key(node):
        return ((isinstance(node, ast.Constant) and node.value == VARIABLE)
                or (isinstance(node, ast.Name) and node.id in key_names))
    def has_key(node):
        return any(is_key(item) for item in ast.walk(node))
    for _ in range(3):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and is_key(node.value):
                key_names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    writes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Subscript) and has_key(target.slice) for target in targets):
                writes.append((node.lineno, ast.dump(node, include_attributes=False)))
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in ("setdefault", "update", "__setitem__", "putenv") and has_key(node):
                writes.append((node.lineno, ast.dump(node, include_attributes=False)))
    return sorted(writes)


def check_python_file(name, data):
    writes = python_openmp_writes(data.decode("utf-8-sig"))
    if name == LEGACY_PATH:
        if hashlib.sha256(data).hexdigest() != LEGACY_SHA256:
            raise ValueError("historical engine bytes changed")
        expected = python_openmp_writes("\n" * 17 + LEGACY_STATEMENT)
        if writes != expected or data.decode("utf-8").splitlines()[17] != LEGACY_STATEMENT:
            raise ValueError("historical assignment changed")
    elif writes:
        raise ValueError(f"new Python OpenMP assignment: {name}:{writes[0][0]}")


def probe_import(entry):
    paths = []
    def audit(event, args):
        if event == "import" and args[0] == "core.runtime.engine":
            paths.append([{"file": Path(frame.filename).relative_to(ROOT).as_posix(),
                           "line": frame.lineno, "code": frame.line}
                          for frame in traceback.extract_stack()
                          if Path(frame.filename).is_absolute() and ROOT in Path(frame.filename).parents])
    sys.addaudithook(audit)
    before = os.getenv(VARIABLE)
    importlib.import_module(entry)
    return dict(entry=entry, before=before, after=os.getenv(VARIABLE),
                legacy_engine_imported="core.runtime.engine" in sys.modules, import_paths=paths)


def run_import_probe(entry, initial):
    args = [sys.executable, "-B", "-m", "tests.test_openmp_runtime_env", "--probe-import", entry]
    # Probe values are set only in child shells; Python never assigns the environment key.
    if os.name == "nt":
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if not shell:
            raise unittest.SkipTest("PowerShell is unavailable")
        setting = ("Remove-Item Env:" + VARIABLE + " -ErrorAction SilentlyContinue" if initial is None
                   else f"$env:{VARIABLE} = '{initial}'")
        command = setting + "\n& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in args)
        process = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command + "\nexit $LASTEXITCODE"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=90)
    else:
        setting = "unset " + VARIABLE if initial is None else "export " + VARIABLE + "=" + shlex.quote(initial)
        process = subprocess.run(["bash", "-c", setting + "\nexec " + shlex.join(args)],
                                 cwd=ROOT, capture_output=True, text=True, timeout=90)
    if process.returncode:
        raise AssertionError(f"import probe failed ({entry}, {initial!r}): {process.stderr}")
    return json.loads(process.stdout.splitlines()[-1])


class OpenMPRuntimeEnvTests(unittest.TestCase):
    def assert_record(self, stderr, value, warning):
        self.assertIn("OPENMP_RUNTIME_ENV", stderr)
        self.assertIn(VARIABLE + "=" + value, stderr)
        self.assertEqual("WARNING:" in stderr, warning)
        if warning:
            self.assertIn("silently produce incorrect results", stderr)
            self.assertIn("Numerical reliability is not guaranteed", stderr)

    def test_all_linux_python_launchers_report_before_python_without_assignment(self):
        found = set()
        for path in (ROOT / "scripts/linux").rglob("*.sh"):
            source = path.read_text(encoding="utf-8")
            launch = re.search(r'(?m)^(?:exec\s+)?(?:python\b|"\$audit_python")|^ARGS=\(python\b', source)
            if launch is None:
                continue
            found.add(path.stem)
            with self.subTest(script=path.name):
                self.assertEqual(source.count(START), 1)
                self.assertLess(source.index(END), launch.start())
                self.assertNotRegex(source, r'(?m)^\s*(?:export\s+)?KMP_DUPLICATE_LIB_OK=')
                self.assertNotIn("KMP_DUPLICATE_LIB_OK:-TRUE", source)
        self.assertTrue(set(BASELINE_ENTRIES).issubset(found))
        self.assertTrue({"run_ch3_learning", "env_preflight", "run_search_value_d1_audit", "run_search_value_d2_audit"}.issubset(found))

    def test_windows_launchers_report_without_setting_environment(self):
        for name in BASELINE_ENTRIES:
            source = (ROOT / "scripts" / (name + ".bat")).read_text(encoding="utf-8")
            with self.subTest(script=name):
                self.assertLess(source.index("setlocal"), source.index(START))
                self.assertLess(source.index(END), source.index("python -B"))
                self.assertNotRegex(source, r'(?im)\bset\s+"?KMP_DUPLICATE_LIB_OK=')
        source = (ROOT / "scripts/run_ch3_learning.ps1").read_text(encoding="utf-8")
        self.assertLess(source.index(END), source.index("& $CondaExecutable"))
        self.assertNotRegex(source, r'\$env:KMP_DUPLICATE_LIB_OK\s*=')

    def test_production_inventory_matches_unchanged_frozen_hashes(self):
        pin = json.loads((ROOT / "docs/chapter3/baselines/frozen_production_source.json").read_text(encoding="utf-8"))["production"]
        actual = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for package in ("core", "chapter3_bser") for p in (ROOT / package).rglob("*.py")}
        self.assertEqual(actual, pin["files"])
        self.assertEqual(hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest(), pin["sha256"])
        self.assertEqual(pin["sha256"], "7a8dda612fb68c0a63bcc38b04ee0973ac80488bddd6c9fffe1fbf9b2f23f491")

    def test_exact_historical_assignment_matches_record_and_frozen_commit(self):
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        historical = record["historical_assignment"]
        self.assertIs(record["historical_openmp_assignment_retained"], True)
        self.assertEqual((historical["path"], historical["line"], historical["statement"], historical["file_sha256"]),
                         (LEGACY_PATH, 18, LEGACY_STATEMENT, LEGACY_SHA256))
        pin = json.loads((ROOT / historical["frozen_manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(pin["frozen_commit"], historical["frozen_commit"])
        self.assertEqual(pin["production"]["files"][LEGACY_PATH], LEGACY_SHA256)
        data = (ROOT / LEGACY_PATH).read_bytes()
        self.assertEqual(data, git("show", historical["frozen_commit"] + ":" + LEGACY_PATH))
        check_python_file(LEGACY_PATH, data)

    def test_no_new_python_assignments_including_tests_and_all_core_files(self):
        names = set(git("ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py").decode().splitlines())
        for directory in ("core", "chapter3_bser", "tools", "scripts"):
            names.update(p.relative_to(ROOT).as_posix() for p in (ROOT / directory).rglob("*.py"))
        self.assertIn(LEGACY_PATH, names)
        self.assertIn("tests/test_openmp_runtime_env.py", names)
        for name in sorted(names):
            with self.subTest(source=name):
                check_python_file(name, (ROOT / name).read_bytes())

    def test_new_writes_and_any_legacy_file_change_are_rejected(self):
        variants = [LEGACY_STATEMENT,
                    'from os import environ as env\nenv["KMP_DUPLICATE_LIB_OK"] = "TRUE"',
                    'os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")',
                    'os.environ.update({"KMP_DUPLICATE_LIB_OK": "TRUE"})',
                    'os.putenv("KMP_DUPLICATE_LIB_OK", "TRUE")',
                    'KEY = "KMP_DUPLICATE_LIB_OK"\nos.environ[KEY] = "TRUE"']
        for name in ("core/new_runtime.py", "chapter3_bser/new_model.py", "tools/new_entry.py", "tests/new_test.py"):
            for source in variants:
                with self.subTest(path=name, source=source), self.assertRaisesRegex(ValueError, "new Python"):
                    check_python_file(name, source.encode())
        original = (ROOT / LEGACY_PATH).read_bytes()
        for data in (original + b"\n# changed\n", original.replace(b'"TRUE"', b'"FALSE"', 1),
                     original + b"\n" + LEGACY_STATEMENT.encode()):
            with self.assertRaisesRegex(ValueError, "historical engine bytes changed"):
                check_python_file(LEGACY_PATH, data)
        self.assertEqual(python_openmp_writes('example = ' + repr(LEGACY_STATEMENT)), [])

    def test_b0_launcher_evolution_only_added_reporting(self):
        manifest = json.loads((ROOT / "docs/chapter3/baselines/framework_protected_b0.json").read_text(encoding="utf-8"))
        previous = manifest["openmp_runtime_evolution"]["previous_launcher_hashes"]
        self.assertEqual(set(previous), {"scripts/run_ch3_basic_prior_eval.bat", "scripts/linux/run_ch3_basic_prior_eval.sh"})
        for name, old_hash in previous.items():
            data = (ROOT / name).read_bytes()
            if name.endswith(".bat"):
                # Windows EOL conversion is allowed. Check the launcher commands
                # separately from Linux runtime provenance, without a byte pin.
                self.assertNotIn(name, manifest["files"])
                source = data.decode().replace("\r\n", "\n")
                old = git("show", manifest["openmp_runtime_evolution"]["reference_commit"] + ":" + name)
                self.assertEqual(source.replace(reporting_block(source), "", 1),
                                 old.decode().replace("\r\n", "\n"))
                continue
            block = reporting_block(data.decode()).encode()
            self.assertEqual(hashlib.sha256(data.replace(block, b"", 1)).hexdigest(), old_hash)
            self.assertEqual(hashlib.sha256(data).hexdigest(), manifest["files"][name])

    def test_linux_syntax_and_baseline_launchers_default_off_and_explicit_opt_in(self):
        bash = bash_executable()
        if not bash:
            self.skipTest("Bash is unavailable")
        for script in (ROOT / "scripts/linux").rglob("*.sh"):
            process = subprocess.run([bash, "-n", script.as_posix()], capture_output=True, text=True, timeout=15)
            self.assertEqual(process.returncode, 0, process.stderr)
        with tempfile.TemporaryDirectory(prefix="openmp-shell-probe-") as temporary:
            directory = Path(temporary)
            fake = directory / "python"
            fake.write_bytes(b'#!/bin/sh\nprintf "OMP=%s\\n" "${KMP_DUPLICATE_LIB_OK-<unset>}"\nprintf "%s\\n" "$@"\nexit 37\n')
            fake.chmod(0o755)
            for initial, value, warning in ((None, "<unset>", False), ("", "", False), ("FALSE", "FALSE", False),
                                             ("TRUE", "TRUE", True), ("true", "true", True), ("1", "1", True)):
                for name, (module, suffix) in BASELINE_ENTRIES.items():
                    setting = "unset " + VARIABLE if initial is None else "export " + VARIABLE + "=" + shlex.quote(initial)
                    command = setting + '\nexport PATH="$(cd -- "$1" && pwd):/usr/bin:/bin:$PATH"\nexec "$BASH" "$2" --output-dir "path with spaces" --seed 17'
                    process = subprocess.run([bash, "-c", command, "probe", directory.as_posix(),
                                              (ROOT / "scripts/linux" / (name + ".sh")).as_posix()],
                                             capture_output=True, text=True, timeout=15)
                    with self.subTest(script=name, initial=initial):
                        self.assertEqual(process.returncode, 37, process.stderr)
                        self.assertEqual(process.stdout.splitlines(),
                                         ["OMP=" + value, "-B", "-m", module, "--output-dir", "path with spaces", "--seed", "17"] + suffix)
                        self.assert_record(process.stderr, value, warning)

    @unittest.skipUnless(os.name == "nt", "Windows batch shell required")
    def test_batch_reporting_default_off_and_explicit_opt_in(self):
        for name in BASELINE_ENTRIES:
            block = reporting_block((ROOT / "scripts" / (name + ".bat")).read_text())
            for initial in ("", "FALSE", "TRUE", "true", "1"):
                with tempfile.TemporaryDirectory(prefix="openmp-batch-probe-") as temporary:
                    probe = Path(temporary) / "probe.cmd"
                    source = (f'@echo off\nsetlocal\nset "{VARIABLE}={initial}"\n' + block
                              + f'if defined {VARIABLE} (set {VARIABLE}) else (echo {VARIABLE}=^<unset^>)\nexit /b 0\n')
                    probe.write_bytes(source.replace("\n", "\r\n").encode())
                    process = subprocess.run(["cmd.exe", "/d", "/c", str(probe)], capture_output=True, text=True, timeout=15)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    value = initial or "<unset>"
                    self.assertEqual(process.stdout.strip(), VARIABLE + "=" + value)
                    self.assert_record(process.stderr, value, initial not in ("", "FALSE"))

    def test_powershell_reporting_default_off_and_explicit_opt_in(self):
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if not shell:
            self.skipTest("PowerShell is unavailable")
        block = reporting_block((ROOT / "scripts/run_ch3_learning.ps1").read_text())
        for initial in ("", "FALSE", "TRUE", "true", "1"):
            command = f"$env:{VARIABLE} = '{initial}'\n" + block + f"\nWrite-Output ('VALUE=' + $env:{VARIABLE})"
            process = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command],
                                     capture_output=True, text=True, timeout=15)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(process.stdout.strip(), "VALUE=" + initial)
            self.assert_record(process.stderr, initial if initial else "<unset>", initial not in ("", "FALSE"))

    def test_b0_b1_and_hgr_entry_imports_preserve_environment_without_old_engine(self):
        for entry in ("tools.ch3_baselines.evaluate", "tools.ch3_baselines.run_baseline", "chapter3_bser.experiments.hgr.cli"):
            for initial in (None, "FALSE", "TRUE"):
                with self.subTest(entry=entry, initial=initial):
                    result = run_import_probe(entry, initial)
                    self.assertEqual((result["before"], result["after"]), (initial, initial))
                    self.assertIs(result["legacy_engine_imported"], False)
                    self.assertEqual(result["import_paths"], [])
                    print("OPENMP_IMPORT_PROBE " + json.dumps(result), flush=True)

    def test_legacy_entry_positive_control_reports_actual_override_path(self):
        result = run_import_probe("core.runtime.training", "FALSE")
        self.assertEqual((result["before"], result["after"]), ("FALSE", "TRUE"))
        self.assertIs(result["legacy_engine_imported"], True)
        files = [frame["file"] for path in result["import_paths"] for frame in path]
        self.assertIn("core/runtime/__init__.py", files)
        self.assertIn("core/runtime/builder.py", files)
        print("OPENMP_IMPORT_PROBE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--probe-import":
        print(json.dumps(probe_import(sys.argv[2])))
    else:
        unittest.main()
