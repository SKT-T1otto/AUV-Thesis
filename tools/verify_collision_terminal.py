"""Local regression / frozen E0 verification with isolated new evidence paths."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def golden_profile(profile):
    import torch
    torch.set_num_threads(1)
    from tools.run_core_golden_e0 import _run_profile
    return _run_profile(profile)  # Reads frozen inputs; never calls historical writers.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("regression", "golden"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("verification output must be a new directory")
    args.output_dir.mkdir(parents=True)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    started = time.perf_counter()
    if args.suite == "regression":
        names = [".".join(path.relative_to(ROOT).with_suffix("").parts)
                 for path in sorted((ROOT / "tests").rglob("test_*.py"))]
        suite = unittest.defaultTestLoader.loadTestsFromNames(names)
        with (args.output_dir / "unittest.log").open("w", encoding="utf-8") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        report = {"tests_run": result.testsRun, "failures": len(result.failures),
                  "errors": len(result.errors), "skipped": len(result.skipped),
                  "failed_tests": [str(test) for test, _ in result.failures + result.errors],
                  "passed": result.wasSuccessful()}
    else:
        from tools.run_core_golden_e0 import GOLDEN_PATH, PROFILES
        if not GOLDEN_PATH.is_file():
            report = {"status": "blocked_missing_historical_input", "missing_input": str(GOLDEN_PATH),
                      "trajectories": 0, "passed_trajectories": 0, "passed": False}
        else:
            rows, mismatches = [], []
            with ProcessPoolExecutor(max_workers=4) as executor:
                for profile, result in zip(PROFILES, executor.map(golden_profile, PROFILES)):
                    rows.extend(result["rows"])
                    mismatches.extend(result["mismatches"])
                    print(f"Frozen E0 {profile}: {sum(r['passed'] for r in result['rows'])}/15", flush=True)
            report = {"trajectories": len(rows), "passed_trajectories": sum(r["passed"] for r in rows),
                      "mismatches": mismatches, "passed": len(rows) == 60 and all(r["passed"] for r in rows)}
            (args.output_dir / "trajectories.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report.update(context="latest_recorded_local_verification_not_CI", suite=args.suite,
                  recorded_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-started)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
