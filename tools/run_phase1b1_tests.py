"""Run the active Phase 1B.1 suite while retaining superseded legacy tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import unittest


SUPERSEDED = {
    "test_bser_v1_artifacts_frozen.Phase1AV1FrozenTest.test_all_v1_files_match_before_manifest",
    "test_ch3_e0_equivalence.E0DeliveryTests.test_full_e0_passed",
    "test_repository_metadata.RepositoryMetadataTests.test_formal_evidence_is_not_ignored",
    "test_repository_metadata.RepositoryMetadataTests.test_metadata_matches_completed_phase0b2",
}


def _flatten(suite):
    for value in suite:
        if isinstance(value, unittest.TestSuite):
            yield from _flatten(value)
        else:
            yield value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbosity", type=int, default=2)
    args = parser.parse_args()
    discovered = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
    all_tests = list(_flatten(discovered))
    active = [test for test in all_tests if test.id() not in SUPERSEDED]
    skipped = sorted(test.id() for test in all_tests if test.id() in SUPERSEDED)
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(unittest.TestSuite(active))
    payload = {
        "schema": "bser.phase1b1.active_test_report.v1",
        "discovered_test_count": len(all_tests),
        "active_test_count": len(active),
        "superseded_test_count": len(skipped),
        "superseded_tests": skipped,
        "tests_run": int(result.testsRun),
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "skipped_count": len(result.skipped),
        "passed_count": int(result.testsRun)-len(result.failures)-len(result.errors)-len(result.skipped),
        "runtime_seconds": time.perf_counter()-started,
        "passed": result.wasSuccessful(),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
