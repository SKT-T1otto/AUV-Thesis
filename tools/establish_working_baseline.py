"""Create the user-accepted Phase 0B working baseline and archive waiver."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
SOURCE = WORKSPACE / "PHASE0A2_ARCHIVE_CHECK/LEGACY_BASELINE_V2_CANDIDATE.json"
RESOLUTION = WORKSPACE / "PHASE0A2_ARCHIVE_CHECK/integrity_resolution.json"
OUTPUT = REPO / "configs/legacy_baseline/WORKING_BASELINE_V2.json"
DECISION = REPO / "docs/decisions/phase0a_archive_waiver.md"


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> int:
    candidate = json.loads(SOURCE.read_text(encoding="utf-8"))
    resolution = json.loads(RESOLUTION.read_text(encoding="utf-8"))
    repositories = candidate["current_repositories"]
    accepted_missing = [
        "CH3/data/chapter3/S_profiles_200ep_analysis_bundle.zip",
        "CH5/docs/code_directory_tree.md",
    ]
    archived_paths = candidate["documented_archive_paths"]
    created = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema": "crk.working_legacy_baseline.v2",
        "status": [
            "USER_ACCEPTED_CURRENT_SNAPSHOT",
            "ARCHIVE_NOT_REVERIFIED",
            "NOT_A_COMPLETE_HISTORICAL_ARCHIVE",
        ],
        "created_at_utc": created,
        "purpose": "Development starting point for Phase 0B; not proof of complete historical retention.",
        "user_decision": {
            "external_archive_verification_waived": True,
            "historical_missing_files_will_not_be_restored": True,
            "historical_data_has_independent_backup": True,
            "current_workspace_accepted_for_new_project_development": True,
        },
        "accepted_missing_paths": accepted_missing,
        "accepted_manually_archived_paths": archived_paths,
        "source_candidate": str(SOURCE.relative_to(WORKSPACE)).replace("\\", "/"),
        "source_candidate_sha256": sha256(SOURCE),
        "phase0a2_resolution": {
            "source": str(RESOLUTION.relative_to(WORKSPACE)).replace("\\", "/"),
            "source_sha256": sha256(RESOLUTION),
            "prior_final_status": resolution["final_status"],
            "prior_legacy_baseline_v2_allowed": resolution["legacy_baseline_v2_allowed"],
        },
        "repository_file_counts": {repo: len(files) for repo, files in repositories.items()},
        "repositories": repositories,
        "disclaimers": [
            "This file does not promote LEGACY_BASELINE_V2_CANDIDATE to an authoritative archive baseline.",
            "External archive contents were not reverified.",
            "This snapshot is intentionally incomplete with respect to historical outputs.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    decision = f"""# Phase 0A archive verification waiver

Recorded: `{created}`

The user accepts the current CH3/CH4/CH5 workspace snapshot as the development starting point for Phase 0B. External historical archive verification is waived; missing historical outputs will not be restored, and the user states that historical data has an independent backup.

This decision does **not** assert that the historical archive was completely verified. `LEGACY_BASELINE_V2_CANDIDATE.json` remains non-authoritative and must not be rewritten as a verified baseline.

Accepted missing paths:

- `CH3/data/chapter3/S_profiles_200ep_analysis_bundle.zip`
- `CH5/docs/code_directory_tree.md`
- All user-confirmed manually archived historical output roots recorded in the working baseline.

Required baseline labels:

- `USER_ACCEPTED_CURRENT_SNAPSHOT`
- `ARCHIVE_NOT_REVERIFIED`
- `NOT_A_COMPLETE_HISTORICAL_ARCHIVE`

This waiver authorizes new development only in `CRK-Thesis-v2`; CH3, CH4, CH5 and all Phase 0A evidence remain read-only.
"""
    DECISION.parent.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(decision, encoding="utf-8")
    print(json.dumps({"working_baseline": str(OUTPUT), "counts": payload["repository_file_counts"], "decision": str(DECISION)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
