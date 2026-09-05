"""Package only an explicit small-artifact allowlist from completed D1/D2 runs."""

import argparse
from pathlib import Path
import zipfile
from .provenance import read_json

D1 = ("audit_manifest.json", "prediction_rows.csv", "prediction_summary.json", "calibration_bins.csv",
      "feature_consistency.csv", "candidate_representation.csv", "training_label_baseline.json", "no_op_validation.json", "progress.json")
D2 = ("branch_manifest.json", "historical_reproduction_check.json", "decision_audit.csv", "candidate_scores.csv",
      "branch_outcomes.csv", "paired_branch_comparison.csv", "branch_summary.json", "replay_validation.json", "progress.json")
OPTIONAL = ("resolved_audit_config.json", "scenario_manifest.json", "run.log", "error.json", "candidate_generation.csv")


def bundle(inputs, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"bundle already exists: {output}")
    entries, missing, total = [], [], 0
    for index, source in enumerate(inputs):
        source = Path(source).resolve(strict=True)
        required = D1 if (source/"audit_manifest.json").is_file() else D2
        missing.extend(str(source/name) for name in required if not (source/name).is_file())
        for name in dict.fromkeys((*required, *OPTIONAL)):
            path = source/name
            if not path.is_file():
                continue
            if path.is_symlink() or path.resolve().parent != source:
                raise ValueError(f"bundle refuses links outside the result directory: {path}")
            size = path.stat().st_size
            if size > 64*1024*1024:
                raise ValueError(f"artifact exceeds 64 MiB compact-bundle limit: {path}")
            total += size
            entries.append((path, f"{index+1}_{source.name}/{name}"))
    if missing:
        raise FileNotFoundError("required analysis artifacts missing:\n" + "\n".join(missing))
    if total > 256*1024*1024:
        raise ValueError("allowlisted artifacts exceed 256 MiB total compact-bundle limit")
    for source in inputs:
        status = read_json(Path(source)/"progress.json")["status"]
        if status not in ("completed", "completed_with_mismatches"):
            raise ValueError(f"incomplete result directory ({status}): {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation; never overwrite an existing artifact or model.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in entries:
            archive.write(path, name)
    return dict(path=str(output), file_count=len(entries), uncompressed_bytes=total,
                excluded="all non-allowlisted files, including .pt, replay, raw_features.npz, full state and traces")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(bundle(args.input_dir, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
