"""Read-only, resumable verifier for CRK legacy external archives.

The verifier never writes to the workspace sources or archive.  All cache and
reports are confined to --output-dir.  It deliberately refuses to recurse into
symbolic links, junctions, or other reparse points.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterable


FILE_FIELDS = [
    "repository", "source_relative_path", "expected_size", "expected_sha256",
    "archive_absolute_path", "archive_size", "archive_sha256", "mapping_method",
    "exists", "size_match", "sha256_match", "duplicate_count",
    "verification_status", "failure_reason",
]
VALID_FILE_STATUSES = {
    "PASS", "MISSING", "SIZE_MISMATCH", "SHA256_MISMATCH", "AMBIGUOUS_MAPPING",
    "BASELINE_HASH_UNAVAILABLE", "ACCESS_ERROR", "UNEXPECTED_FILE_TYPE", "REVIEW_REQUIRED",
}


def normalized(value: str) -> str:
    return value.replace("\\", "/").strip("/").casefold()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def is_reparse(stat_result: os.stat_result) -> bool:
    attributes = getattr(stat_result, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def scan_archive(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    files: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []

    def visit(directory: Path) -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            blocked.append({"path": str(directory), "reason": f"ACCESS_ERROR: {type(exc).__name__}: {exc}"})
            return
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
                relative = path.relative_to(root).as_posix()
                if entry.is_symlink() or is_reparse(info):
                    blocked.append({"path": str(path), "reason": "REVIEW_REQUIRED_REPARSE_POINT_NOT_FOLLOWED"})
                    continue
                if stat.S_ISDIR(info.st_mode):
                    visit(path)
                elif stat.S_ISREG(info.st_mode):
                    files.append(
                        {
                            "path": path,
                            "absolute": str(path.resolve()),
                            "relative": relative,
                            "normalized_relative": normalized(relative),
                            "size": info.st_size,
                            "mtime_ns": info.st_mtime_ns,
                        }
                    )
                else:
                    blocked.append({"path": str(path), "reason": "UNEXPECTED_FILE_TYPE"})
            except OSError as exc:
                blocked.append({"path": str(path), "reason": f"ACCESS_ERROR: {type(exc).__name__}: {exc}"})

    visit(root)
    return files, blocked


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def expected_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for source in read_manifest(path):
        repository = source.get("repository", "").strip()
        relative = (source.get("source_relative_path") or source.get("relative_path") or "").replace("\\", "/")
        size_text = source.get("expected_size") or source.get("size_bytes") or ""
        digest = (source.get("expected_sha256") or source.get("sha256") or "").strip().lower()
        rows.append(
            {
                "repository": repository,
                "source_relative_path": relative,
                "expected_size": int(size_text) if size_text else None,
                "expected_sha256": digest,
            }
        )
    return rows


class ArchiveIndex:
    def __init__(self, root: Path, files: list[dict[str, Any]], resume_cache: dict[str, Any]):
        self.root = root
        self.files = files
        self.by_relative = {item["normalized_relative"]: item for item in files}
        self.hash_cache: dict[str, str] = {}
        self.resume_cache = resume_cache

    def digest(self, item: dict[str, Any]) -> str:
        key = item["absolute"]
        cached = self.hash_cache.get(key)
        if cached:
            return cached
        previous = self.resume_cache.get(key)
        if previous and previous.get("size") == item["size"] and previous.get("mtime_ns") == item["mtime_ns"] and previous.get("sha256"):
            value = previous["sha256"]
        else:
            value = sha256_file(item["path"])
            self.resume_cache[key] = {"size": item["size"], "mtime_ns": item["mtime_ns"], "sha256": value}
        self.hash_cache[key] = value
        return value

    def map(self, repository: str, relative: str, expected_size: int | None, expected_sha: str) -> tuple[dict[str, Any] | None, str, int, str]:
        repo_key = normalized(f"{repository}/{relative}")
        relative_key = normalized(relative)
        if repo_key in self.by_relative:
            return self.by_relative[repo_key], "exact_repo_relative", 1, "HIGH"
        if relative_key in self.by_relative:
            return self.by_relative[relative_key], "exact_relative", 1, "HIGH"
        candidates = [item for item in self.files if item["normalized_relative"].endswith("/" + relative_key) or item["normalized_relative"] == relative_key]
        if expected_size is not None:
            candidates = [item for item in candidates if item["size"] == expected_size]
        if expected_sha:
            matching = []
            for item in candidates:
                try:
                    if self.digest(item) == expected_sha:
                        matching.append(item)
                except OSError:
                    pass
            candidates = matching
        if len(candidates) == 1:
            return candidates[0], "unique_suffix", 1, "HIGH"
        if len(candidates) > 1:
            return None, "unique_suffix", len(candidates), "LOW"
        return None, "", 0, "LOW"


def verify_one(row: dict[str, Any], index: ArchiveIndex) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result = {field: "" for field in FILE_FIELDS}
    result.update(
        {
            "repository": row["repository"],
            "source_relative_path": row["source_relative_path"],
            "expected_size": "" if row["expected_size"] is None else row["expected_size"],
            "expected_sha256": row["expected_sha256"],
            "exists": "false", "size_match": "false", "sha256_match": "false", "duplicate_count": 0,
        }
    )
    if row["expected_size"] is None or not row["expected_sha256"]:
        result["verification_status"] = "BASELINE_HASH_UNAVAILABLE"
        result["failure_reason"] = "expected size or SHA256 missing"
        return result, None
    try:
        item, method, duplicates, confidence = index.map(
            row["repository"], row["source_relative_path"], row["expected_size"], row["expected_sha256"]
        )
        result["mapping_method"] = method
        result["duplicate_count"] = duplicates
        if item is None:
            if duplicates > 1:
                result["verification_status"] = "AMBIGUOUS_MAPPING"
                result["failure_reason"] = "multiple size-and-SHA matching suffix candidates"
            else:
                result["verification_status"] = "MISSING"
                result["failure_reason"] = "no exact or unique size-and-SHA archive mapping"
            return result, None
        result["archive_absolute_path"] = item["absolute"]
        result["archive_size"] = item["size"]
        result["exists"] = "true"
        result["size_match"] = str(item["size"] == row["expected_size"]).lower()
        if item["size"] != row["expected_size"]:
            result["verification_status"] = "SIZE_MISMATCH"
            result["failure_reason"] = "archive size differs from baseline"
            return result, item
        archive_sha = index.digest(item)
        result["archive_sha256"] = archive_sha
        result["sha256_match"] = str(archive_sha == row["expected_sha256"]).lower()
        if archive_sha != row["expected_sha256"]:
            result["verification_status"] = "SHA256_MISMATCH"
            result["failure_reason"] = "archive SHA256 differs from baseline"
        elif confidence != "HIGH":
            result["verification_status"] = "REVIEW_REQUIRED"
            result["failure_reason"] = "mapping confidence is not HIGH"
        else:
            result["verification_status"] = "PASS"
        return result, item
    except PermissionError as exc:
        result["verification_status"] = "ACCESS_ERROR"
        result["failure_reason"] = f"PermissionError: {exc}"
        return result, None
    except OSError as exc:
        result["verification_status"] = "ACCESS_ERROR"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return result, None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_curated(workspace: Path, manifest_path: Path, file_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapped = {(row["repository"].casefold(), normalized(row["source_relative_path"])): row for row in file_results}
    results = []
    for row in read_manifest(manifest_path):
        repo = row["repository"]
        relative = row["source_relative_path"]
        expected = row["sha256"].lower()
        curated = workspace / Path(row["curated_relative_path"])
        source = workspace / repo / Path(relative)
        status = "PASS"
        reason = ""
        curated_sha = ""
        source_or_archive_sha = ""
        try:
            if not curated.is_file():
                status, reason = "FAIL", "curated copy missing"
            else:
                curated_sha = sha256_file(curated)
                if curated_sha != expected or curated.stat().st_size != int(row["size_bytes"]):
                    status, reason = "FAIL", "curated size or SHA mismatch"
                elif source.is_file():
                    source_or_archive_sha = sha256_file(source)
                    if source_or_archive_sha != curated_sha:
                        status, reason = "FAIL", "workspace source differs from curated"
                else:
                    archive_row = mapped.get((repo.casefold(), normalized(relative)))
                    source_or_archive_sha = archive_row.get("archive_sha256", "") if archive_row else ""
                    if not archive_row or archive_row.get("verification_status") != "PASS" or source_or_archive_sha != curated_sha:
                        status, reason = "FAIL", "source absent and no matching verified archive copy"
            if not row.get("source_snapshot_id") or not row.get("copy_status"):
                status, reason = "FAIL", "manifest provenance/copy status missing"
        except OSError as exc:
            status, reason = "FAIL", f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "repository": repo, "source_relative_path": relative,
                "curated_relative_path": row["curated_relative_path"], "expected_size": row["size_bytes"],
                "expected_sha256": expected, "curated_exists": str(curated.is_file()).lower(),
                "curated_sha256": curated_sha, "source_exists": str(source.is_file()).lower(),
                "source_or_archive_sha256": source_or_archive_sha, "status": status, "failure_reason": reason,
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--file-manifest", required=True, type=Path)
    parser.add_argument("--directory-manifest", required=True, type=Path)
    parser.add_argument("--curated-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace_root.resolve()
    archive = args.archive_root.resolve()
    if not workspace.is_dir() or not args.file_manifest.is_file() or not args.directory_manifest.is_file() or not args.curated_manifest.is_file():
        print("configuration error: workspace or manifest missing", file=sys.stderr)
        return 2
    if not args.archive_root.is_dir():
        print(f"configuration error: archive root missing: {args.archive_root}", file=sys.stderr)
        return 2
    if archive == workspace or is_relative_to(archive, workspace):
        print("configuration error: ARCHIVE_ROOT_INSIDE_WORKSPACE", file=sys.stderr)
        return 2
    workers = max(1, min(4, args.workers))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "archive_verification_progress.json"
    resume_cache: dict[str, Any] = {}
    if args.resume and progress_path.is_file():
        resume_cache = json.loads(progress_path.read_text(encoding="utf-8")).get("hash_cache", {})

    files, blocked = scan_archive(archive)
    index = ArchiveIndex(archive, files, resume_cache)
    rows = expected_rows(args.file_manifest)
    results: list[dict[str, Any]] = []
    mapped_items: set[str] = set()
    # Mapping can require candidate hashes; digest is protected by the GIL-level
    # cache and ordinary files are hashed at most once in normal exact layouts.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(verify_one, row, index): row for row in rows}
        for count, future in enumerate(as_completed(futures), 1):
            result, item = future.result()
            results.append(result)
            if item:
                mapped_items.add(item["absolute"])
            if count % 250 == 0:
                progress_path.write_text(json.dumps({"completed": count, "total": len(rows), "hash_cache": resume_cache}, indent=2) + "\n", encoding="utf-8")
    results.sort(key=lambda row: (row["repository"].casefold(), normalized(row["source_relative_path"])))
    progress_path.write_text(json.dumps({"completed": len(rows), "total": len(rows), "hash_cache": resume_cache}, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "archive_file_verification.csv", results, FILE_FIELDS)
    (output / "archive_file_verification.json").write_text(json.dumps({"schema": "phase0a2a.archive_file_verification.v1", "entries": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    path_map = [
        {
            "source_repository": row["repository"], "source_relative_path": row["source_relative_path"],
            "archive_absolute_path": row["archive_absolute_path"], "mapping_method": row["mapping_method"],
            "confidence": "HIGH" if row["verification_status"] == "PASS" else "LOW",
        }
        for row in results if row["archive_absolute_path"]
    ]
    (output / "archive_path_map.detected.json").write_text(json.dumps(path_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    dirs = []
    for source in read_manifest(args.directory_manifest):
        directory = source["directory"].replace("\\", "/").strip("/")
        expected = [row for row in results if normalized(f"{row['repository']}/{row['source_relative_path']}").startswith(normalized(directory) + "/")]
        verified = sum(row["verification_status"] == "PASS" for row in expected)
        missing = sum(row["verification_status"] == "MISSING" for row in expected)
        mismatches = sum(row["verification_status"] in ("SHA256_MISMATCH", "SIZE_MISMATCH") for row in expected)
        workspace_source = workspace / Path(directory)
        complete = len(expected) == int(source["file_count"]) and verified == len(expected) and not missing and not mismatches
        status = "PASS_ARCHIVED_SOURCE_REMOVED" if complete and not workspace_source.exists() else "PASS_COMPLETE_ARCHIVE" if complete else "PARTIAL_ARCHIVE" if verified else "FAILED_ARCHIVE"
        dirs.append(
            {
                "directory": directory, "expected_file_count": source["file_count"], "expected_total_bytes": source["total_bytes"],
                "archive_file_count": len(expected), "archive_total_bytes": sum(int(row["archive_size"] or 0) for row in expected),
                "verified_file_count": verified, "missing_count": missing, "unexpected_count": 0,
                "sha_mismatch_count": mismatches, "workspace_source_exists": str(workspace_source.exists()).lower(),
                "status": status, "reason": "all expected descendants verified" if complete else "one or more descendants failed or manifest count differs",
            }
        )
    dir_fields = ["directory", "expected_file_count", "expected_total_bytes", "archive_file_count", "archive_total_bytes", "verified_file_count", "missing_count", "unexpected_count", "sha_mismatch_count", "workspace_source_exists", "status", "reason"]
    write_csv(output / "archive_directory_verification.csv", dirs, dir_fields)

    curated = verify_curated(workspace, args.curated_manifest, results)
    curated_fields = list(curated[0]) if curated else ["status"]
    write_csv(output / "curated_asset_revalidation.csv", curated, curated_fields)
    (output / "curated_asset_revalidation.json").write_text(json.dumps({"schema": "phase0a2a.curated_revalidation.v1", "entries": curated}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    unexpected = []
    for item in files:
        if item["absolute"] not in mapped_items:
            try:
                item_sha = index.digest(item)
                error = ""
            except OSError as exc:
                item_sha, error = "", f"{type(exc).__name__}: {exc}"
            unexpected.append(
                {
                    "path": item["absolute"], "size": item["size"], "sha256": item_sha,
                    "possible_origin": "not mapped by supplied manifests", "duplicates_other_file": "REVIEW_REQUIRED",
                    "review_required": "YES", "error": error,
                }
            )
    for item in blocked:
        unexpected.append({"path": item["path"], "size": "", "sha256": "", "possible_origin": item["reason"], "duplicates_other_file": "", "review_required": "YES", "error": item["reason"]})
    unexpected_fields = ["path", "size", "sha256", "possible_origin", "duplicates_other_file", "review_required", "error"]
    write_csv(output / "unexpected_archive_files.csv", unexpected, unexpected_fields)

    failures = [row for row in results if row["verification_status"] != "PASS"]
    write_csv(output / "archive_verification_failures.csv", failures, FILE_FIELDS)
    all_pass = not failures and all(row["status"].startswith("PASS_") for row in dirs) and all(row["status"] == "PASS" for row in curated) and not blocked
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
