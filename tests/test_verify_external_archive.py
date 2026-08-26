from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "verify_external_archive.py"
SPEC = importlib.util.spec_from_file_location("verify_external_archive", TOOL)
verifier = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verifier)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class VerifyExternalArchiveTests(unittest.TestCase):
    def manifests(self, root: Path, relative: str, data: bytes) -> tuple[Path, Path, Path]:
        file_manifest = root / "files.csv"
        directory_manifest = root / "dirs.csv"
        curated_manifest = root / "curated.csv"
        write_csv(
            file_manifest,
            ["repository", "source_relative_path", "size_bytes", "sha256"],
            [{"repository": "CH3", "source_relative_path": relative, "size_bytes": len(data), "sha256": digest(data)}],
        )
        write_csv(
            directory_manifest,
            ["directory", "file_count", "total_bytes"],
            [{"directory": "CH3/data", "file_count": 1, "total_bytes": len(data)}],
        )
        write_csv(
            curated_manifest,
            ["repository", "source_relative_path", "size_bytes", "sha256", "curated_relative_path", "source_snapshot_id", "copy_status"],
            [{
                "repository": "CH3", "source_relative_path": relative, "size_bytes": len(data), "sha256": digest(data),
                "curated_relative_path": f"DATA_TO_KEEP_CURATED/CH3/{Path(relative).name}",
                "source_snapshot_id": "unit-test", "copy_status": "COPIED_AND_VERIFIED",
            }],
        )
        return file_manifest, directory_manifest, curated_manifest

    def args(self, workspace: Path, archive: Path, manifests: tuple[Path, Path, Path], output: Path) -> list[str]:
        files, dirs, curated = manifests
        return [
            "--workspace-root", str(workspace), "--archive-root", str(archive),
            "--file-manifest", str(files), "--directory-manifest", str(dirs),
            "--curated-manifest", str(curated), "--output-dir", str(output), "--workers", "2",
        ]

    def test_exact_mapping_full_hash_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, archive, output = root / "workspace", root / "archive", root / "out"
            data = "中文 archive payload".encode("utf-8")
            relative = "data/结果.bin"
            source = workspace / "CH3" / relative
            curated = workspace / "DATA_TO_KEEP_CURATED/CH3/结果.bin"
            archived = archive / "CH3" / relative
            for path in (source, curated, archived):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            manifests = self.manifests(root, relative, data)
            self.assertEqual(verifier.main(self.args(workspace, archive, manifests, output)), 0)
            rows = verifier.read_manifest(output / "archive_file_verification.csv")
            self.assertEqual(rows[0]["verification_status"], "PASS")
            self.assertEqual(rows[0]["mapping_method"], "exact_repo_relative")

    def test_ambiguous_suffix_is_not_auto_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, archive, output = root / "workspace", root / "archive", root / "out"
            data = b"same"
            relative = "data/x.bin"
            source = workspace / "CH3" / relative
            curated = workspace / "DATA_TO_KEEP_CURATED/CH3/x.bin"
            for path in (source, curated, archive / "custom-a/CH3/data/x.bin", archive / "custom-b/CH3/data/x.bin"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            manifests = self.manifests(root, relative, data)
            self.assertEqual(verifier.main(self.args(workspace, archive, manifests, output)), 1)
            rows = verifier.read_manifest(output / "archive_file_verification.csv")
            self.assertEqual(rows[0]["verification_status"], "AMBIGUOUS_MAPPING")
            self.assertEqual(rows[0]["duplicate_count"], "2")

    def test_missing_archive_root_is_configuration_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            manifests = self.manifests(root, "data/x.bin", b"x")
            missing = root / "does-not-exist"
            self.assertEqual(verifier.main(self.args(workspace, missing, manifests, root / "out")), 2)
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
