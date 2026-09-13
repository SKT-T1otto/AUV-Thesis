"""Production source identity shared by policy checkpoints and simulator state."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess

IMPLEMENTATION_VERSION = "hgr.exact_score_cycle.v1"


def fresh_source_identity():
    root = Path(__file__).resolve().parents[3]
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for package in ("core", "chapter3_bser") for p in sorted((root/package).rglob("*.py"))}
    return dict(implementation_version=IMPLEMENTATION_VERSION,
                sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files=files)


@lru_cache(maxsize=1)
def source_identity():
    return fresh_source_identity()


def validate_source_identity(identity):
    if not isinstance(identity, dict) or identity.get('implementation_version') != IMPLEMENTATION_VERSION:
        raise ValueError('invalid production source identity version')
    files = identity.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('production source identity requires its complete file inventory')
    for name, value in files.items():
        if not isinstance(name, str):
            raise ValueError('invalid production source filename')
        path = PurePosixPath(name)
        if (name != path.as_posix() or path.is_absolute() or not path.parts or '..' in path.parts
                or path.parts[0] not in ('core', 'chapter3_bser') or path.suffix != '.py'
                or not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None):
            raise ValueError(f'invalid production source record: {name!r}')
    expected = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if identity.get('sha256') != expected:
        raise ValueError('production source file inventory/aggregate hash mismatch')
    return identity


def require_source_match(saved, current, *, context='evaluation'):
    validate_source_identity(saved)
    validate_source_identity(current)
    if saved != current:
        changed = sorted(name for name in saved['files'].keys() | current['files'].keys()
                         if saved['files'].get(name) != current['files'].get(name))
        raise ValueError(f'{context} production source mismatch: checkpoint={saved["sha256"]}, '
                         f'current={current["sha256"]}; differing files ({len(changed)}): '
                         + ', '.join(changed[:12]) + '; evaluate the checkpoint at its original fixed source version')


def checkout_identity():
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.run(['git', '-c', 'safe.directory=' + root.as_posix(), *args],
                              cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    try:
        status = git('status', '--porcelain', '--untracked-files=normal')
        return dict(head=git('rev-parse', 'HEAD'), dirty=bool(status), status=status)
    except (OSError, subprocess.CalledProcessError):
        return dict(head=None, dirty=None, status='git metadata unavailable; content identity remains authoritative')
