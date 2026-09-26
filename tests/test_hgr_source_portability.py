"""Git must preserve frozen HGR source bytes across checkout platforms."""
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
FROZEN_PATHS = (
    "chapter3_bser/controllers/action_adapter.py",
    "chapter3_bser/experiments/phase1b1_pilot/run_pilot.py",
    "chapter3_bser/hysteresis/policy.py",
    "chapter3_bser/online/allocator.py",
    "chapter3_bser/online/config.py",
    "chapter3_bser/online/controller.py",
)


@pytest.mark.parametrize("autocrlf", ["false", "true"])
def test_git_round_trip_preserves_frozen_source_bytes(tmp_path, autocrlf):
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-c", f"safe.directory={repository.as_posix()}",
             "-c", f"core.autocrlf={autocrlf}", "-c", "core.safecrlf=false", *args],
            cwd=repository, check=True, capture_output=True,
        ).stdout

    git("init", "--quiet")
    (repository / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
    expected = {name: (ROOT / name).read_bytes() for name in FROZEN_PATHS}
    # Both full CRLF and mixed-ending files must survive unchanged.
    assert any(b"\r\n" in value for value in expected.values())
    assert any(b"\n" in value.replace(b"\r\n", b"") for value in expected.values())
    for name, value in expected.items():
        target = repository / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    ordinary = "ordinary.py"
    (repository / ordinary).write_bytes(b"first = 1\r\nsecond = 2\r\n")
    git("add", "--", ".gitattributes", ordinary, *FROZEN_PATHS)

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git("checkout-index", "--all", f"--prefix={checkout.as_posix()}/")
    for name, value in expected.items():
        assert git("show", f":{name}") == value, name
        assert (checkout / name).read_bytes() == value, name
    # The byte-preservation exception must not change normal Python LF rules.
    assert (checkout / ordinary).read_bytes() == b"first = 1\nsecond = 2\n"
