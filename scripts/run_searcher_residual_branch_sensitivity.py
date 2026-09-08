"""Root-relative diagnostic entry point; no sys.path injection."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "chapter3_bser.experiments.phase1c_prrac.run_residual_branch_sensitivity", *sys.argv[1:]], cwd=Path(__file__).resolve().parents[1]))
