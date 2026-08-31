#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT=""; OUTPUT_DIR=""; WORKERS=4; EPISODES=10; FORMAL=0
while [[ $# -gt 0 ]]; do case "$1" in --checkpoint) CHECKPOINT="$2"; shift 2;; --output-dir) OUTPUT_DIR="$2"; shift 2;; --workers) WORKERS="$2"; shift 2;; --episodes) EPISODES="$2"; shift 2;; --formal) FORMAL=1; shift;; *) echo "ERROR: unknown argument $1" >&2; exit 2;; esac; done
[[ -n "${CHECKPOINT}" && -n "${OUTPUT_DIR}" ]] || { echo "ERROR: --checkpoint and --output-dir are required" >&2; exit 2; }
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda was not found on PATH" >&2; exit 1; }
CONDA_BASE="$(conda info --base)"; source "${CONDA_BASE}/etc/profile.d/conda.sh"; conda activate "${CRK_CONDA_ENV:-AUV}"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
ARGS=(python -B -m chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints --config configs/chapter3/bser_phase1c_prrac_s2a_collision_ablation.json --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT_DIR}" --workers "${WORKERS}" --modes full_prrac --execution-variants B1_ATOMIC_LAST_VALID --search-recovery-variants S2A_C0_BASELINE S2A_C1_ROUTE_REFRESH S2A_C2_EGRESS_ROUTE)
if [[ ${FORMAL} -eq 0 ]]; then ARGS+=(--episodes "${EPISODES}"); fi
"${ARGS[@]}"
