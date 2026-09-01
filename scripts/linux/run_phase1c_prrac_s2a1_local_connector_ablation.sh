#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CHECKPOINT=""
OUTPUT_DIR="outputs/chapter3/phase1c_prrac/s2a1_local_connector_ablation"
WORKERS=4
EPISODES=10
FORMAL=0
SCENARIO_ID_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --formal) FORMAL=1; shift ;;
    --scenario-id-file) SCENARIO_ID_FILE="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "${CHECKPOINT}" ]] || { echo "ERROR: --checkpoint is required" >&2; exit 2; }
if [[ ${FORMAL} -eq 1 && -n "${SCENARIO_ID_FILE}" ]]; then
  echo "ERROR: --formal cannot be combined with --scenario-id-file" >&2
  exit 2
fi
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda was not found on PATH" >&2; exit 1; }
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CRK_CONDA_ENV:-AUV}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
ARGS=(python -B -m chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints --config configs/chapter3/bser_phase1c_prrac_s2a1_local_connector_ablation.json --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT_DIR}" --workers "${WORKERS}" --modes full_prrac --execution-variants B1_ATOMIC_LAST_VALID --search-recovery-variants S2A1_C0_BASELINE S2A1_C1_FORCED_REFRESH S2A1_C2_LOCAL_CONNECTOR)
if [[ ${FORMAL} -eq 1 ]]; then
  ARGS+=(--formal)
elif [[ -n "${SCENARIO_ID_FILE}" ]]; then
  ARGS+=(--scenario-id-file "${SCENARIO_ID_FILE}")
else
  ARGS+=(--episodes "${EPISODES}")
fi
"${ARGS[@]}"
