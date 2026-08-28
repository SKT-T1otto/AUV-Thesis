#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CRK_CONDA_ENV="${CRK_CONDA_ENV:-AUV}"
CHECKPOINT=""
OUTPUT_DIR=""
RUNTIME_ORIGIN="legacy"
WORKERS="4"
EPISODES="50"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --runtime-origin) RUNTIME_ORIGIN="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        *) echo "unsupported argument: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "${CHECKPOINT}" || -z "${OUTPUT_DIR}" ]]; then echo "--checkpoint and --output-dir are required" >&2; exit 2; fi
if [[ "${RUNTIME_ORIGIN}" == "legacy" ]]; then CONFIG="configs/chapter3/bser_phase1c_prrac_s1_search_diag_legacy.json"; elif [[ "${RUNTIME_ORIGIN}" == "native" ]]; then CONFIG="configs/chapter3/bser_phase1c_prrac_s1_search_diag_native.json"; else echo "--runtime-origin must be legacy or native" >&2; exit 2; fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CRK_CONDA_ENV}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
cd "${REPO_ROOT}"
python -B -m chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT_DIR}" --workers "${WORKERS}" --episodes "${EPISODES}" --modes full_prrac searcher_residual_off --disable-failure-trace
