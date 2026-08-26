#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found on PATH" >&2
    exit 1
fi

CRK_CONDA_ENV="${CRK_CONDA_ENV:-AUV}"
CONDA_BASE="$(conda info --base)"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: conda.sh was not found at ${CONDA_SH}" >&2
    exit 1
fi
source "${CONDA_SH}"
conda activate "${CRK_CONDA_ENV}"

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python was not found on PATH" >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python --version
python -B - <<'PY'
import torch

print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is required for the Linux Phase1C-v2 training launcher")
print(f"gpu_name={torch.cuda.get_device_name(0)}")
PY

python -B -m chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.train_phase1c_v2 "$@"
