#!/bin/bash

set -e
# OpenMP runtime compatibility (explicit user opt-in only).
printf 'OPENMP_RUNTIME_ENV KMP_DUPLICATE_LIB_OK=%s\n' "${KMP_DUPLICATE_LIB_OK-<unset>}" >&2
case "${KMP_DUPLICATE_LIB_OK-}" in
    ""|[Ff][Aa][Ll][Ss][Ee]|0|[Nn][Oo]|[Oo][Ff][Ff]) ;;
    *) printf '%s\n' 'WARNING: OpenMP duplicate-runtime bypass explicitly requested; this unsafe workaround may crash or silently produce incorrect results. Numerical reliability is not guaranteed.' >&2 ;;
esac
# End OpenMP runtime compatibility.

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
python -B - "$@" <<'PY'
import sys

import torch

cuda_requested = any(
    argument.startswith("--device=cuda")
    or (
        argument == "--device"
        and index + 1 < len(sys.argv)
        and sys.argv[index + 1].startswith("cuda")
    )
    for index, argument in enumerate(sys.argv[1:], start=1)
)
print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gpu_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
if cuda_requested and not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA was explicitly requested for diagnostic evaluation")
PY

python -B -m chapter3_bser.experiments.phase1c_bser_rmaddpg.evaluate_phase1c_checkpoints "$@"
