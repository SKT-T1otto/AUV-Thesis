#!/usr/bin/env bash

set -euo pipefail
# OpenMP runtime compatibility (explicit user opt-in only).
printf 'OPENMP_RUNTIME_ENV KMP_DUPLICATE_LIB_OK=%s\n' "${KMP_DUPLICATE_LIB_OK-<unset>}" >&2
case "${KMP_DUPLICATE_LIB_OK-}" in
    ""|[Ff][Aa][Ll][Ss][Ee]|0|[Nn][Oo]|[Oo][Ff][Ff]) ;;
    *) printf '%s\n' 'WARNING: OpenMP duplicate-runtime bypass explicitly requested; this unsafe workaround may crash or silently produce incorrect results. Numerical reliability is not guaranteed.' >&2 ;;
esac
# End OpenMP runtime compatibility.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CRK_CONDA_ENV="${CRK_CONDA_ENV:-AUV}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found on PATH" >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: conda.sh was not found at ${CONDA_SH}" >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate "${CRK_CONDA_ENV}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
python -B -m \
    chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints \
    "$@"
