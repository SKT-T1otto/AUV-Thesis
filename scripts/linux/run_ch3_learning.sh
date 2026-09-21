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
CH3_CONDA_EXE="${CRK_CONDA_EXE:-${CONDA_EXE:-conda}}"
CONDA_BASE="$("${CH3_CONDA_EXE}" info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CRK_CONDA_ENV:-AUV}"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${REPO_ROOT}"
exec python -B -m chapter3_bser.experiments.hgr.cli "$@"
