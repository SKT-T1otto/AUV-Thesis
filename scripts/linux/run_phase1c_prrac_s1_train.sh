#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CRK_CONDA_ENV="${CRK_CONDA_ENV:-AUV}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CRK_CONDA_ENV}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
FORMAL=false
PASSTHROUGH=()
for argument in "$@"; do
    if [[ "${argument}" == "--formal" ]]; then FORMAL=true; else PASSTHROUGH+=("${argument}"); fi
done
if [[ "${FORMAL}" == false ]]; then PASSTHROUGH+=("--dry-run"); fi
cd "${REPO_ROOT}"
python -B -m chapter3_bser.experiments.phase1c_prrac.train_phase1c_prrac --config configs/chapter3/bser_phase1c_prrac_s1_train.json "${PASSTHROUGH[@]}"
