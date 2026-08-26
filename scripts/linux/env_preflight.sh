#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPORT_PATH="${REPO_ROOT}/outputs/runtime/linux_preflight.json"

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

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi was not found on PATH" >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
python -B -c "import torch"
nvidia-smi -L >/dev/null
mkdir -p "$(dirname "${REPORT_PATH}")"

python -B - "${REPORT_PATH}" <<'PY'
import json
import socket
import sys
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: torch.cuda.is_available() is False")

report = {
    "hostname": socket.gethostname(),
    "python_version": sys.version.split()[0],
    "torch_version": str(torch.__version__),
    "cuda_runtime": torch.version.cuda,
    "gpu_name": torch.cuda.get_device_name(0),
    "device_count": int(torch.cuda.device_count()),
}
path = Path(sys.argv[1])
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, sort_keys=True))
PY

echo "Linux preflight report: ${REPORT_PATH}"
