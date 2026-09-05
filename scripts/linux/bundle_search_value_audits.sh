#!/usr/bin/env bash
set -euo pipefail
audit_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$audit_root"
audit_python="${AUV_AUDIT_PYTHON:-python}"
command -v "$audit_python" >/dev/null || { printf 'Python not found: %s\n' "$audit_python" >&2; exit 1; }
exec "$audit_python" -m chapter3_bser.experiments.phase1c_prrac.search_value_audit.analysis_bundle "$@"
