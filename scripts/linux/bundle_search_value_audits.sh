#!/usr/bin/env bash
set -euo pipefail
# OpenMP runtime compatibility (explicit user opt-in only).
printf 'OPENMP_RUNTIME_ENV KMP_DUPLICATE_LIB_OK=%s\n' "${KMP_DUPLICATE_LIB_OK-<unset>}" >&2
case "${KMP_DUPLICATE_LIB_OK-}" in
    ""|[Ff][Aa][Ll][Ss][Ee]|0|[Nn][Oo]|[Oo][Ff][Ff]) ;;
    *) printf '%s\n' 'WARNING: OpenMP duplicate-runtime bypass explicitly requested; this unsafe workaround may crash or silently produce incorrect results. Numerical reliability is not guaranteed.' >&2 ;;
esac
# End OpenMP runtime compatibility.
audit_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$audit_root"
audit_python="${AUV_AUDIT_PYTHON:-python}"
command -v "$audit_python" >/dev/null || { printf 'Python not found: %s\n' "$audit_python" >&2; exit 1; }
exec "$audit_python" -m chapter3_bser.experiments.phase1c_prrac.search_value_audit.analysis_bundle "$@"
