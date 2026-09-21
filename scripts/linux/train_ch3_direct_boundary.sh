#!/usr/bin/env bash
set -u
# OpenMP runtime compatibility (explicit user opt-in only).
printf 'OPENMP_RUNTIME_ENV KMP_DUPLICATE_LIB_OK=%s\n' "${KMP_DUPLICATE_LIB_OK-<unset>}" >&2
case "${KMP_DUPLICATE_LIB_OK-}" in
    ""|[Ff][Aa][Ll][Ss][Ee]|0|[Nn][Oo]|[Oo][Ff][Ff]) ;;
    *) printf '%s\n' 'WARNING: OpenMP duplicate-runtime bypass explicitly requested; this unsafe workaround may crash or silently produce incorrect results. Numerical reliability is not guaranteed.' >&2 ;;
esac
# End OpenMP runtime compatibility.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
cd -- "$ROOT" || exit 1
exec python -B -m tools.ch3_baselines.run_training "$@" --baseline B3_direct_boundary
