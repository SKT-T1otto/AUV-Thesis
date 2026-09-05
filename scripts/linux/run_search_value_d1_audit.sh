#!/usr/bin/env bash
set -euo pipefail
audit_mode="${1:?Usage: bash run_search_value_d1_audit.sh smoke|diagnostic [--manifest PATH] [--save-features]}"
case "$audit_mode" in smoke|diagnostic) ;; *) printf 'Invalid mode: %s\n' "$audit_mode" >&2; exit 1 ;; esac
shift
source "$(dirname -- "${BASH_SOURCE[0]}")/_search_value_audit_common.sh"
exec "$audit_python" -u -m chapter3_bser.experiments.phase1c_prrac.search_value_audit.prediction_audit "${audit_args[@]}" "--$audit_mode" "$@"
