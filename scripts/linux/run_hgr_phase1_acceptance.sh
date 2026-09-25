#!/usr/bin/env bash
# NEW manual-only runtime acceptance. Never starts training/resume.
set -euo pipefail
cd "$(dirname "$0")/../.."
output="${1:-outputs/chapter3/hgr_phase1/collision_terminal/manual_acceptance_01}"
python_bin="${PYTHON:-python}"
log="${output}.console.log"
if [[ -e "$log" ]]; then
  printf 'Refusing existing log: %s\n' "$log" >&2
  exit 1
fi
if [[ -e "$output" ]] && { [[ ! -d "$output" ]] || [[ -n "$(ls -A "$output")" ]]; }; then
  printf 'Use a new or empty output directory: %s\n' "$output" >&2
  exit 1
fi
mkdir -p "$(dirname "$log")"
"$python_bin" -m chapter3_bser.experiments.hgr.phase1_acceptance --config configs/chapter3/hgr_phase1_zero.json --output "$output" 2>&1 | tee "$log"
