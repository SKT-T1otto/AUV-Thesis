#!/usr/bin/env bash
# Sourced by D1/D2 launchers. No training or automatic checkpoint restoration.
set -euo pipefail
audit_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$audit_root"
: "${AUV_AUDIT_CHECKPOINT:?Set AUV_AUDIT_CHECKPOINT to the trained Ep100 file}"
: "${AUV_AUDIT_OFF_OUTPUT:?Set AUV_AUDIT_OFF_OUTPUT to the historical OFF directory}"
: "${AUV_AUDIT_ON_OUTPUT:?Set AUV_AUDIT_ON_OUTPUT to the historical ON directory}"
: "${AUV_AUDIT_OUTPUT_DIR:?Set AUV_AUDIT_OUTPUT_DIR to a NEW output directory}"
audit_python="${AUV_AUDIT_PYTHON:-python}"
command -v "$audit_python" >/dev/null || { printf 'Python not found: %s\n' "$audit_python" >&2; exit 1; }
[[ -f "$AUV_AUDIT_CHECKPOINT" ]] || { printf 'Missing checkpoint: %s\n' "$AUV_AUDIT_CHECKPOINT" >&2; exit 1; }
audit_source=()
if [[ -n "${AUV_AUDIT_TRAINING_CONFIG:-}" && -n "${AUV_AUDIT_TRAINING_MANIFEST:-}" ]]; then
  printf '%s\n' 'Set only one of AUV_AUDIT_TRAINING_CONFIG / AUV_AUDIT_TRAINING_MANIFEST' >&2; exit 1
elif [[ -n "${AUV_AUDIT_TRAINING_CONFIG:-}" ]]; then
  [[ -f "$AUV_AUDIT_TRAINING_CONFIG" ]] || { printf 'Missing training config: %s\n' "$AUV_AUDIT_TRAINING_CONFIG" >&2; exit 1; }
  audit_source=(--training-config "$AUV_AUDIT_TRAINING_CONFIG")
elif [[ -n "${AUV_AUDIT_TRAINING_MANIFEST:-}" ]]; then
  [[ -f "$AUV_AUDIT_TRAINING_MANIFEST" ]] || { printf 'Missing training manifest: %s\n' "$AUV_AUDIT_TRAINING_MANIFEST" >&2; exit 1; }
  audit_source=(--training-manifest "$AUV_AUDIT_TRAINING_MANIFEST")
else
  printf '%s\n' 'Set AUV_AUDIT_TRAINING_CONFIG or AUV_AUDIT_TRAINING_MANIFEST' >&2; exit 1
fi
for audit_dir in "$AUV_AUDIT_OFF_OUTPUT" "$AUV_AUDIT_ON_OUTPUT"; do
  for audit_file in resolved_evaluation_config.json evaluation_manifest.json episode_evaluation.csv; do
    [[ -f "$audit_dir/$audit_file" ]] || { printf 'Missing historical input: %s/%s\n' "$audit_dir" "$audit_file" >&2; exit 1; }
  done
done
[[ -f "$AUV_AUDIT_ON_OUTPUT/search_value_guidance_metrics.json" ]] || { printf '%s\n' 'Missing ON search_value_guidance_metrics.json' >&2; exit 1; }
audit_args=(--checkpoint "$AUV_AUDIT_CHECKPOINT" "${audit_source[@]}"
  --historical-off-output "$AUV_AUDIT_OFF_OUTPUT" --historical-on-output "$AUV_AUDIT_ON_OUTPUT"
  --output-dir "$AUV_AUDIT_OUTPUT_DIR" --workers "${AUV_AUDIT_WORKERS:-1}")
