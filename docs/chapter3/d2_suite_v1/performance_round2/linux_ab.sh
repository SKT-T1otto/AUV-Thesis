#!/usr/bin/env bash
# Run manually on Linux. All outputs go to a NEW directory; no formal job runs.
set -euo pipefail
: "${BASELINE:?Set BASELINE to a separate pristine c468663 checkout}"
: "${OPTIMIZED:?Set OPTIMIZED to the complete optimized checkout}"
: "${MANIFEST:?Set MANIFEST to the frozen linux_pipeline_02/inputs/train.json}"
: "${AB_OUTPUT:?Set AB_OUTPUT to a new absolute result directory}"
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for name in BASELINE OPTIMIZED MANIFEST AB_OUTPUT; do
  if [[ "${!name}" != /* ]]; then
    printf '%s must be an absolute Linux path\n' "$name" >&2
    exit 2
  fi
done
test "$(git -C "$BASELINE" rev-parse HEAD)" = c468663acf05fa11b41eb068169ae44cc79a2376
test -z "$(git -C "$BASELINE" status --porcelain)"
test ! -e "$AB_OUTPUT"
mkdir -p "$AB_OUTPUT/inputs"
cp "$MANIFEST" "$AB_OUTPUT/inputs/train.json"
MANIFEST="$AB_OUTPUT/inputs/train.json"

# Resolve IDs from the actual frozen manifest, never from assumed indices.
read -r NORMAL_A SLOW_A NORMAL_B SLOW_B < <(
  "$PYTHON" - "$MANIFEST" <<'PY'
import json, sys
scenes = json.load(open(sys.argv[1], encoding="utf-8"))["scenarios"]
ids = ["unknown_train_m20_0072", "unknown_train_m20_0200",
       "unknown_train_m20_0089", "unknown_train_m20_0120"]
indices = []
for name in ids:
    matches = [i for i, scene in enumerate(scenes) if scene["scenario_id"] == name]
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one {name}: {matches}")
    indices.append(matches[0])
print(*indices)
PY
)
printf 'Resolved scene indices: %s %s %s %s\n' "$NORMAL_A" "$SLOW_A" "$NORMAL_B" "$SLOW_B"

run_case() {
  local checkout="$1" label="$2" arm="$3" normal="$4" slow="$5"
  (
    cd "$checkout"
    # Load the SAME diagnostic harness externally. The old checkout's sealed
    # benchmark and all production files stay unchanged. No sys.path changes.
    "$PYTHON" -c '
import importlib.util, pathlib, sys
path = sys.argv.pop(1)
spec = importlib.util.spec_from_file_location("d2_ab_diagnostic", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT = pathlib.Path.cwd()
module.main()
' "$OPTIMIZED/tools/benchmark_d2_performance.py" \
      --manifest "$MANIFEST" --scenes "$normal" "$slow" --arm "$arm" \
      --seed 2729 --device cpu --threads 1 --training-episodes 2 \
      --label "$label" --output "$AB_OUTPUT/$label"
  )
}

# Both arms use the same planner. These are bounded fresh-model episodes,
# preserving the real warmup/update cadence, with no checkpoint restoration.
for arm in B2 B3; do
  for pair in A B; do
    if [[ "$pair" = A ]]; then normal="$NORMAL_A"; slow="$SLOW_A"
    else normal="$NORMAL_B"; slow="$SLOW_B"; fi
    run_case "$BASELINE" "${arm}_${pair}_before" "$arm" "$normal" "$slow"
    run_case "$OPTIMIZED" "${arm}_${pair}_after" "$arm" "$normal" "$slow"
    (
      cd "$OPTIMIZED"
      "$PYTHON" -m tools.compare_d2_performance \
        "$AB_OUTPUT/${arm}_${pair}_before/benchmark.json" \
        "$AB_OUTPUT/${arm}_${pair}_after/benchmark.json" \
        --require-training --output "$AB_OUTPUT/${arm}_${pair}_exact.json"
    )
  done
done
