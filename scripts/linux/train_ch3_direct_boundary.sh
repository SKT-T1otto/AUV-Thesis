#!/usr/bin/env bash
set -u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
cd -- "$ROOT" || exit 1
exec python -B -m tools.ch3_baselines.run_training "$@" --baseline B3_direct_boundary
