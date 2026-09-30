#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec "${PYTHON:-python}" -u -m chapter3_bser.experiments.d2_suite_v1.linux train "$@"
