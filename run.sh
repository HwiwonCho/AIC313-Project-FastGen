#!/usr/bin/env bash
# Activate the existing FastGen environment before invoking this script.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
DATA_ROOT="${1:-/root/AIC313-Project-FastGen/data/pokemon-generation-one-22k}"
RUN_ROOT="${2:-/root/FastGen/runs/deadline}"
shift "$(( $# >= 2 ? 2 : $# ))"
exec python -u -B pipeline.py run --config configs/deadline.json \
  --data-root "$DATA_ROOT" --run-root "$RUN_ROOT" "$@"
