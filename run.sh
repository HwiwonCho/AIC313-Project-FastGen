#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
DATA_ROOT="${1:-./data/pokemon-generation-one-22k}"
RUN_ROOT="${2:-./runs/best4}"
python -u -B train.py --data-root "$DATA_ROOT" --run-root "$RUN_ROOT"
python -B package.py --source "$RUN_ROOT/best.pt" --output "$RUN_ROOT/submission"
python -B "$RUN_ROOT/submission/evaluate_few.py" --data-root "$DATA_ROOT" \
  --checkpoint "$RUN_ROOT/submission/checkpoints/few_nfe.ckpt" --output "$RUN_ROOT/official_evaluation"
