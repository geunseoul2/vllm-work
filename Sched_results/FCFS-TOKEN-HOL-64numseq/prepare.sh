#!/usr/bin/env bash
set -euo pipefail
BASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Requires the same tokenizer environment as the serving experiment.
python3 "$BASE/../../build_real_mixed_workload.py" \
  --output-dir "$BASE/workload" --model "${MODEL:-Qwen/Qwen3-8B}" \
  --short-count 16 --long-count 4 \
  --short-target 128 --short-min 112 --short-max 144 \
  --long-target 1536 --long-min 1400 --long-max 1664 \
  --fixed-output 128 --seed 42 "$@"
