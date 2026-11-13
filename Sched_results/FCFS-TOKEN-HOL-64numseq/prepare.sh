#!/usr/bin/env bash
set -euo pipefail

MODEL=Qwen/Qwen3-8B

python3 ../../build_real_mixed_workload.py \
  --model "$MODEL" \
  --output-dir workload \
  --short-count 16 \
  --long-count 4 \
  --short-target 128 --short-min 112 --short-max 144 \
  --long-target 1536 --long-min 1400 --long-max 1664 \
  --fixed-output 128 \
  --seed 42 \
  "$@"
