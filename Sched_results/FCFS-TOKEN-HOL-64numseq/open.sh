#!/usr/bin/env bash
set -euo pipefail

GPU_ID=1
PORT=8001
HOST=127.0.0.1
MODEL=Qwen/Qwen3-8B
GPU_MEMORY_UTILIZATION=0.95
MAX_MODEL_LEN=2048

export VLLM_SCHED_TRACE="$PWD/fcfs_steps.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 64 \
  --max-num-batched-tokens 1024 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --scheduling-policy fcfs \
  --per-request-spec-decode-metrics summary \
  --speculative-config '{
    "method": "draft_model",
    "model": "Qwen/Qwen3-0.6B",
    "num_speculative_tokens": 5
  }' \
  2>&1 | tee -a server.log
