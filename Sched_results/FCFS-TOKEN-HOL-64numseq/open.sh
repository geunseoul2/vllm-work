#!/usr/bin/env bash
set -euo pipefail
BASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${RUN_ID:?Set RUN_ID to a new run name, shared with run.sh}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
RUN_DIR="$BASE/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
[[ ! -e "$RUN_DIR/fcfs_steps.csv" && ! -e "$RUN_DIR/server.log" ]] || { echo 'Run already exists; choose a new RUN_ID.' >&2; exit 2; }
export VLLM_SCHED_TRACE="$RUN_DIR/fcfs_steps.csv"
printf '%s
' "model=${MODEL:-Qwen/Qwen3-8B}" "gpu=${GPU_ID:-1}" 'max_num_seqs=64' 'max_num_batched_tokens=1024' 'K=5' > "$RUN_DIR/config.txt"
command -v nvidia-smi >/dev/null && nvidia-smi > "$RUN_DIR/gpu.txt" || true
python3 -m pip show vllm > "$RUN_DIR/vllm_package.txt" 2>&1 || true
CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" vllm serve "${MODEL:-Qwen/Qwen3-8B}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8001}" \
  --gpu-memory-utilization 0.95 --max-model-len 2048 \
  --max-num-seqs 64 --max-num-batched-tokens 1024 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --scheduling-policy fcfs --per-request-spec-decode-metrics summary \
  --speculative-config '{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":5}' \
  2>&1 | tee "$RUN_DIR/server.log"
