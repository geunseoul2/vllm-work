#!/usr/bin/env bash
set -euo pipefail

PORT=8001
HOST=127.0.0.1
MODEL=Qwen/Qwen3-8B

# bash run.sh [WORKLOAD_DIR] [WORKLOAD_NAME]
DIR="${1:-workload}"
NAME="${2:-dolly_long_first_fixed128}"
DATASET="$DIR/$NAME.jsonl"

if [[ ! -f "$DATASET" ]]; then
    echo "Dataset not found: $DATASET" >&2
    echo "Run bash prepare.sh first." >&2
    exit 1
fi

mkdir -p latency_test

# 먼저 inf로 실행. 필요하면 아래를 8 12 inf 등으로 변경.
for RPS in inf; do
    echo "Workload: $NAME / Request Rate: $RPS req/s"

    curl -fsS "http://${HOST}:${PORT}/metrics" \
      > "latency_test/${NAME}_rps${RPS}_before.prom"

    vllm bench serve \
      --backend vllm \
      --host "$HOST" \
      --port "$PORT" \
      --endpoint /v1/completions \
      --model "$MODEL" \
      --dataset-name custom \
      --dataset-path "$DATASET" \
      --custom-output-len -1 \
      --skip-chat-template \
      --disable-shuffle \
      --no-oversample \
      --num-prompts 20 \
      --num-warmups 10 \
      --request-rate "$RPS" \
      --burstiness 1.0 \
      --seed 42 \
      --ignore-eos \
      --percentile-metrics ttft,tpot,e2el \
      --metric-percentiles 50,90,95,99 \
      --save-result \
      --save-detailed \
      --result-dir latency_test \
      --result-filename "${NAME}_rps${RPS}.json"

    curl -fsS "http://${HOST}:${PORT}/metrics" \
      > "latency_test/${NAME}_rps${RPS}_after.prom"
done
