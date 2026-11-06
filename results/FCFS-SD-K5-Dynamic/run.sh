PORT=8001
HOST=127.0.0.1
MODEL=Qwen/Qwen3-8B

#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash run_real_mixed_bench.sh [WORKLOAD_DIR] [WORKLOAD_NAME]
DIR="${1:-real_mixed_workload}"
NAME="${2:-dolly_mixed_random_fixed128}"

DATASET="$DIR/$NAME.jsonl"

# Dataset 확인
if [[ ! -f "$DATASET" ]]; then
    echo "Dataset not found: $DATASET" >&2
    echo "Run build_real_mixed_workload.py first." >&2
    exit 1
fi

# 결과 저장 디렉터리
mkdir -p "results/$NAME"

# Request Rate별 실험
for RPS in 2 4 6 8; do
    echo "=================================="
    echo "Workload: $NAME"
    echo "Request Rate: $RPS req/s"
    echo "=================================="

    FLAGS=()

    # Fixed Output Length 실험에서 EOS 무시
    if [[ "$NAME" == *fixed128 ]]; then
        FLAGS+=(--ignore-eos)
    fi

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
        --num-prompts 200 \
        --num-warmups 10 \
        --request-rate "$RPS" \
        --burstiness 1.0 \
        --seed 42 \
        --percentile-metrics ttft,tpot,e2el \
        --metric-percentiles 50,90,95,99 \
        --save-result \
        --save-detailed \
        --result-dir "results/$NAME" \
        --result-filename "${NAME}_rps${RPS}.json" \
        "${FLAGS[@]}"

done