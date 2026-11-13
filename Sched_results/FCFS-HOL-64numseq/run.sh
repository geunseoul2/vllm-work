PORT=8001
HOST=127.0.0.1
MODEL=Qwen/Qwen3-8B

#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash run.sh [WORKLOAD_DIR] [WORKLOAD_NAME]
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
mkdir -p "latency_test"

# Prometheus queue-time histogram은 서버 시작 이후 누적되므로 전후를 저장한다.
save_queue_metrics() {
    local output_file="$1"
    if ! curl -fsS --max-time 10 "http://${HOST}:${PORT}/metrics" \
        | grep '^vllm:request_queue_time_seconds' \
        | tee "$output_file"; then
        echo "WARNING: queue-time metrics 조회 실패 또는 메트릭 없음: $output_file" >&2
    fi
}

# Request Rate별 실험
for RPS in 4 8 12 inf; do
    echo "=================================="
    echo "Workload: $NAME"
    echo "Request Rate: $RPS req/s"
    echo "=================================="

    FLAGS=()

    # Fixed Output Length 실험에서 EOS 무시
    if [[ "$NAME" == *fixed128 ]]; then
        FLAGS+=(--ignore-eos)
    fi

    save_queue_metrics "latency_test/${NAME}_rps${RPS}_queue_before.prom"

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
        --result-dir "latency_test" \
        --result-filename "${NAME}_rps${RPS}.json" \
        "${FLAGS[@]}"

    echo "Queue time metrics (cumulative):"
    save_queue_metrics "latency_test/${NAME}_rps${RPS}_queue_after.prom"

done
