#!/usr/bin/env bash
set -euo pipefail
BASE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${RUN_ID:?Use the RUN_ID of the server started by open.sh}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
ORDER="${1:-long_first}"
RPS="${2:-inf}"
case "$ORDER" in long_first|short_first|mixed_random) ;; *) echo 'Use long_first, short_first or mixed_random' >&2; exit 2;; esac
RUN_DIR="$BASE/runs/$RUN_ID"
DATASET="$BASE/workload/dolly_${ORDER}_fixed128.jsonl"
[[ -f "$DATASET" && -f "$RUN_DIR/config.txt" ]] || { echo 'Prepare workload and start the server first.' >&2; exit 2; }
[[ ! -e "$RUN_DIR/result.json" && ! -e "$RUN_DIR/benchmark.log" ]] || { echo 'One benchmark per fresh server RUN_ID.' >&2; exit 2; }
[[ $(wc -l < "$DATASET") -eq 20 ]] || { echo 'Expected 20 requests, below sequence limit 64.' >&2; exit 2; }
cp "$BASE/workload/dolly_${ORDER}_fixed128_manifest.csv" "$RUN_DIR/manifest.csv"
printf '%s
' "order=$ORDER" "request_rate=$RPS" >> "$RUN_DIR/config.txt"
curl -fsS "http://${HOST:-127.0.0.1}:${PORT:-8001}/metrics" > "$RUN_DIR/before.prom"
vllm bench serve --backend vllm --host "${HOST:-127.0.0.1}" --port "${PORT:-8001}" \
  --endpoint /v1/completions --model "${MODEL:-Qwen/Qwen3-8B}" \
  --dataset-name custom --dataset-path "$DATASET" --custom-output-len -1 \
  --skip-chat-template --disable-shuffle --no-oversample \
  --num-prompts 20 --num-warmups 10 --request-rate "$RPS" --burstiness 1.0 \
  --seed 42 --ignore-eos --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,90,95,99 --save-result --save-detailed \
  --result-dir "$RUN_DIR" --result-filename result.json 2>&1 | tee "$RUN_DIR/benchmark.log"
curl -fsS "http://${HOST:-127.0.0.1}:${PORT:-8001}/metrics" > "$RUN_DIR/after.prom"
echo 'Stop the server gracefully (Ctrl-C), then run analyze.py on this RUN_ID to include buffered trace rows.'
