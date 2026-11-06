
import csv
import json
import re
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

MODEL = "Qwen/Qwen3-8B"
HOST = "127.0.0.1"
PORT = 8001

DATASET_DIR = "data/speed_bench"
SUBSET = "throughput_1k"

RATES = [1, 2, 4, 8]
NUM_PROMPTS = 200
REPEATS = 3
MAX_CONCURRENCY = 16
OUTPUT_LEN = 128

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

METRICS_URL = f"http://{HOST}:{PORT}/metrics"

# vLLM Prometheus metric names.
# Prometheus may expose counters with a _total suffix.
ACCEPTED = "vllm:spec_decode_num_accepted_tokens"
DRAFTED = "vllm:spec_decode_num_draft_tokens"
PREEMPTIONS = "vllm:num_preemptions"

LINE_RE = re.compile(
    r'^([\w:.]+)(?:\{[^}]*\})?\s+([-+\d.eE]+)'
)


def get_metrics():
    """Read Prometheus metrics from the vLLM server."""
    raw = urlopen(METRICS_URL, timeout=10).read().decode()
    result = {}

    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue

        match = LINE_RE.match(line)
        if not match:
            continue

        name = match.group(1).removesuffix("_total")
        value = float(match.group(2))

        result[name] = result.get(name, 0.0) + value

    return result

def counter_delta(before, after, metric):
    """Return None if metric is missing or counter reset."""
    if metric not in before or metric not in after:
        return None

    delta = after[metric] - before[metric]
    return delta if delta >= 0 else None


def run_benchmark(rate, repeat):
    name = f"rate_{rate}_repeat_{repeat}"
    output_file = f"{name}.json"

    command = [
        "vllm", "bench", "serve",
        "--backend", "openai-chat",
        "--endpoint", "/v1/chat/completions",
        "--host", HOST,
        "--port", str(PORT),
        "--model", MODEL,

        # NVIDIA SPEED-Bench
        "--dataset-name", "speed_bench",
        "--dataset-path", DATASET_DIR,
        "--speed-bench-dataset-subset", SUBSET,
        "--speed-bench-output-len", str(OUTPUT_LEN),

        # Load configuration
        "--num-prompts", str(NUM_PROMPTS),
        "--no-oversample",
        "--request-rate", str(rate),
        "--max-concurrency", str(MAX_CONCURRENCY),

        # Reproducibility
        "--temperature", "0",
        "--ignore-eos",
        "--seed", "42",

        # Measurements
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95,99",
        "--save-result",
        "--save-detailed",
        "--result-dir", str(RESULT_DIR),
        "--result-filename", output_file,
    ]

    print(f"\n========== Rate={rate}, Repeat={repeat} ==========")

    before = get_metrics()

    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started

    after = get_metrics()

    with open(RESULT_DIR / output_file) as f:
        bench = json.load(f)

    accepted = counter_delta(before, after, ACCEPTED)
    drafted = counter_delta(before, after, DRAFTED)
    preemptions = counter_delta(before, after, PREEMPTIONS)

    acceptance_rate = (
        accepted / drafted
        if accepted is not None
        and drafted is not None
        and drafted > 0
        else None
    )

    row = {
        "rate": rate,
        "repeat": repeat,
        "elapsed_s": elapsed,

        "throughput_tok_s": bench.get("output_throughput"),
        "request_throughput": bench.get("request_throughput"),

        "mean_ttft_ms": bench.get("mean_ttft_ms"),
        "p99_ttft_ms": bench.get("p99_ttft_ms"),

        "mean_tpot_ms": bench.get("mean_tpot_ms"),
        "p99_tpot_ms": bench.get("p99_tpot_ms"),

        "accepted_tokens": accepted,
        "drafted_tokens": drafted,
        "acceptance_rate": acceptance_rate,

        "preemptions": preemptions,
    }

    # Keep raw metric snapshots for debugging.
    with open(RESULT_DIR / f"{name}_metrics.json", "w") as f:
        json.dump({"before": before, "after": after}, f, indent=2)

    print(json.dumps(row, indent=2))
    return row


def main():
    dataset = Path(DATASET_DIR) / f"{SUBSET}.jsonl"
    if not dataset.exists():
        raise FileNotFoundError(
            f"{dataset} not found. Prepare SPEED-Bench first."
        )

    # Check server availability before starting.
    get_metrics()

    results = []

    for repeat in range(1, REPEATS + 1):
        # Alternate rate order to reduce order-related bias.
        rates = RATES if repeat % 2 else list(reversed(RATES))

        for rate in rates:
            row = run_benchmark(rate, repeat)
            results.append(row)

            with open(RESULT_DIR / "summary.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

            # Give the server time to become idle.
            time.sleep(5)

    print("\nBenchmark finished.")
    print("Results saved to results/summary.csv")


if __name__ == "__main__":
    main()
