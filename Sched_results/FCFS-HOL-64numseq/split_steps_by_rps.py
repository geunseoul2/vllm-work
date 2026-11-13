#!/usr/bin/env python3
"""Split scheduler trace using benchmark request timestamps (excludes warmup).

Run after the server has flushed/closed its trace. Benchmark and server must
share the same monotonic clock, as they do when run on the same host.
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=base / "fcfs_steps.csv")
    parser.add_argument("--results-dir", type=Path, default=base / "latency_test")
    parser.add_argument("--output-dir", type=Path, default=base / "split_steps")
    args = parser.parse_args()
    files = sorted(args.results_dir.glob("*_rps*.json"))
    if not files:
        parser.error(f"No benchmark JSON files found in {args.results_dir}")
    with args.trace.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not rows:
        parser.error("Trace contains no steps")
    times = [float(row["timestamp"]) for row in rows]
    if any(b < a for a, b in zip(times, times[1:])):
        parser.error("Trace timestamps are not monotonic")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        data = json.loads(path.read_text())
        starts = data.get("start_times", [])
        latencies = data.get("latencies", [])
        if not starts or len(starts) != len(latencies):
            print(f"SKIP {path.name}: missing or mismatched start_times/latencies")
            continue
        start = min(starts)
        end = max(t + latency for t, latency in zip(starts, latencies))
        selected = [row for row, t in zip(rows, times) if start <= t <= end]
        if not selected:
            print(f"SKIP {path.name}: no overlapping steps; check server run/clock")
            continue
        output = args.output_dir / f"{path.stem}_steps.csv"
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
        print(f"{output}: {len(selected)} steps, window {start:.6f}..{end:.6f}")
        if times[0] > start or times[-1] < end:
            print("WARNING: trace does not cover the full request window; "
                  "flush/close server trace and rerun, or check matching experiment.")


if __name__ == "__main__":
    main()
