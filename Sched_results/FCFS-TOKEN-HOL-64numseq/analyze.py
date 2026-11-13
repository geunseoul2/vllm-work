#!/usr/bin/env python3
"""Analyze measured-window trace only; never infer HOL causality from counters."""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('result', nargs='?', default='dolly_long_first_fixed128_rpsinf.json',
                    help='JSON filename in latency_test, or an absolute path')
parser.add_argument('--trace', type=Path, help='Optional archived trace path')
args = parser.parse_args()
p = Path(__file__).resolve().parent
result_path = Path(args.result)
if not result_path.is_absolute():
    result_path = p / 'latency_test' / result_path
trace_path = args.trace or p / 'fcfs_steps.csv'
d = json.loads(result_path.read_text())
starts, latencies = d['start_times'], d['latencies']
if len(starts) != 20 or len(latencies) != 20 or d.get('completed') != 20 or d.get('failed', 0):
    raise SystemExit('Expected 20 completed requests without failures')
a, b = min(starts), max(t + l for t, l in zip(starts, latencies))
with trace_path.open() as f:
    all_rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
if not all_rows:
    raise SystemExit('Empty trace; flush/close server first')
if any(y['timestamp'] < x['timestamp'] for x,y in zip(all_rows, all_rows[1:])):
    raise SystemExit('Non-monotonic trace')
rows = [r for r in all_rows if a <= r['timestamp'] <= b]
if not rows:
    raise SystemExit('No matching trace window; check host clock and matching trace')
summary = {k: d.get(k) for k in ['completed', 'request_throughput', 'mean_ttft_ms', 'p95_ttft_ms', 'mean_e2el_ms', 'p95_e2el_ms']}
summary.update(steps=len(rows), peak_running=max(r['running_after'] for r in rows), peak_kv=max(r['kv_cache_usage'] for r in rows))
for k in ['kv_fail_running', 'kv_fail_waiting', 'preemptions', 'input_exhausted_with_waiting', 'token_exhausted_with_waiting']:
    summary[k] = sum(r[k] for r in rows)
summary['trace_end_gap_seconds'] = max(0, b-all_rows[-1]['timestamp'])
# A scheduler timestamp can precede client completion; report, do not silently call a partial trace complete.
summary['trace_coverage'] = 'covers_client_window' if all_rows[0]['timestamp'] <= a and all_rows[-1]['timestamp'] >= b else 'boundary_not_fully_covered_check_server_flush'
summary['resource_confound_observed'] = bool(summary['kv_fail_running'] or summary['kv_fail_waiting'] or summary['preemptions'] or summary['peak_running'] >= 64)
summary['interpretation'] = 'Order sensitivity only; request-level allocation trace and a policy comparison are needed to establish avoidable HOL.'
result_path.with_name(result_path.stem + '_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
