#!/usr/bin/env python3
"""Analyze measured-window trace only; never infer HOL causality from counters."""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_id')
args = parser.parse_args()
if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
    parser.error('Invalid run ID')
p = Path(__file__).resolve().parent / 'runs' / args.run_id
d = json.loads((p / 'result.json').read_text())
starts, latencies = d['start_times'], d['latencies']
if len(starts) != 20 or len(latencies) != 20 or d.get('completed') != 20 or d.get('failed', 0):
    raise SystemExit('Expected 20 completed requests without failures')
a, b = min(starts), max(t + l for t, l in zip(starts, latencies))
with (p / 'fcfs_steps.csv').open() as f:
    all_rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
if not all_rows:
    raise SystemExit('Empty trace; flush/close server first')
if any(y['timestamp'] < x['timestamp'] for x,y in zip(all_rows, all_rows[1:])):
    raise SystemExit('Non-monotonic trace')
rows = [r for r in all_rows if a <= r['timestamp'] <= b]
if not rows:
    raise SystemExit('No matching trace window; check host clock and RUN_ID')
summary = {k: d.get(k) for k in ['completed', 'request_throughput', 'mean_ttft_ms', 'p95_ttft_ms', 'mean_e2el_ms', 'p95_e2el_ms']}
summary.update(steps=len(rows), peak_running=max(r['running_after'] for r in rows), peak_kv=max(r['kv_cache_usage'] for r in rows))
for k in ['kv_fail_running', 'kv_fail_waiting', 'preemptions', 'input_exhausted_with_waiting', 'token_exhausted_with_waiting']:
    summary[k] = sum(r[k] for r in rows)
summary['trace_end_gap_seconds'] = max(0, b-all_rows[-1]['timestamp'])
# A scheduler timestamp can precede client completion; report, do not silently call a partial trace complete.
summary['trace_coverage'] = 'covers_client_window' if all_rows[0]['timestamp'] <= a and all_rows[-1]['timestamp'] >= b else 'boundary_not_fully_covered_check_server_flush'
summary['resource_confound_observed'] = bool(summary['kv_fail_running'] or summary['kv_fail_waiting'] or summary['preemptions'] or summary['peak_running'] >= 64)
summary['interpretation'] = 'Order sensitivity only; request-level allocation trace and a policy comparison are needed to establish avoidable HOL.'
(p/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
