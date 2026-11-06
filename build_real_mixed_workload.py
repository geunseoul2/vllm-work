#!/usr/bin/env python3
"""Build reproducible, meaningful vLLM CustomDataset requests from Dolly 15K.

Requires: pip install 'transformers>=4.51' huggingface_hub
Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k
Dataset license: CC BY-SA 3.0, attribution required for redistributed samples.

No artificial keyword padding or mid-sentence truncation: samples are filtered by
Qwen tokenizer lengths after chat templating, preserving complete instructions and
reference contexts.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter
from pathlib import Path


def load_source(local_path: str | None) -> list[dict]:
    if local_path:
        path = Path(local_path)
        if not path.exists():
            raise SystemExit(f"Source data missing: {path}")
    else:
        from huggingface_hub import hf_hub_download
        print("Downloading Databricks Dolly 15k, ~13 MB (first run only)...")
        path = Path(hf_hub_download(
            repo_id="databricks/databricks-dolly-15k",
            filename="databricks-dolly-15k.jsonl", repo_type="dataset"))
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def meaningful_prompt(item: dict) -> str:
    instruction = (item.get("instruction") or "").strip()
    context = (item.get("context") or "").strip()
    if not instruction:
        return ""
    if context:
        return ("Use the supplied reference passage when answering the user's request. "
                "If the reference does not contain the answer, state the limitation.\n\n"
                f"Reference passage:\n{context}\n\n"
                f"User request:\n{instruction}")
    return instruction


def selected_candidates(source, tokenizer, seed, target, lo, hi):
    pool = []
    seen = set()
    for idx, item in enumerate(source):
        prompt = meaningful_prompt(item)
        if len(prompt) < 30 or prompt in seen:
            continue
        seen.add(prompt)
        # Match the chat framing that will be sent in vllm --skip-chat-template mode.
        framed = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        tokens = len(tokenizer.encode(framed, add_special_tokens=False))
        if lo <= tokens <= hi and tokens + 128 <= 2048:
            pool.append({
                "source_row": idx,
                "category": str(item.get("category") or "unknown"),
                "prompt": framed,
                "input_tokens": tokens,
                "target_distance": abs(tokens - target),
                "response": str(item.get("response") or ""),
            })
    # Among near-target samples, choose randomly by category for content diversity.
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool.sort(key=lambda r: (r["target_distance"] // 16))
    return pool


def take_diverse(pool, count):
    """Prefer near-target samples while avoiding collapse into a single category."""
    bins = {}
    for entry in pool:
        bins.setdefault(entry["category"], []).append(entry)
    if len(pool) < count:
        raise RuntimeError(f"Need {count} samples; only {len(pool)} qualify. "
                           "Try widening the --short-min/max or --long-min/max ranges.")
    selected = []
    while len(selected) < count and any(bins.values()):
        categories = sorted((k for k, v in bins.items() if v),
                            key=lambda k: (sum(1 for x in selected if x["category"] == k),
                                           bins[k][0]["target_distance"]))
        for cat in categories:
            if len(selected) >= count:
                break
            if bins[cat]:
                selected.append(bins[cat].pop(0))
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-file", default=None,
                    help="Optional local Dolly 15k JSONL, skips network download")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--output-dir", default="real_mixed_workload")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--short-count", type=int, default=160)
    ap.add_argument("--long-count", type=int, default=40)
    ap.add_argument("--short-target", type=int, default=128)
    ap.add_argument("--short-min", type=int, default=95)
    ap.add_argument("--short-max", type=int, default=165)
    ap.add_argument("--long-target", type=int, default=768)
    ap.add_argument("--long-min", type=int, default=600)
    ap.add_argument("--long-max", type=int, default=975)
    ap.add_argument("--fixed-output", type=int, default=128)
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Please install: pip install 'transformers>=4.51' huggingface_hub") from exc

    source = load_source(args.source_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    print(f"Loaded {len(source)} source rows; filtering meaningful complete prompts...")
    def choose(count, target, lo, hi, rng_seed, group):
        for lower, upper in [(lo, hi), (max(64, lo-35), hi+35),
                             (max(64, lo-100), hi+150),
                             (max(64, lo-200), min(1400, hi+300))]:
            pool = selected_candidates(source, tokenizer, rng_seed, target,
                                       lower, upper)
            if len(pool) >= count:
                if (lower, upper) != (lo, hi):
                    print(f"NOTICE: widened {group} token range to "
                          f"{lower}..{upper} due to source data distribution")
                return take_diverse(pool, count)
        raise RuntimeError(f"Cannot find {count} valid {group} requests. "
                           "Try different length targets or use a larger dataset.")

    shorts = choose(args.short_count, args.short_target, args.short_min,
                    args.short_max, args.seed, "short")
    longs = choose(args.long_count, args.long_target, args.long_min,
                   args.long_max, args.seed+1, "long")
    rows = ([dict(entry, group="short") for entry in shorts] +
            [dict(entry, group="long") for entry in longs])
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    def write_variant(name, entries, natural=False):
        fname = out / f"dolly_{name}.jsonl"
        manifest = out / f"dolly_{name}_manifest.csv"
        with fname.open("w", encoding="utf-8") as f, \
             manifest.open("w", encoding="utf-8", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=(
                "request_index", "group", "source_row", "category",
                "input_tokens", "requested_output_tokens"))
            writer.writeheader()
            for i, row in enumerate(entries):
                response_len = len(tokenizer.encode(row["response"],
                                                    add_special_tokens=False))
                output_tokens = (min(max(response_len, 16), 512)
                                 if natural else args.fixed_output)
                f.write(json.dumps({"prompt": row["prompt"],
                                    "output_tokens": output_tokens},
                                   ensure_ascii=False) + "\n")
                writer.writerow({"request_index": i, "group": row["group"],
                                 "source_row": row["source_row"],
                                 "category": row["category"],
                                 "input_tokens": row["input_tokens"],
                                 "requested_output_tokens": output_tokens})
        print(f"Saved: {fname} and {manifest}")

    write_variant("mixed_random_fixed128", rows)
    write_variant("long_first_fixed128", [*([r for r in rows if r["group"] == "long"]),
                                          *([r for r in rows if r["group"] == "short"])])
    write_variant("short_first_fixed128", [*([r for r in rows if r["group"] == "short"]),
                                           *([r for r in rows if r["group"] == "long"])])
    write_variant("mixed_random_natural", rows, natural=True)

    avg = statistics.mean(x["input_tokens"] for x in rows)
    print(f"Created {len(rows)} meaningful requests: {args.short_count} short + {args.long_count} long")
    print(f"Actual mean input: {avg:.1f} tokens (previous uniform baseline: 256)")
    print(f"Short: {min(x['input_tokens'] for x in shorts)}..{max(x['input_tokens'] for x in shorts)} tokens")
    print(f"Long : {min(x['input_tokens'] for x in longs)}..{max(x['input_tokens'] for x in longs)} tokens")
    print(f"Categories: {dict(Counter(x['category'] for x in rows))}")
    print("NOTE: Each manifest order aligns with its JSONL when --disable-shuffle is used.")
    print("SOURCE: Databricks Dolly 15k (human-created), CC BY-SA 3.0: "
          "https://huggingface.co/datasets/databricks/databricks-dolly-15k")

if __name__ == "__main__":
    main()
