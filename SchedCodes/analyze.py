
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULT_DIR = Path("results")
PLOT_DIR = RESULT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULT_DIR / "summary.csv")

metrics = {
    "throughput_tok_s": "Output Throughput (tokens/s)",
    "mean_ttft_ms": "Mean TTFT (ms)",
    "p99_ttft_ms": "P99 TTFT (ms)",
    "mean_tpot_ms": "Mean TPOT (ms)",
    "p99_tpot_ms": "P99 TPOT (ms)",
    "acceptance_rate": "Draft Acceptance Rate",
    "preemptions": "Preemption Count",
}

summary = df.groupby("rate").agg(
    ["mean", "std"]
)

for metric, ylabel in metrics.items():
    grouped = df.groupby("rate")[metric]

    mean = grouped.mean()
    std = grouped.std().fillna(0)

    if mean.isna().all():
        print(f"Skipping {metric}: no recorded data.")
        continue

    plt.figure(figsize=(7, 4.5))

    plt.errorbar(
        mean.index,
        mean.values,
        yerr=std.values,
        marker="o",
        capsize=4,
    )

    plt.xlabel("Request Rate (req/s)")
    plt.ylabel(ylabel)
    plt.title(f"Fixed K=4: {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted(df["rate"].unique()))

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{metric}.png", dpi=160)
    plt.close()

    print(f"\n{metric}")
    print(
        pd.DataFrame({
            "mean": mean,
            "std": std,
        })
    )

print(f"\nGraphs saved to {PLOT_DIR}")
