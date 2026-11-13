import pandas as pd

df = pd.read_csv("/tmp/fcfs_rps4_steps.csv")

waiting = df[df["waiting_after"] > 0]

print("Waiting Steps:", len(waiting))

print("\n=== Waiting Steps Only ===")
print(waiting[[
    "running_after",
    "waiting_after",
    "token_budget_remaining",
    "input_budget_remaining",
    "kv_cache_usage",
    "scheduled_requests"
]].describe())

print("\n=== Budget Remaining ===")
print("Min:", waiting["token_budget_remaining"].min())
print("Median:", waiting["token_budget_remaining"].median())
print("Max:", waiting["token_budget_remaining"].max())

print("\n=== High Budget Usage While Waiting ===")
print("Remaining <= 64:",
      (waiting["token_budget_remaining"] <= 64).sum())

print("Remaining <= 128:",
      (waiting["token_budget_remaining"] <= 128).sum())

print("Remaining <= 256:",
      (waiting["token_budget_remaining"] <= 256).sum())