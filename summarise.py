"""Aggregate results.jsonl -> results_summary.csv (mean and std over seeds)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
import json, pandas as pd
rows = [json.loads(l) for l in open(os.path.join(HERE, "results", "results.jsonl"))]
df = pd.DataFrame(rows).drop(columns=["val_curve"])
metrics = [c for c in ["rmse", "mae", "r2", "naive_rmse", "accuracy", "f1", "train_s", "s_per_epoch",
                       "epochs", "best_epoch", "latency_ms", "numpy_latency_ms"] if c in df]
g = df.groupby(["app", "model"], sort=False)
summ = g[metrics].agg(["mean", "std"])
summ.columns = [f"{a}_{b}" for a, b in summ.columns]
fixed = g[["params", "size_kb", "macs", "lookback", "n_features", "n_train", "n_test"]].first()
out = fixed.join(summ).reset_index()
out.to_csv(os.path.join(HERE, "results", "results_summary.csv"), index=False)
pd.set_option("display.width", 250, "display.max_columns", 50)
for app, d in out.groupby("app", sort=False):
    cols = ["model", "params", "size_kb", "macs"] + [c for c in d if any(c.startswith(m) for m in
            ["rmse", "mae", "r2", "naive", "accuracy", "f1", "train_s", "s_per", "epochs", "best", "numpy"])]
    print(f"\n=== {app}"); print(d[cols].dropna(axis=1, how="all").round(4).to_string(index=False))
