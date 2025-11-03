"""
Computes and visualizes model accuracy metrics on held-out test sets

Compares architectures (Potts/MLP) on quality, independent of hardware 

Shows bars for test MSE and PCC when available
"""

import argparse, json, csv
import numpy as np
import matplotlib.pyplot as plt

from __future__ import annotations
from pathlib import Path

def load_test_metrics(run_dir: Path):
    j = run_dir / "test_metrics.json"
    if j.exists():
        try:
            return json.loads(j.read_text())
        except Exception:
            pass
    # fallback: last row from metrics.csv
    m = run_dir / "metrics.csv"
    if not m.exists(): return {}
    with m.open() as f:
        rows = list(csv.DictReader(f))
    if not rows: return {}
    last = rows[-1]
    out = {}
    if "val_mse" in last: out["mse"] = float(last["val_mse"])
    if "val_pcc" in last: out["pcc"] = float(last["val_pcc"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="benchmarks", type=str)
    ap.add_argument("--outdir", default="reports/figures", type=str)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    runs = sorted([p for p in Path(args.benchmarks).iterdir() if p.is_dir()])

    names, mses, pccs = [], [], []
    for d in runs:
        metrics = load_test_metrics(d)
        if not metrics: continue
        names.append(d.name)
        mses.append(metrics.get("mse", np.nan))
        pccs.append(metrics.get("pcc", np.nan))

    if not names:
        print("No test metrics found.")
        return

    # Two stacked subplots: MSE and PCC
    fig, axes = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    x = np.arange(len(names))

    axes[0].bar(x, mses)
    axes[0].set_ylabel("Test MSE ↓")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x, pccs)
    axes[1].set_ylabel("Test PCC ↑")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Model Accuracy (Held-out Test)")
    plt.tight_layout()
    out = outdir / "model_accuracy.png"
    plt.savefig(out, dpi=180)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
