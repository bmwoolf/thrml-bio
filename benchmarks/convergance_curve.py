from __future__ import annotations
"""
Visualizes training convergence for energy-based models.

Plots validation MSE or PCC vs. training epochs for multiple runs to
compare learning speed and stability (GPU vs thrml vs model types)
"""

import argparse, csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_metrics_csv(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            # tolerant parsing
            row = {k: (float(v) if v not in ("", None) else np.nan) for k, v in row.items()}
            rows.append(row)
    return rows

def discover_runs(benchmarks_dir: Path):
    for d in sorted(p for p in benchmarks_dir.iterdir() if p.is_dir()):
        metrics = d / "metrics.csv"
        if metrics.exists():
            yield d.name, d, load_metrics_csv(metrics)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="benchmarks", type=str)
    ap.add_argument("--outdir", default="reports/figures", type=str)
    ap.add_argument("--metric", choices=["mse", "pcc"], default="mse",
                    help="What to plot on Y-axis (validation).")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    runs = list(discover_runs(Path(args.benchmarks)))
    if not runs:
        print("No runs with metrics.csv found.")
        return

    plt.figure(figsize=(7,5))
    for name, d, rows in runs:
        if not rows:
            continue
        epochs = np.arange(1, len(rows)+1)
        if args.metric == "mse":
            y = np.array([r.get("val_mse", np.nan) for r in rows], dtype=float)
            ylabel = "Validation MSE ↓"
        else:
            y = np.array([r.get("val_pcc", np.nan) for r in rows], dtype=float)
            ylabel = "Validation PCC ↑"
        plt.plot(epochs, y, marker="o", linewidth=1.5, label=name)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title("Training Convergence")
    plt.legend(frameon=False)
    plt.grid
