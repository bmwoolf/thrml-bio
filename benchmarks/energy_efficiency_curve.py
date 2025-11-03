from __future__ import annotations
"""
Plots model performance (MSE or PCC) vs runtime or energy

Used to compare the physical efficiency of GPU vs thrml when
reaching equivalent performance levels
"""

import argparse, csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_metrics(path: Path):
    data = []
    if not path.exists(): return data
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row = {k: float(v) if v not in ("", None) else np.nan for k, v in row.items()}
            except Exception:
                continue
            data.append(row)
    return data

def discover(benchmarks_dir: Path):
    for d in sorted(p for p in benchmarks_dir.iterdir() if p.is_dir()):
        m = d / "metrics.csv"
        if m.exists():
            yield d.name, d, load_metrics(m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="benchmarks", type=str)
    ap.add_argument("--outdir", default="reports/figures", type=str)
    ap.add_argument("--metric", choices=["mse", "pcc"], default="mse",
                    help="Quality axis uses best validation MSE (lower) or PCC (higher).")
    ap.add_argument("--x", choices=["runtime", "energy"], default="runtime",
                    help="X-axis: total wall-clock seconds or total joules (if available).")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    xs, ys, labels = [], [], []
    for name, d, rows in discover(Path(args.benchmarks)):
        if not rows: continue
        # Aggregate
        total_time = np.nansum([r.get("wall_clock_s", np.nan) for r in rows])
        total_energy = np.nansum([r.get("joules", np.nan) for r in rows])
        best_mse = np.nanmin([r.get("val_mse", np.nan) for r in rows])
        best_pcc = np.nanmax([r.get("val_pcc", np.nan) for r in rows])

        if args.metric == "mse":
            y = best_mse
            ylabel = "Best Validation MSE ↓"
        else:
            y = best_pcc
            ylabel = "Best Validation PCC ↑"

        if args.x == "energy":
            x = total_energy if np.isfinite(total_energy) and total_energy > 0 else np.nan
            xlabel = "Total Energy (J)"
        else:
            x = total_time
            xlabel = "Total Runtime (s)"

        if np.isfinite(x) and np.isfinite(y):
            xs.append(x); ys.append(y); labels.append(name)

    # plot
    plt.figure(figsize=(7,5))
    plt.scatter(xs, ys, s=55)
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"Performance vs {args.x.capitalize()}")
    plt.grid(alpha=0.2)
    if args.metric == "mse":
        plt.gca().invert_yaxis()  # lower is better → show best lower left
    plt.tight_layout()
    out = outdir / f"energy_efficiency_{args.metric}_vs_{args.x}.png"
    plt.savefig(out, dpi=180)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
