"""
Data preprocessing pipeline for perturbation experiments

Converts h5ad single-cell RNA-seq data to model-ready tensors

Handles HVG selection, normalization, z-scoring, and ternarization
for Potts models 

Creates train/val/test splits and encodes perturbation conditions
"""
import argparse, json
import numpy as np
import torch
import scanpy as sc
import anndata as ad
import pandas as pd

from pathlib import Path


# select highly variable genes
def select_hvgs(h5ad_path, batch_key="batch", n_top=2000):
    adata = sc.read_h5ad(h5ad_path)  # small full read once for HVG stats
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", n_top_genes=n_top, batch_key=batch_key
    )
    hvgs = adata.var_names[adata.var["highly_variable"]].to_list()
    return hvgs


# build train/val/test splits
def build_splits(obs, key="target_gene", seed=1337, val_frac=0.1, test_frac=0.1):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(obs[key].astype(str).unique()))
    rng.shuffle(uniq)
    n = len(uniq); n_test = max(1, int(n*test_frac)); n_val = max(1, int(n*val_frac))
    test_keys = set(uniq[:n_test]); val_keys = set(uniq[n_test:n_test+n_val])
    mask_test = obs[key].astype(str).isin(test_keys).values
    mask_val  = obs[key].astype(str).isin(val_keys).values
    mask_train= ~(mask_test | mask_val)
    return {
        "train_idx": np.where(mask_train)[0].tolist(),
        "val_idx":   np.where(mask_val)[0].tolist(),
        "test_idx":  np.where(mask_test)[0].tolist(),
        "heldout_key": key,
        "val_keys": sorted(list(val_keys)),
        "test_keys": sorted(list(test_keys)),
        "seed": seed,
    }


# z-score normalization
def zscore_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd==0] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


# apply z-score normalization
def zscore_apply(X, mu, sd):
    return (X - mu) / sd


# ternarize for Potts model
def ternarize(Z, tau=0.8):
    Xp = np.zeros_like(Z, dtype=np.int8)
    Xp[Z >  tau]  =  1
    Xp[Z < -tau]  = -1
    return Xp


# main function
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)                  # path/to/arc.h5ad
    ap.add_argument("--outdir", required=True)                 # artifacts/potts
    ap.add_argument("--n_hvg", type=int, default=2000)
    ap.add_argument("--batch_key", default="batch")
    ap.add_argument("--target_key", default="target_gene")
    ap.add_argument("--tau", type=float, default=0.8)          # Potts threshold
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # 1) HVGs (full read once for stats; keeps memory reasonable)
    hvgs = select_hvgs(args.input, batch_key=args.batch_key, n_top=args.n_hvg)

    # 2) backed read for big matrix, then subset to HVGs → densify safely
    adata_b = sc.read_h5ad(args.input, backed="r")
    obs = adata_b.obs.copy()
    var_names = adata_b.var_names
    hvg_mask = var_names.isin(hvgs)
    if hasattr(hvg_mask, 'values'):
        hvg_mask = hvg_mask.values
    # backed slicing -> materialize to new in-memory AnnData for HVGs only
    adata = ad.read_h5ad(args.input)[:, hvg_mask].copy()
    # normalize/log on HVGs only (fast)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 3) splits (by target)
    splits = build_splits(obs, key=args.target_key, seed=args.seed)

    # 4) z-score: fit on train only
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    train_idx = np.array(splits["train_idx"], dtype=np.int64)
    mu, sd = zscore_fit(X[train_idx])
    Z = zscore_apply(X, mu, sd).astype(np.float32)
    
    # drop any genes with near-zero std to avoid exploding z-scores
    keep_mask = sd > 1e-6
    Z = Z[:, keep_mask]
    mu = mu[keep_mask]
    sd = sd[keep_mask]
    print(f"Filtered {np.sum(~keep_mask)} low-variance genes")
    
    # clip extreme z-scores to [-8, 8] for numerical stability
    Z = np.clip(Z, -8, 8)

    # 5) Ternarize for Potts model
    Xout = ternarize(Z, tau=args.tau)  # int8 {-1,0,1}
    x_dtype = "int8"

    # 6) conditions (encodings for f(p))
    df = pd.DataFrame({
        "target": obs[args.target_key].astype(str).values,
        "batch":  obs[args.batch_key].astype(str).values if args.batch_key in obs.columns else "NA",
    })
    # (extend later if we add dose/time/cell type)
    cats = {
        "target_vocab": {v:i for i,v in enumerate(sorted(df["target"].unique()))},
        "batch_vocab":  {v:i for i,v in enumerate(sorted(df["batch"].unique()))},
    }
    target_id = df["target"].map(cats["target_vocab"]).astype(np.int64).values
    batch_id  = df["batch"].map(cats["batch_vocab"]).astype(np.int64).values

    # 7) save artifacts
    torch.save(
        {"X": torch.from_numpy(Xout), "dtype": x_dtype, "mu": torch.from_numpy(mu), "sd": torch.from_numpy(sd)},
        out / "tensors.pt",
    )
    torch.save(
        {"target_id": torch.from_numpy(target_id), "batch_id": torch.from_numpy(batch_id)},
        out / "conditions.pt",
    )
    (out / "vocab.json").write_text(json.dumps(cats, indent=2))
    (out / "splits.json").write_text(json.dumps(splits, indent=2))
    (out / "preprocess_meta.json").write_text(json.dumps({
        "mode": "potts",
        "n_cells": int(Xout.shape[0]),
        "n_genes": int(Xout.shape[1]),
        "hvg_count": len(hvgs),
        "heldout_key": args.target_key,
        "batch_key": args.batch_key,
        "x_dtype": x_dtype
    }, indent=2))

    print(f"✔ Preprocessed → {out} | shape: {Xout.shape} | mode=potts")

if __name__ == "__main__":
    main()
