"""
Data comparison tool- more AI generated than actual just for
terminal outputs

Displays side-by-side comparisons of original h5ad data with preprocessed
tensors for Potts models 

Shows transformation steps, value distributions, 
and numerical stability improvements
"""
import torch
import json
import scanpy as sc
import numpy as np

from pathlib import Path


# show comparison
def show_comparison():
    """Show samples comparing original and preprocessed data."""
    
    print("\n" + "="*80)
    print("COMPARISON: ORIGINAL vs PREPROCESSED DATA")
    print("="*80 + "\n")
    
    # Load original data (small sample)
    print("1. ORIGINAL DATA (from H5AD file):")
    print("-" * 80)
    adata = sc.read_h5ad("/home/bradley-woolf/Desktop/data/arc/vcc_data/adata_Training.h5ad", backed='r')
    
    print(f"   Shape: {adata.shape} (cells ? genes)")
    print(f"   Total genes: {adata.n_vars:,}")
    print(f"   Data type: {adata.X.dtype}")
    print(f"   Matrix format: {'Sparse CSR' if hasattr(adata.X, 'format') else 'Dense'}")
    
    # Sample first 5 cells, first 10 genes
    print(f"\n   Sample: First 5 cells, first 10 genes:")
    sample_slice = adata.X[:5, :10]
    if hasattr(sample_slice, 'toarray'):
        sample_original = np.array(sample_slice.toarray())
    else:
        sample_original = np.array(sample_slice)
    print(f"   {sample_original}")
    print(f"   Min: {sample_original.min():.4f}, Max: {sample_original.max():.4f}")
    print(f"   Mean: {sample_original.mean():.4f}, Std: {sample_original.std():.4f}")
    
    # Show metadata
    print(f"\n   Cell metadata columns: {list(adata.obs.columns)}")
    print(f"   First 5 cells metadata:")
    print(adata.obs.head())
    
    print(f"\n   Gene names (first 10): {adata.var_names[:10].tolist()}")
    
    # Keep adata open for later use - will close at end
    n_genes_total = adata.n_vars
    
    # Load preprocessed Potts
    print(f"\n\n2. PREPROCESSED POTTS DATA:")
    print("-" * 80)
    tensors_p = torch.load("artifacts/potts/tensors.pt")
    conditions_p = torch.load("artifacts/potts/conditions.pt")
    vocab_p = json.loads(Path("artifacts/potts/vocab.json").read_text())
    
    print(f"   Shape: {tensors_p['X'].shape} (cells ? genes)")
    print(f"   Total genes: {tensors_p['X'].shape[1]:,} (reduced from {n_genes_total:,})")
    print(f"   Data type: {tensors_p['X'].dtype}")
    print(f"   Values: Discrete int8 {-1, 0, 1} (ternarized)")
    
    print(f"\n   Same cells: First 5 cells, first 10 genes:")
    print(f"   {tensors_p['X'][:5, :10]}")
    print(f"   Unique values: {torch.unique(tensors_p['X']).tolist()}")
    print(f"   Value distribution:")
    unique, counts = torch.unique(tensors_p['X'], return_counts=True)
    for val, count in zip(unique, counts):
        pct = (count / tensors_p['X'].numel() * 100).item()
        print(f"     {val.item():3d}: {count.item():,} ({pct:.2f}%)")
    
    print(f"\n   Condition encodings (first 5 cells):")
    for i in range(5):
        target_idx = conditions_p['target_id'][i].item()
        batch_idx = conditions_p['batch_id'][i].item()
        target_name = [k for k, v in vocab_p['target_vocab'].items() if v == target_idx][0]
        batch_name = [k for k, v in vocab_p['batch_vocab'].items() if v == batch_idx][0]
        print(f"     Cell {i}: target='{target_name}' (ID:{target_idx}), batch='{batch_name}' (ID:{batch_idx})")
    
    # Side-by-side comparison
    print(f"\n\n3. SIDE-BY-SIDE COMPARISON (First cell, first 10 genes):")
    print("-" * 80)
    print(f"   {'Gene':<6} {'Original':<12} {'Potts':<10}")
    print(f"   {'-'*6} {'-'*12} {'-'*10}")
    
    # Note: The genes are different (HVGs vs all genes), so we can't directly compare
    # But we can show the structure
    print(f"\n   Original (first 10 genes):")
    # Reload for this section since we closed it earlier
    adata2 = sc.read_h5ad("/home/bradley-woolf/Desktop/data/arc/vcc_data/adata_Training.h5ad", backed='r')
    orig_slice = adata2.X[0, :10]
    if hasattr(orig_slice, 'toarray'):
        orig_row = np.array(orig_slice.toarray()).flatten()
    else:
        orig_row = np.array(orig_slice).flatten()
    for i, val in enumerate(orig_row):
        print(f"     Gene {i}: {val:.4f}")
    adata2.file.close()
    
    print(f"\n   Potts (first 10 HVGs):")
    potts_row = tensors_p['X'][0, :10]
    for i, val in enumerate(potts_row):
        print(f"     HVG {i}: {val.item():3d}")
    
    # Show what preprocessing did
    print(f"\n\n4. PREPROCESSING TRANSFORMATION:")
    print("-" * 80)
    print(f"   Steps applied:")
    print(f"     1. Selected {tensors_p['X'].shape[1]:,} highly variable genes (from {n_genes_total:,} total)")
    print(f"     2. Normalized to 10K reads per cell")
    print(f"     3. Log(x+1) transformation")
    print(f"     4. Z-scored (mean=0, std=1) using train set statistics")
    print(f"     5. ? Filtered low-variance genes (std < 1e-6) - prevents exploding z-scores")
    print(f"     6. ? Clipped extreme z-scores to [-8, 8] - prevents numerical instability")
    print(f"     7. Potts: Ternarized to {{-1, 0, 1}} where |z| > 0.8 ? ?1, else 0")
    print(f"\n   Normalization parameters (mu, sd):")
    print(f"     mu shape: {tensors_p['mu'].shape}, sample: {tensors_p['mu'][:5].tolist()}")
    print(f"     sd shape: {tensors_p['sd'].shape}, sample: {tensors_p['sd'][:5].tolist()}")
    
    # Close the backed file
    if hasattr(adata, 'file') and adata.file is not None:
        adata.file.close()
    
    print("\n" + "="*80)

if __name__ == "__main__":
    show_comparison()
