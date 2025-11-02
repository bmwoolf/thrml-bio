"""
visualize gene expression patterns as 3D surface

creates 3D surface plot showing expression values across samples and genes
downsamples for visualization, uses colormap for expression levels
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_expression_data(artifacts_dir, split='test', max_samples=400, max_genes=400):
    """
    load gene expression data from artifacts
    
    downsamples to max_samples x max_genes for visualization
    """
    artifacts_dir = Path(artifacts_dir)
    
    # load tensors and splits
    tensors = torch.load(artifacts_dir / "tensors.pt")
    splits = json.loads((artifacts_dir / "splits.json").read_text())
    
    X = tensors["X"].numpy().astype(np.float32)  # convert from int8 to float
    
    # get split indices
    if split == 'train':
        idx = np.array(splits["train_idx"])
    elif split == 'val':
        idx = np.array(splits["val_idx"])
    else:  # test
        idx = np.array(splits["test_idx"])
    
    X_split = X[idx]
    
    # downsample if needed
    n_samples, n_genes = X_split.shape
    
    if n_samples > max_samples:
        sample_idx = np.linspace(0, n_samples-1, max_samples, dtype=int)
        X_split = X_split[sample_idx]
    
    if n_genes > max_genes:
        gene_idx = np.linspace(0, n_genes-1, max_genes, dtype=int)
        X_split = X_split[:, gene_idx]
    
    return X_split


def plot_expression_surface(X, output_path, split_name='test'):
    """create 3D surface plot of gene expression matrix"""
    n_samples, n_genes = X.shape
    
    # create meshgrid
    samples = np.arange(n_samples)
    genes = np.arange(n_genes)
    S, G = np.meshgrid(samples, genes)
    
    # transpose X for meshgrid compatibility
    Z = X.T
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # surface plot with colormap
    surf = ax.plot_surface(S, G, Z, cmap='coolwarm', 
                          edgecolor='none', alpha=0.9,
                          linewidth=0, antialiased=True,
                          vmin=-1, vmax=1)
    
    # labels
    ax.set_xlabel('Samples', fontsize=11)
    ax.set_ylabel('Genes', fontsize=11)
    ax.set_zlabel('Expression (ternary)', fontsize=11)
    ax.set_title(f'Gene Expression Patterns ({split_name} set)', fontsize=13, pad=20)
    
    # colorbar
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.set_label('Expression', rotation=270, labelpad=15)
    
    # viewing angle
    ax.view_init(elev=25, azim=45)
    
    # set z limits for ternary data
    ax.set_zlim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"saved gene expression surface to {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/potts", 
                    help="path to preprocessed artifacts directory")
    ap.add_argument("--split", choices=['train', 'val', 'test'], default='test',
                    help="which split to visualize")
    ap.add_argument("--outdir", default="reports/figures", type=str)
    ap.add_argument("--max_samples", type=int, default=400,
                    help="max samples to plot (downsamples if larger)")
    ap.add_argument("--max_genes", type=int, default=400,
                    help="max genes to plot (downsamples if larger)")
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # load data
    print(f"loading {args.split} expression data from {args.artifacts}")
    X = load_expression_data(args.artifacts, args.split, 
                            args.max_samples, args.max_genes)
    print(f"plotting {X.shape[0]} samples x {X.shape[1]} genes")
    
    # plot
    output_path = outdir / f"gene_expression_surface_{args.split}.png"
    plot_expression_surface(X, output_path, args.split)


if __name__ == "__main__":
    main()

