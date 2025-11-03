"""
visualize energy landscape of trained Potts EBM

creates 3D surface plot showing energy values across state space
uses PCA to reduce gene dimensions to 2D for visualization
"""
import argparse, pickle
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


def load_potts_model(checkpoint_path):
    """load trained Potts model checkpoint"""
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    return ckpt


def compute_energy_landscape(model_params, n_genes, grid_size=50, pca_components=2):
    """
    compute energy landscape over 2D state space projection
    
    uses PCA on J matrix to find principal directions
    samples states along these directions and computes energy
    """
    # extract J coupling matrix
    J = model_params['J']
    J = 0.5 * (J + J.T) - jnp.diag(jnp.diag(J))
    
    # get principal components of J for visualization
    U, S, Vt = jnp.linalg.svd(J)
    pc1 = Vt[0]  # first principal component
    pc2 = Vt[1]  # second principal component
    
    # create grid in PC space
    x_range = np.linspace(-2, 2, grid_size)
    y_range = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(x_range, y_range)
    
    # compute energies
    energies = np.zeros_like(X)
    
    for i in range(grid_size):
        for j in range(grid_size):
            # project back to state space
            state = X[i,j] * pc1 + Y[i,j] * pc2
            # clip to {-1, 0, 1} for valid Potts states
            state = jnp.clip(jnp.round(state), -1, 1)
            
            # compute energy E = 0.5 * x^T J x
            energy = 0.5 * jnp.dot(state, jnp.dot(J, state))
            energies[i,j] = float(energy)
    
    return X, Y, energies


def plot_energy_landscape(X, Y, Z, output_path):
    """create 3D surface plot of energy landscape"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # surface plot with colormap
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                          edgecolor='none', alpha=0.9,
                          linewidth=0, antialiased=True)
    
    # labels
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_zlabel('Energy', fontsize=11)
    ax.set_title('Potts EBM Energy Landscape', fontsize=13, pad=20)
    
    # colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"saved energy landscape to {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, 
                    help="path to model checkpoint (eg benchmarks/potts_ebm_jax/model_checkpoint.pkl)")
    ap.add_argument("--outdir", default="reports/figures", type=str)
    ap.add_argument("--grid_size", type=int, default=50, 
                    help="resolution of energy landscape grid")
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # load model
    print(f"loading model from {args.checkpoint}")
    ckpt = load_potts_model(args.checkpoint)
    model_params = ckpt['model_params']
    n_genes = ckpt['config']['n_genes']
    
    # compute landscape
    print(f"computing energy landscape (grid_size={args.grid_size})")
    X, Y, Z = compute_energy_landscape(model_params, n_genes, args.grid_size)
    
    # plot
    model_name = Path(args.checkpoint).parent.name
    output_path = outdir / f"energy_landscape_{model_name}.png"
    plot_energy_landscape(X, Y, Z, output_path)


if __name__ == "__main__":
    main()

