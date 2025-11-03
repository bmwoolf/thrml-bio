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
from scipy.interpolate import griddata

from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


def load_potts_model(checkpoint_path):
    """load trained Potts model checkpoint"""
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    return ckpt


def compute_energy_landscape(model_params, n_genes, grid_size=100, pca_components=2, rng_key=None):
    """
    compute energy landscape over 2D state space projection
    
    samples actual states from model and projects to PC space
    uses PCA on sampled states to find principal directions
    """
    import jax.random as jr
    
    if rng_key is None:
        rng_key = jr.PRNGKey(42)
    
    # extract J coupling matrix
    J = model_params['J']
    J = 0.5 * (J + J.T) - jnp.diag(jnp.diag(J))
    
    # sample diverse states efficiently using vectorized operations
    n_samples = grid_size * grid_size
    
    # generate random initializations in batch
    keys = jr.split(rng_key, n_samples)
    x_init_batch = jnp.stack([
        jr.choice(keys[i], jnp.array([-1, 0, 1]), shape=(n_genes,))
        for i in range(n_samples)
    ])
    
    # vectorized Gibbs updates (simplified - just a few passes)
    def gibbs_step(x_batch):
        """vectorized Gibbs step for all states"""
        # compute local field: J @ x for all states
        h_local = jnp.einsum('ij,nj->ni', J, x_batch)  # [n_samples, n_genes]
        
        # for each gene, pick best state
        x_new = x_batch
        for g in range(n_genes):
            # try all 3 states for gene g
            candidates = jnp.array([-1, 0, 1])
            # compute energy change for each candidate
            delta_energy = jnp.zeros((n_samples, 3))
            for idx, cand in enumerate(candidates):
                x_test = x_new.at[:, g].set(cand)
                # energy = 0.5 * x^T J x
                energies = 0.5 * jnp.einsum('ni,ij,nj->n', x_test, J, x_test)
                delta_energy = delta_energy.at[:, idx].set(energies)
            
            # pick lowest energy state for each sample
            best_idx = jnp.argmin(delta_energy, axis=1)
            x_new = x_new.at[:, g].set(candidates[best_idx])
        
        return x_new
    
    # run several Gibbs steps
    x_batch = x_init_batch
    for _ in range(10):  # more steps for better sampling
        x_batch = gibbs_step(x_batch)
    
    # compute energies for all states
    energies_direct = 0.5 * jnp.einsum('ni,ij,nj->n', x_batch, J, x_batch)
    states = x_batch
    energies_direct = np.array(energies_direct)
    
    # PCA on sampled states to find principal directions
    states_centered = states - jnp.mean(states, axis=0)
    U, S, Vt = jnp.linalg.svd(states_centered, full_matrices=False)
    pc1 = Vt[0]  # first principal component
    pc2 = Vt[1]  # second principal component
    
    # project states to PC space
    pc1_coords = jnp.dot(states_centered, pc1)
    pc2_coords = jnp.dot(states_centered, pc2)
    
    # create grid in PC space
    pc1_range = jnp.linspace(jnp.min(pc1_coords), jnp.max(pc1_coords), grid_size)
    pc2_range = jnp.linspace(jnp.min(pc2_coords), jnp.max(pc2_coords), grid_size)
    X, Y = jnp.meshgrid(pc1_range, pc2_range)
    
    # interpolate energies onto grid using cubic interpolation
    X_flat = np.array(X.flatten())
    Y_flat = np.array(Y.flatten())
    pc1_coords_np = np.array(pc1_coords)
    pc2_coords_np = np.array(pc2_coords)
    energies_np = np.array(energies_direct)
    
    Z = griddata((pc1_coords_np, pc2_coords_np), energies_np, 
                 (X_flat, Y_flat), method='cubic', fill_value=np.nan)
    Z = Z.reshape(X.shape)
    
    # fill NaN values with nearest neighbor
    mask = np.isnan(Z)
    if np.any(mask):
        mask_flat = mask.flatten()
        Z[mask] = griddata((pc1_coords_np, pc2_coords_np), energies_np,
                          (X_flat[mask_flat], Y_flat[mask_flat]), method='nearest')
    
    return np.array(X), np.array(Y), np.array(Z)


def plot_energy_landscape(X, Y, Z, output_path):
    """create 3D surface plot of energy landscape with vibrant colors"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # use vibrant colormap like 'plasma', 'inferno', or 'turbo'
    # 'turbo' gives the purple->blue->green->yellow->red gradient
    surf = ax.plot_surface(X, Y, Z, cmap='turbo', 
                          edgecolor='none', alpha=0.95,
                          linewidth=0, antialiased=True,
                          shade=True)
    
    # labels with better styling
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_zlabel('Energy', fontsize=12, fontweight='bold')
    ax.set_title('Potts EBM Energy Landscape', fontsize=14, fontweight='bold', pad=20)
    
    # colorbar with better styling
    cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Energy', fontsize=11, fontweight='bold')
    
    # grid styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # better viewing angle to show peaks/valleys
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"saved energy landscape to {output_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, 
                    help="path to model checkpoint (eg benchmarks/potts_ebm_jax/model_checkpoint.pkl)")
    ap.add_argument("--outdir", default="reports/figures", type=str)
    ap.add_argument("--grid_size", type=int, default=100, 
                    help="resolution of energy landscape grid (higher = more detailed)")
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

