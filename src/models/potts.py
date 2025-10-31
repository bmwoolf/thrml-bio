"""
Potts energy-based model for discrete gene expression

Implements a Potts energy function for ternary states {-1, 0, 1},
representing down-regulated, unchanged, and up-regulated genes respectively

Uses block Gibbs sampling for inference on ternarized data
"""
import flax.linen as nn
import jax.numpy as jnp


# Potts energy-based model
class PottsEBM(nn.Module):
    """
    Potts EBM (discrete x_i ∈ {-1, 0, +1}).

    energy function:
        E_\theta(x | p) = ∑_i h_i^{(p)} x_i  +  1/2 · ∑_{i≠j} J_{ij} x_i x_j

    where:
        x ∈ {-1,0,+1}^G        # ternary gene state per gene (down/neutral/up)
        p                      # condition embedded to f(p)
        h^{(p)} = W · f(p)     # condition-dependent local fields (Dense: cond_dim → G)
        J ∈ ℝ^{G×G}            # learned symmetric pairwise couplings, diag(J)=0

    notes:
        • we use the convention above (linear + ½ pairwise term)
        • J is parameterized unconstrained then symmetrized: J ← ½(J + J^T); diag(J)=0.
        • sampling for CD uses block Gibbs/coordinate updates based on the local field:
              ℓ = h^{(p)} + Jx         # [B,G]
          and picks x_i ∈ {-1,0,+1} that minimizes the local contribution
        • defines P_\theta(x|p) ∝ exp(-E_\theta(x|p))
    """
    n_genes: int
    cond_dim: int = 64

    @nn.compact
    def __call__(self, x_ternary, p_emb):
        x = x_ternary.astype(jnp.float32)                  # {-1,0,1}
        h_p = nn.Dense(self.n_genes)(p_emb)                # [B,G]
        J = self.param('J', nn.initializers.zeros, (self.n_genes, self.n_genes))
        J = 0.5 * (J + J.T)
        J = J - jnp.diag(jnp.diag(J))
        pair = jnp.matmul(x, J)                            # [B,G]
        E = jnp.sum(h_p * x, axis=-1) + 0.5 * jnp.sum(x * pair, axis=-1)
        return E
