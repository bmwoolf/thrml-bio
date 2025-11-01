"""
Gaussian energy-based model for continuous gene expression data

Implements a quadratic energy function E(x|p) with learned mean mu(p)
and symmetric coupling matrix J 

Uses Langevin sampling for inference

Suitable for z-scored continuous expression data, 
like in the preprocessing script
"""
import flax.linen as nn
import jax.numpy as jnp


# Gaussian energy-based model
class GaussianEBM(nn.Module):
    """
    Gaussian EBM (continuous x)

    energy function:
        E_\theta(x | p) = 1/2 · (x - μ_\theta(p))^T · J · (x - μ_\theta(p))

    where:
        x ∈ ℝ^G               # z-scored gene-expression vector (per gene)
        p                     # condition (target gene, batch, dose/time, …) embedded to f(p)
        μ_\theta(p) = W · f(p)# condition-dependent mean (Dense from cond_dim → G)
        J ∈ ℝ^{G×G}           # learned symmetric coupling (diagonal fixed to 0)

    notes:
        • J is parameterized unconstrained then symmetrized in-call: J ← ½(J + J^T); diag(J)=0.
        • Setting J=I reduces E to ½‖x-μ(p)‖² (isotropic).
        • sampling for CD uses Langevin: x_{t+1} = x_t - α∇_x E(x_t|p) + √(2α)·η_t.
        • this defines the conditional Boltzmann:  P_\theta(x|p) ∝ exp(-E_\theta(x|p))
    """
    n_genes: int
    cond_dim: int = 64

    @nn.compact
    def __call__(self, x, p_emb):
        # mu(p) = W f(p)
        mu = nn.Dense(self.n_genes)(p_emb)                 # [B,G]
        z = x - mu                                         # [B,G]
        # J: unconstrained param -> symmetrize & zero diag in-call
        J = self.param('J', nn.initializers.zeros, (self.n_genes, self.n_genes))
        J = 0.5 * (J + J.T)
        J = J - jnp.diag(jnp.diag(J))
        Jz = jnp.matmul(z, J)                               # [B,G]
        E = 0.5 * jnp.sum(z * Jz, axis=-1)                 # [B]
        return E
