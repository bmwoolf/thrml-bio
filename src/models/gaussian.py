import flax.linen as nn
import jax.numpy as jnp

# Gaussian energy-based model
class GaussianEBM(nn.Module):
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
