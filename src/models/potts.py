import flax.linen as nn
import jax.numpy as jnp

# Potts energy-based model
class PottsEBM(nn.Module):
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
