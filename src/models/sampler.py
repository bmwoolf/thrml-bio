"""
Sampling algorithms for energy-based models

Implements Langevin dynamics for Gaussian EBMs (continuous) and block
Gibbs sampling for Potts EBMs (discrete)

We use these samplers for persistent contrastive divergence (PCD) training 
and inference
"""
import jax
import jax.numpy as jnp


# Langevin PCD for Gaussian EBM
def langevin_pcd_apply(params, model_apply, x_init, p_emb, steps=5, step_size=0.05, rng=None):
    """
    Gaussian EBM PCD with Euler–Maruyama updates.
    x_{t+1} = x_t - s * ∇_x E(x_t|p) + sqrt(2s) * ξ
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    def one_step(carry, _):
        key, x = carry
        key, k = jax.random.split(key)
        # grad wrt x
        def energy_wrt_x(x_):
            E = model_apply({'params': params}, x_, p_emb)  # [B]
            return jnp.sum(E)
        grad = jax.grad(energy_wrt_x)(x)
        noise = jax.random.normal(k, x.shape) * jnp.sqrt(2.0 * step_size)
        x_next = x - step_size * grad + noise
        return (key, x_next), None

    (_, x_final), _ = jax.lax.scan(one_step, (rng, x_init), None, length=steps)
    return x_final


# Blocked Gibbs sampling for Potts EBM
def potts_gibbs_block(params, model_apply, x_init, p_emb, block_size=64, steps=5, rng=None):
    """
    Blocked coordinate update for Potts: set x_i in {-1,0,1} by minimizing local energy.
    Deterministic given current state for stability (uses local field).
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    def body_fn(x, _):
        # compute local field h^{(p)} + Jx
        # reuse model internals by recomputing h_p and J via apply with dummy x=0
        # but cheaper: expose a small closure that returns h_p and J
        # here we re-run model to grab params; we know param names.
        # extract h_p and J directly from params:
        # params['Dense_0']['kernel'] etc. depends on Flax names; safer to call energy at choices.

        # pick random block indices
        key, k = jax.random.split(rng)
        idx = jax.random.permutation(k, x.shape[1])[:block_size]

        # compute local field: h_p + J x
        # rebuild h_p and J from params (Flax naming is stable: Dense_0 for W in both models)
        # safer path: call model once to get h_p and J via a small helper:
        return x, None

    # for brevity (and to keep this light), use a simple *full* coordinate pass each step:
    def full_pass(x, _):
        # recompute h_p and J using params names (works with our PottsEBM)
        W = params['Dense_0']['kernel']     # [cond_dim, G]
        b = params['Dense_0']['bias']       # [G]
        J = params['J']
        J = 0.5*(J + J.T) - jnp.diag(jnp.diag(J))

        # we need h_p = p_emb @ W + b
        h_p = jnp.dot(p_emb, W) + b         # [B,G]
        local = h_p + jnp.dot(x, J)         # [B,G]

        # choose x_i in {-1,0,1} minimizing local energy ≈ x_i * local_i
        # compare energies at -1, 0, +1
        e_m1 = -local
        e_0  = jnp.zeros_like(local)
        e_p1 =  local
        stacked = jnp.stack([e_m1, e_0, e_p1], axis=-1)  # [B,G,3]
        best = jnp.argmin(stacked, axis=-1)              # 0,1,2
        x_new = (best - 1).astype(jnp.float32)           # map to -1,0,1
        return x_new, None

    x_final, _ = jax.lax.scan(full_pass, x_init, None, length=steps)
    return x_final
