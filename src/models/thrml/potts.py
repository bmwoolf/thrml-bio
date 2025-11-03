"""
Potts EBM using thrml for block Gibbs sampling

wraps thrml's IsingEBM and block sampling for efficient discrete sampling
on ternary gene expression states {-1, 0, 1}
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from typing import List, Tuple


class PottsEBMThrml(nn.Module):
    """
    Potts EBM with thrml block Gibbs sampling
    
    energy:
        E(x|p) = sum_i h_i^{(p)} x_i + 1/2 * sum_{i!=j} J_{ij} x_i x_j
        
    uses thrml for efficient block Gibbs sampling of ternary states
    """
    n_genes: int
    cond_dim: int = 64
    
    @nn.compact
    def __call__(self, x_ternary, p_emb):
        """compute energy E(x|p)"""
        x = x_ternary.astype(jnp.float32)
        h_p = nn.Dense(self.n_genes)(p_emb)
        J = self.param('J', nn.initializers.zeros, (self.n_genes, self.n_genes))
        J = 0.5 * (J + J.T)
        J = J - jnp.diag(jnp.diag(J))
        pair = jnp.matmul(x, J)
        E = jnp.sum(h_p * x, axis=-1) + 0.5 * jnp.sum(x * pair, axis=-1)
        return E


def create_potts_graph(n_genes: int):
    """create thrml graph structure for n genes"""
    nodes = [SpinNode() for _ in range(n_genes)]
    # fully connected graph (will use J matrix to determine actual weights)
    edges = [(nodes[i], nodes[j]) for i in range(n_genes) for j in range(i+1, n_genes)]
    return nodes, edges


def sample_potts_thrml(
    key: jax.random.PRNGKey,
    nodes: List[SpinNode],
    edges: List[Tuple[SpinNode, SpinNode]],
    biases: jnp.ndarray,
    weights: jnp.ndarray,
    n_steps: int = 5,
    block_size: int = 64,
):
    """
    sample from Potts using thrml block Gibbs
    
    args:
        key: JAX random key
        nodes: list of SpinNode
        edges: list of (node_i, node_j) tuples
        biases: local fields [n_genes]
        weights: edge weights [n_edges]
        n_steps: Gibbs sampling steps
        block_size: block size for coloring
        
    returns:
        sample [n_genes]
    """
    beta = jnp.array(1.0)
    model = IsingEBM(nodes, edges, biases, weights, beta)
    
    # create blocks for two-coloring
    n_genes = len(nodes)
    if n_genes <= block_size:
        free_blocks = [Block(nodes)]
    else:
        free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    
    k_init, k_samp = jax.random.split(key)
    init_state = hinton_init(k_init, model, free_blocks, ())
    
    schedule = SamplingSchedule(
        n_warmup=n_steps,
        n_samples=1,
        steps_per_sample=1
    )
    
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    return samples[0]


def one_block_update(params, model_apply, x, p_emb, block_idx, key):
    """
    vectorized block update for Potts states
    
    x: [B, G], block_idx: [B, K], q=3 states {-1,0,1}
    """
    cand = jnp.array([-1, 0, 1])  # [q]
    B, G = x.shape
    K = block_idx.shape[1]
    q = 3
    
    # for each position k and candidate c, create state with cand[c] at block_idx[:, k]
    # x_cand[b, k, c, :] = x with block_idx[b, k] set to cand[c]
    def set_candidate_at_pos(pos_idx, cand_val):
        # pos_idx: scalar, cand_val: scalar
        x_cand_kc = x.at[jnp.arange(B), block_idx[:, pos_idx]].set(cand_val)
        return x_cand_kc
    
    # vmap over positions and candidates
    x_cand = jax.vmap(lambda k: jax.vmap(lambda c: set_candidate_at_pos(k, c))(cand))(jnp.arange(K))
    # result: [K, q, B, G], transpose to [B, K, q, G]
    x_cand = jnp.transpose(x_cand, (2, 0, 1, 3))  # [B, K, q, G]
    
    # evaluate energy for all candidates
    x_flat = x_cand.reshape(B * K * q, G)
    p_emb_expanded = jnp.repeat(p_emb[:, None, None, :], repeats=K, axis=1)
    p_emb_expanded = jnp.repeat(p_emb_expanded, repeats=q, axis=2)
    p_emb_flat = p_emb_expanded.reshape(B * K * q, p_emb.shape[1])
    
    E_flat = model_apply({'params': params}, x_flat, p_emb_flat)  # [B*K*q]
    E = E_flat.reshape(B, K, q)
    
    # pick state with lowest energy for each position
    k_choice = jnp.argmin(E, axis=2)  # [B, K]
    
    # scatter choices back to x (match dtype)
    chosen_vals = cand[k_choice].astype(x.dtype)  # [B, K]
    batch_idx = jnp.arange(B)[:, None]
    x_new = x.at[batch_idx, block_idx].set(chosen_vals)
    
    return x_new


def gibbs(params, model_apply, x0, p_emb, block_indices, n_genes, steps, key):
    """vectorized Gibbs sampling using lax.scan (JIT happens at call site)"""
    # block_indices: [n_blocks, block_size] (padded)
    n_blocks = block_indices.shape[0]
    
    def step_fn(x, k):
        key_k = jax.random.fold_in(key, k)
        # choose block k (or cycle)
        block_idx = block_indices[k % n_blocks]
        # mask out padding (indices >= n_genes)
        valid_mask = block_idx < n_genes
        block_idx_valid = jnp.where(valid_mask, block_idx, 0)  # replace invalid with 0
        # broadcast to batch dimension
        batch_size = x.shape[0]
        block_batch = jnp.tile(block_idx_valid[None, :], (batch_size, 1))  # [B, K]
        # only update valid positions
        x = one_block_update(params, model_apply, x, p_emb, block_batch, key_k)
        return x, None
    
    x_final, _ = jax.lax.scan(step_fn, x0, jnp.arange(steps))
    return x_final


def thrml_potts_sampler(params, model_apply, x_init, p_emb, block_size=64, steps=5, rng=None):
    """
    vectorized sampler compatible with train_epoch_potts
    uses pure JAX block Gibbs sampling (GPU-accelerated)
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    batch_size, n_genes = x_init.shape
    
    # create blocks for block updates
    # simple strategy: split genes into blocks of size block_size
    n_blocks = (n_genes + block_size - 1) // block_size
    block_indices = []
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n_genes)
        block_indices.append(jnp.arange(start, end))
    
    # pad to same size and stack [n_blocks, block_size]
    block_indices_padded = jnp.zeros((n_blocks, block_size), dtype=jnp.int32)
    for i, b in enumerate(block_indices):
        block_indices_padded = block_indices_padded.at[i, :len(b)].set(b)
    block_indices = block_indices_padded
    
    # run vectorized Gibbs sampling (JIT happens inside)
    x_final = gibbs(params, model_apply, x_init, p_emb, block_indices, n_genes, steps, rng)
    
    return x_final
