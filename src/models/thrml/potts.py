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


def thrml_potts_sampler(params, model_apply, x_init, p_emb, block_size=64, steps=5, rng=None):
    """
    sampler compatible with train_epoch_potts
    uses thrml block Gibbs sampling
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    W = params['Dense_0']['kernel']
    b = params['Dense_0']['bias']
    J = params['J']
    J = 0.5 * (J + J.T) - jnp.diag(jnp.diag(J))
    
    batch_size, n_genes = x_init.shape
    nodes, edges = create_potts_graph(n_genes)
    
    # extract weights from J matrix for edges
    weights = jnp.array([J[i, j] for i in range(n_genes) for j in range(i+1, n_genes)])
    
    samples = []
    for i in range(batch_size):
        biases = jnp.dot(p_emb[i], W) + b
        key_i = jax.random.fold_in(rng, i)
        sample = sample_potts_thrml(key_i, nodes, edges, biases, weights, n_steps=steps, block_size=block_size)
        samples.append(sample)
    
    return jnp.stack(samples, axis=0)
