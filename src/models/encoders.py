"""
Perturbation condition encoder for gene expression models

Embeds categorical conditions (target gene, batch) and optional numeric
features (dose, time) into a fixed-dimensional vector for our models
"""
import torch
import torch.nn as nn
import jax.numpy as jnp
import flax.linen as fnn


# perturbation condition encoder
class PerturbEncoder(nn.Module):
    """
    f(p): embeds categorical conditions (target, batch) and numeric (dose, time)
    into a fixed-dim vector for conditioning energy models + MLP baseline
    """
    def __init__(
        self,
        n_targets: int,
        n_batches: int,
        emb_dim_target: int = 64,
        emb_dim_batch: int = 16,
        out_dim: int = 64,
        use_numeric: bool = False,
    ):
        super().__init__()
        self.use_numeric = use_numeric
        self.target_emb = nn.Embedding(n_targets, emb_dim_target)
        self.batch_emb = nn.Embedding(max(n_batches, 1), emb_dim_batch)

        in_dim = emb_dim_target + emb_dim_batch + (2 if use_numeric else 0)
        self.project = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

        nn.init.normal_(self.target_emb.weight, std=0.02)
        nn.init.normal_(self.batch_emb.weight, std=0.02)

    # target_id: embedding vector for target gene
    # batch_id: embedding vector for batch
    def forward(self, target_id, batch_id, dose=None, time=None):
        e = [self.target_emb(target_id), self.batch_emb(batch_id)]
        if self.use_numeric:
            assert dose is not None and time is not None
            e.append(torch.stack([dose, time], dim=1))
        h = torch.cat(e, dim=1)
        return self.project(h)  # [B, out_dim]


# JAX/Flax version of PerturbEncoder
class PerturbEncoderJAX(fnn.Module):
    """
    JAX/Flax version of perturbation condition encoder
    
    Embeds categorical conditions (target gene, batch) into a fixed-dimensional vector
    """
    n_targets: int
    n_batches: int
    emb_dim_target: int = 64
    emb_dim_batch: int = 16
    out_dim: int = 64
    
    @fnn.compact
    def __call__(self, target_id, batch_id):
        # embeddings
        target_emb = fnn.Embed(self.n_targets, self.emb_dim_target)(target_id)
        batch_emb = fnn.Embed(max(self.n_batches, 1), self.emb_dim_batch)(batch_id)
        
        # concatenate embeddings
        h = jnp.concatenate([target_emb, batch_emb], axis=-1)
        
        # project to output dimension
        h = fnn.Dense(128)(h)
        h = fnn.gelu(h)
        h = fnn.Dense(self.out_dim)(h)
        
        return h
