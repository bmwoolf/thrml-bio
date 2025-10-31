import torch
import torch.nn as nn


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
