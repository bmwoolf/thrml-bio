"""
Conditional MLP baseline predictor for gene expression data

Simple feedforward neural network that predicts gene expression from
perturbation embeddings

Used as a non-EBM baseline for comparison with energy-based models
"""
import torch
import torch.nn as nn


# conditional MLP baseline
class ConditionalMLP(nn.Module):
    def __init__(self, n_genes: int, cond_dim: int = 64, hidden: int = 1024, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = [nn.Linear(cond_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, depth-1)):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, n_genes)]
        self.net = nn.Sequential(*layers)

    def forward(self, p_emb):
        return self.net(p_emb)
