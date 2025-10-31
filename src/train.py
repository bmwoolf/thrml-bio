import time, json
import torch, torch.nn as nn
import jax, optax, jax.numpy as jnp
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.encoders import PerturbEncoder
from models.gaussian import GaussianEBM
from models.mlp import ConditionalMLP
from models.potts import PottsEBM
from models.sampler import langevin_pcd_apply, potts_gibbs_block


def make_train_state(rng, model, tx, sample_x, sample_p):
    params = model.init(rng, sample_x, sample_p)['params']
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# Gaussian EBM
def train_epoch_gaussian(state, enc_apply, enc_params, model, batch_iter, sampler, steps=5, step_size=0.05):
    @jax.jit
    def loss_fn(params, x_pos, p_emb):
        E_pos = model.apply({'params': params}, x_pos, p_emb)  # [B]
        x_init = x_pos  # PCD init at data (can cache chains)
        x_neg  = sampler(params, model.apply, x_init, p_emb, steps=steps, step_size=step_size)
        E_neg = model.apply({'params': params}, x_neg, p_emb)
        loss = jnp.mean(E_pos) - jnp.mean(E_neg)
        return loss

    @jax.jit
    def step(state, x_pos, p):
        p_emb = enc_apply({'params': enc_params}, **p)
        grads = jax.grad(lambda prm: loss_fn(prm, x_pos, p_emb))(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    for batch in batch_iter:
        x = batch['x']          # [B,G] float32
        p = batch['p']          # dict of ids (target_id, batch_id, ...)
        state = step(state, x, p)
    return state


# Potts EBM
def train_epoch_potts(state, enc_apply, enc_params, model, batch_iter, sampler, steps=5, block_size=64):
    """PCD with block Gibbs for Potts x ∈ {-1,0,1}."""
    @jax.jit
    def loss_fn(params, x_pos, p_emb):
        # x_pos is ternary in {-1,0,1} (int8/float ok)
        E_pos = model.apply({'params': params}, x_pos, p_emb)  # [B]
        x_init = x_pos
        x_neg  = sampler(params, model.apply, x_init, p_emb, block_size=block_size, steps=steps)
        E_neg = model.apply({'params': params}, x_neg, p_emb)
        return jnp.mean(E_pos) - jnp.mean(E_neg)

    @jax.jit
    def step(state, x_pos, p):
        p_emb = enc_apply({'params': enc_params}, **p)
        grads = jax.grad(lambda prm: loss_fn(prm, x_pos, p_emb))(state.params)
        return state.apply_gradients(grads=grads)

    for batch in batch_iter:
        x = batch['x']          # [B,G] values in {-1,0,1}
        p = batch['p']          # dict of ids (target_id, batch_id, ...)
        state = step(state, x, p)
    return state


def predict_mu(enc_apply, enc_params, model_apply, model_params, p):
    p_emb = enc_apply({'params': enc_params}, **p)
    # TODO: for Gaussian: mu = Dense(p_emb)
    # TODO: for reporting PCC/MSE on val/test, call model pieces or add a helper
    return p_emb


# Torch MLP Baseline (non-EBM)
def train_mlp_baseline(artifacts_dir, epochs=30, batch_size=256, lr=1e-3, balance=False):
    """Train a simple Torch MLP predictor using the same condition encoder."""
    import json
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load artifacts
    tensors = torch.load(Path(artifacts_dir) / "tensors.pt")
    conds = torch.load(Path(artifacts_dir) / "conditions.pt")
    vocab = json.loads(Path(artifacts_dir) / "vocab.json").read_text()

    X = tensors["X"].float()
    target_id = conds["target_id"].long()
    batch_id = conds["batch_id"].long()

    # train/val/test splits
    splits = json.loads(Path(artifacts_dir) / "splits.json").read_text()
    train_idx = torch.tensor(splits["train_idx"], dtype=torch.long)
    val_idx = torch.tensor(splits["val_idx"], dtype=torch.long)

    # Dataset class
    class DS(torch.utils.data.Dataset):
        def __init__(self, idx):
            self.idx = idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            j = self.idx[i].item()
            return X[j], target_id[j], batch_id[j]

    train_ds, val_ds = DS(train_idx), DS(val_idx)
    if balance:
        counts = torch.bincount(target_id[train_idx])
        w = 1.0 / counts.clamp_min(1.0)
        weights = w[target_id[train_idx]]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    n_targets = len(vocab["target_vocab"])
    n_batches = len(vocab["batch_vocab"])
    n_genes = X.shape[1]

    enc = PerturbEncoder(n_targets, n_batches, out_dim=64).to(device)
    mlp = ConditionalMLP(n_genes, cond_dim=64).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(mlp.parameters()), lr=lr)
    mse = nn.MSELoss()

    def evaluate(loader):
        enc.eval(); mlp.eval()
        se_sum, n = 0.0, 0
        with torch.no_grad():
            for x, t, b in loader:
                x, t, b = x.to(device), t.to(device), b.to(device)
                p = enc(t, b)
                y = mlp(p)
                se_sum += torch.sum((y - x) ** 2).item()
                n += x.numel()
        return se_sum / n

    for ep in range(1, epochs + 1):
        enc.train(); mlp.train()
        for x, t, b in train_loader:
            x, t, b = x.to(device), t.to(device), b.to(device)
            opt.zero_grad(set_to_none=True)
            p = enc(t, b)
            y = mlp(p)
            loss = mse(y, x)
            loss.backward()
            opt.step()
        val_loss = evaluate(val_loader)
        print(f"[Epoch {ep}] val_mse={val_loss:.6f}")

    torch.save({"enc": enc.state_dict(), "mlp": mlp.state_dict()}, Path(artifacts_dir) / "mlp_baseline.pt")
    print("✅ MLP training complete and model saved.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gaussian", "potts", "mlp"], required=True)
    ap.add_argument("--artifacts", required=True)
    # add your other args (epochs, lr, steps, etc.) as needed
    args = ap.parse_args()

    if args.mode == "mlp":
        train_mlp_baseline(args.artifacts)
    elif args.mode == "gaussian":
        # build encoder + GaussianEBM + optax + state, then call train_gaussian(...)
        pass
    else:
        # build encoder + PottsEBM + optax + state, then call train_potts(...)
        pass
