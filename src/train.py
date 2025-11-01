"""
training loops and utilities for energy-based models

implements training functions for Gaussian and Potts EBMs using persistent
contrastive divergence (PCD), plus PyTorch MLP baseline for comparison
includes JAX/Flax training state management and optimizer integration
"""
import time, json, csv
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import jax, optax, jax.numpy as jnp
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.encoders import PerturbEncoder
from models.gaussian import GaussianEBM
from models.mlp import ConditionalMLP
from models.potts import PottsEBM
from models.sampler import langevin_pcd_apply, potts_gibbs_block


# helper: compute Pearson correlation coefficient
def compute_pcc(y_pred, y_true):
    """compute Pearson correlation between predictions and targets (flattened)"""
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    mean_pred = np.mean(y_pred)
    mean_true = np.mean(y_true)
    num = np.sum((y_pred - mean_pred) * (y_true - mean_true))
    denom = np.sqrt(np.sum((y_pred - mean_pred)**2) * np.sum((y_true - mean_true)**2))
    return num / (denom + 1e-8)


# helper: write metrics to CSV
def init_metrics_csv(run_dir: Path):
    """initialize metrics CSV with headers"""
    csv_path = run_dir / "metrics.csv"
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_mse", "val_pcc", "wall_clock_s", "joules"])
        writer.writeheader()
    return csv_path


def append_metrics_csv(csv_path: Path, epoch: int, val_mse: float, val_pcc: float, wall_clock_s: float, joules: float = None):
    """append one epoch's metrics to CSV"""
    with csv_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_mse", "val_pcc", "wall_clock_s", "joules"])
        writer.writerow({
            "epoch": epoch,
            "val_mse": val_mse,
            "val_pcc": val_pcc,
            "wall_clock_s": wall_clock_s,
            "joules": joules if joules is not None else ""
        })


def save_test_metrics(run_dir: Path, test_mse: float, test_pcc: float):
    """save final test metrics to JSON"""
    test_path = run_dir / "test_metrics.json"
    test_path.write_text(json.dumps({
        "mse": test_mse,
        "pcc": test_pcc
    }, indent=2))


# make train state for JAX/Flax
def make_train_state(rng, model, tx, sample_x, sample_p):
    params = model.init(rng, sample_x, sample_p)['params']
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# train epoch for Gaussian EBM
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


# train epoch for Potts EBM
def train_epoch_potts(state, enc_apply, enc_params, model, batch_iter, sampler, steps=5, block_size=64):
    """PCD with block Gibbs for Potts x in {-1,0,1}"""
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
        p = batch['p']          # dict of ids (target_id, batch_id, etc)
        state = step(state, x, p)
    return state


# predict mean for Gaussian EBM
def predict_mu(enc_apply, enc_params, model_apply, model_params, p):
    p_emb = enc_apply({'params': enc_params}, **p)
    # TODO: for Gaussian mu = Dense(p_emb)
    # TODO: for reporting PCC/MSE on val/test call model pieces or add a helper
    return p_emb


# train MLP baseline (non-EBM)
def train_mlp_baseline(artifacts_dir, run_dir, epochs=30, batch_size=256, lr=1e-3, balance=False):
    """train a simple PyTorch MLP predictor using the same condition encoder"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create run directory
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # initialize metrics CSV
    csv_path = init_metrics_csv(run_dir)

    # load artifacts
    artifacts_dir = Path(artifacts_dir)
    tensors = torch.load(artifacts_dir / "tensors.pt")
    conds = torch.load(artifacts_dir / "conditions.pt")
    vocab = json.loads((artifacts_dir / "vocab.json").read_text())

    X = tensors["X"].float()
    target_id = conds["target_id"].long()
    batch_id = conds["batch_id"].long()

    # train/val/test splits
    splits = json.loads((artifacts_dir / "splits.json").read_text())
    train_idx = torch.tensor(splits["train_idx"], dtype=torch.long)
    val_idx = torch.tensor(splits["val_idx"], dtype=torch.long)
    test_idx = torch.tensor(splits["test_idx"], dtype=torch.long)

    # Dataset class
    class DS(torch.utils.data.Dataset):
        def __init__(self, idx):
            self.idx = idx
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            j = self.idx[i].item()
            return X[j], target_id[j], batch_id[j]

    train_ds, val_ds, test_ds = DS(train_idx), DS(val_idx), DS(test_idx)
    if balance:
        counts = torch.bincount(target_id[train_idx])
        w = 1.0 / counts.clamp_min(1.0)
        weights = w[target_id[train_idx]]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    n_targets = len(vocab["target_vocab"])
    n_batches = len(vocab["batch_vocab"])
    n_genes = X.shape[1]

    enc = PerturbEncoder(n_targets, n_batches, out_dim=64).to(device)
    mlp = ConditionalMLP(n_genes, cond_dim=64).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(mlp.parameters()), lr=lr)
    mse_fn = nn.MSELoss()

    def evaluate(loader):
        """compute MSE and PCC on a dataset"""
        enc.eval(); mlp.eval()
        all_preds, all_targets = [], []
        se_sum, n = 0.0, 0
        with torch.no_grad():
            for x, t, b in loader:
                x, t, b = x.to(device), t.to(device), b.to(device)
                p = enc(t, b)
                y = mlp(p)
                se_sum += torch.sum((y - x) ** 2).item()
                n += x.numel()
                all_preds.append(y.cpu().numpy())
                all_targets.append(x.cpu().numpy())
        mse = se_sum / n
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        pcc = compute_pcc(preds, targets)
        return mse, pcc

    # training loop with benchmarking
    for ep in range(1, epochs + 1):
        epoch_start = time.time()
        
        enc.train(); mlp.train()
        for x, t, b in train_loader:
            x, t, b = x.to(device), t.to(device), b.to(device)
            opt.zero_grad(set_to_none=True)
            p = enc(t, b)
            y = mlp(p)
            loss = mse_fn(y, x)
            loss.backward()
            opt.step()
        
        # evaluate on validation set
        val_mse, val_pcc = evaluate(val_loader)
        
        epoch_time = time.time() - epoch_start
        
        # log metrics
        append_metrics_csv(csv_path, ep, val_mse, val_pcc, epoch_time)
        print(f"[Epoch {ep}/{epochs}] val_mse={val_mse:.6f} | val_pcc={val_pcc:.4f} | time={epoch_time:.2f}s")

    # evaluate on test set
    test_mse, test_pcc = evaluate(test_loader)
    save_test_metrics(run_dir, test_mse, test_pcc)
    print(f"test: mse={test_mse:.6f} | pcc={test_pcc:.4f}")

    # save model
    torch.save({"enc": enc.state_dict(), "mlp": mlp.state_dict()}, run_dir / "mlp_model.pt")
    print(f"MLP training complete, results saved to {run_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train energy-based models with benchmarking")
    ap.add_argument("--mode", choices=["gaussian", "potts", "mlp"], required=True,
                    help="Model type to train")
    ap.add_argument("--artifacts", required=True,
                    help="Path to preprocessed artifacts directory (e.g., artifacts/gaussian)")
    ap.add_argument("--run_dir", required=True,
                    help="Output directory for this run (e.g., benchmarks/mlp_baseline_gpu)")
    ap.add_argument("--epochs", type=int, default=30,
                    help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="Batch size for training")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate")
    ap.add_argument("--balance", action="store_true",
                    help="Use balanced sampling for imbalanced datasets")
    
    # EBM-specific args
    ap.add_argument("--langevin_steps", type=int, default=5,
                    help="Number of Langevin/Gibbs steps for EBM sampling")
    ap.add_argument("--step_size", type=float, default=0.05,
                    help="Langevin step size (for Gaussian EBM)")
    ap.add_argument("--block_size", type=int, default=64,
                    help="Block size for Gibbs sampling (for Potts EBM)")
    
    args = ap.parse_args()

    if args.mode == "mlp":
        train_mlp_baseline(
            artifacts_dir=args.artifacts,
            run_dir=args.run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            balance=args.balance
        )
    elif args.mode == "gaussian":
        # TODO: implement Gaussian EBM training loop with benchmarking
        print("warning: Gaussian EBM training not yet implemented with benchmarking")
        print("    use train_epoch_gaussian() as a template")
    else:
        # TODO: implement Potts EBM training loop with benchmarking
        print("warning: Potts EBM training not yet implemented with benchmarking")
        print("    use train_epoch_potts() as a template")
