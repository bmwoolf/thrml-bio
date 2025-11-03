"""
training loops and utilities for energy-based models

implements training functions for Potts EBMs using persistent
contrastive divergence (PCD), plus PyTorch MLP baseline for comparison
includes JAX/Flax training state management and optimizer integration
"""
import time, json, pickle
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import jax, optax, jax.numpy as jnp
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils import (
    verify_gpu_available, 
    configure_jax_gpu, 
    compute_pcc,
    init_metrics_csv, 
    append_metrics_csv, 
    save_test_metrics
)
from models.encoders import PerturbEncoder, PerturbEncoderJAX
from models.jax.potts import PottsEBM as PottsEBM_JAX
from models.thrml.potts import PottsEBMThrml, thrml_potts_sampler
from models.mlp import ConditionalMLP
from models.sampler import potts_gibbs_block


# make train state for JAX/Flax
def make_train_state(rng, model, tx, sample_x, sample_p):
    params = model.init(rng, sample_x, sample_p)['params']
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# train epoch for Potts EBM
def train_epoch_potts(state, enc_apply, enc_params, model, batch_iter, sampler, steps=5, block_size=64, rng=None):
    """PCD with block Gibbs for Potts x in {-1,0,1}"""
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    @jax.jit
    def loss_fn(params, x_pos, p_emb, rng_key):
        # x_pos is ternary in {-1,0,1} (int8/float ok)
        E_pos = model.apply({'params': params}, x_pos, p_emb)  # [B]
        x_init = x_pos
        x_neg  = sampler(params, model.apply, x_init, p_emb, block_size=block_size, steps=steps, rng=rng_key)
        E_neg = model.apply({'params': params}, x_neg, p_emb)
        return jnp.mean(E_pos) - jnp.mean(E_neg)

    @jax.jit
    def step(state, x_pos, p, rng_key):
        p_emb = enc_apply({'params': enc_params}, **p)
        grads = jax.grad(lambda prm: loss_fn(prm, x_pos, p_emb, rng_key))(state.params)
        return state.apply_gradients(grads=grads)

    for i, batch in enumerate(batch_iter):
        x = batch['x']          # [B,G] values in {-1,0,1}
        p = batch['p']          # dict of ids (target_id, batch_id, etc)
        rng_key = jax.random.fold_in(rng, i)
        state = step(state, x, p, rng_key)
    return state


# full training loop for Potts EBM
def train_potts_ebm(artifacts_dir, run_dir, backend="jax", epochs=30, batch_size=256, lr=1e-3, 
                    gibbs_steps=5, block_size=64, balance=False, max_genes_thrml=100):
    """
    train Potts EBM with either JAX or thrml backend
    
    uses persistent contrastive divergence (PCD) with block Gibbs sampling
    for thrml backend, subsamples to max_genes_thrml to avoid edge explosion
    """
    print(f"\nTraining Potts EBM ({backend} backend)")
    print("=" * 60)
    
    # configure JAX for GPU
    configure_jax_gpu()
    
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
    
    X = tensors["X"].numpy()  # convert to numpy for JAX
    target_id = conds["target_id"].numpy()
    batch_id = conds["batch_id"].numpy()
    
    # for thrml: subsample genes to avoid edge explosion
    if backend == "thrml" and X.shape[1] > max_genes_thrml:
        print(f"thrml backend: subsampling {X.shape[1]} genes -> {max_genes_thrml} genes")
        # select top N most variable genes
        gene_var = np.var(X, axis=0)
        top_gene_idx = np.argsort(gene_var)[-max_genes_thrml:]
        X = X[:, top_gene_idx]
        print(f"selected top {max_genes_thrml} most variable genes")
        n_edges = max_genes_thrml * (max_genes_thrml - 1) // 2
        print(f"graph will have {n_edges:,} edges")
    
    # train/val/test splits
    splits = json.loads((artifacts_dir / "splits.json").read_text())
    train_idx = np.array(splits["train_idx"])
    val_idx = np.array(splits["val_idx"])
    test_idx = np.array(splits["test_idx"])
    
    n_targets = len(vocab["target_vocab"])
    n_batches = len(vocab["batch_vocab"])
    n_genes = X.shape[1]
    
    print(f"Dataset: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    print(f"Genes: {n_genes}, Targets: {n_targets}, Batches: {n_batches}")
    
    # select model and sampler based on backend
    if backend == "thrml":
        model_class = PottsEBMThrml
        sampler = thrml_potts_sampler
        print("Using thrml Potts EBM with block Gibbs sampler")
    else:  # jax
        model_class = PottsEBM_JAX
        sampler = potts_gibbs_block
        print("Using JAX Potts EBM with block Gibbs sampler")
    
    # check for existing checkpoint to resume
    checkpoint_path = run_dir / "checkpoint_latest.pkl"
    start_epoch = 1
    
    if checkpoint_path.exists():
        print(f"found checkpoint at {checkpoint_path}, resuming...")
        try:
            with open(checkpoint_path, 'rb') as f:
                ckpt = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"  checkpoint corrupted ({e}), starting fresh")
            checkpoint_path.unlink()
            ckpt = None
        
        if ckpt is not None:
            rng = ckpt['rng']
            enc_params = ckpt['enc_params']
            start_epoch = ckpt['epoch'] + 1
            n_genes = ckpt['config']['n_genes']
            
            # recreate model and encoder
            encoder = PerturbEncoderJAX(n_targets, n_batches, out_dim=64)
            model = model_class(n_genes=n_genes, cond_dim=64)
            
            # recreate TrainState from saved params and opt_state
            sample_x = jnp.zeros((1, n_genes))
            sample_p = jnp.zeros((1, 64))
            tx = optax.adam(lr)
            state = TrainState.create(
                apply_fn=model.apply,
                params=ckpt['model_params'],
                tx=tx
            )
            state = state.replace(opt_state=ckpt['opt_state'])
            
            print(f"resuming from epoch {start_epoch}/{epochs}")
        else:
            # checkpoint was corrupted, initialize from scratch
            rng = jax.random.PRNGKey(42)
            rng, enc_rng, model_rng = jax.random.split(rng, 3)
            
            encoder = PerturbEncoderJAX(n_targets, n_batches, out_dim=64)
            sample_t = jnp.array([0])
            sample_b = jnp.array([0])
            enc_params = encoder.init(enc_rng, target_id=sample_t, batch_id=sample_b)['params']
            
            model = model_class(n_genes=n_genes, cond_dim=64)
            sample_x = jnp.zeros((1, n_genes))
            sample_p = jnp.zeros((1, 64))
            
            tx = optax.adam(lr)
            state = make_train_state(model_rng, model, tx, sample_x, sample_p)
            
            print(f"Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(state.params))} parameters")
    else:
        # initialize models from scratch
        rng = jax.random.PRNGKey(42)
        rng, enc_rng, model_rng = jax.random.split(rng, 3)
        
        # encoder for conditions
        encoder = PerturbEncoderJAX(n_targets, n_batches, out_dim=64)
        sample_t = jnp.array([0])
        sample_b = jnp.array([0])
        enc_params = encoder.init(enc_rng, target_id=sample_t, batch_id=sample_b)['params']
        
        # Potts model
        model = model_class(n_genes=n_genes, cond_dim=64)
        sample_x = jnp.zeros((1, n_genes))
        sample_p = jnp.zeros((1, 64))
        
        # create optimizer and training state
        tx = optax.adam(lr)
        state = make_train_state(model_rng, model, tx, sample_x, sample_p)
        
        print(f"Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(state.params))} parameters")
    
    # create data iterator
    def batch_generator(indices, batch_size, shuffle=True):
        if shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            yield {
                'x': jnp.array(X[batch_idx]),
                'p': {
                    'target_id': jnp.array(target_id[batch_idx]),
                    'batch_id': jnp.array(batch_id[batch_idx])
                }
            }
    
    # evaluation function (optimized: fewer steps, smaller batches)
    def evaluate(indices, eval_gibbs_steps=2, eval_batch_size=64):
        """compute MSE and PCC by sampling from EBM and comparing to actual data"""
        all_preds = []
        all_targets = []
        
        eval_rng = jax.random.PRNGKey(12345)
        
        # use smaller batches for evaluation
        for batch in batch_generator(indices, eval_batch_size, shuffle=False):
            x_actual = batch['x']
            p_emb = encoder.apply({'params': enc_params}, **batch['p'])
            
            # sample predictions from trained EBM
            # initialize from zeros, then sample given conditions
            x_init = jnp.zeros_like(x_actual)
            eval_rng, sample_rng = jax.random.split(eval_rng)
            
            x_pred = sampler(
                state.params,
                model.apply,
                x_init,
                p_emb,
                block_size=block_size,
                steps=eval_gibbs_steps,  # fewer steps for faster eval
                rng=sample_rng
            )
            
            all_preds.append(np.array(x_pred))
            all_targets.append(np.array(x_actual))
        
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        mse = np.mean((preds - targets) ** 2)
        pcc = compute_pcc(preds, targets)
        
        return mse, pcc
    
    # training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)
    
    for ep in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        
        # train one epoch
        train_batches = list(batch_generator(train_idx, batch_size, shuffle=True))
        rng, epoch_rng = jax.random.split(rng)
        state = train_epoch_potts(
            state, 
            encoder.apply, 
            enc_params, 
            model, 
            train_batches, 
            sampler,
            steps=gibbs_steps,
            block_size=block_size,
            rng=epoch_rng
        )
        
        # evaluate on validation set
        val_mse, val_pcc = evaluate(val_idx)
        
        epoch_time = time.time() - epoch_start
        
        # log metrics
        append_metrics_csv(csv_path, ep, val_mse, val_pcc, epoch_time)
        print(f"[Epoch {ep}/{epochs}] val_mse={val_mse:.6f} | val_pcc={val_pcc:.4f} | time={epoch_time:.2f}s")
        
        # save checkpoint after each epoch (save params and opt_state separately, not full state)
        checkpoint = {
            'epoch': ep,
            'model_params': state.params,
            'opt_state': state.opt_state,
            'enc_params': enc_params,
            'rng': rng,
            'config': {
                'n_genes': n_genes,
                'n_targets': n_targets,
                'n_batches': n_batches,
                'backend': backend,
                'gibbs_steps': gibbs_steps,
                'block_size': block_size
            }
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"  checkpoint saved to {checkpoint_path}")
    
    # evaluate on test set
    test_mse, test_pcc = evaluate(test_idx)
    save_test_metrics(run_dir, test_mse, test_pcc)
    print(f"\nTest: mse={test_mse:.6f} | pcc={test_pcc:.4f}")
    
    # save model checkpoint
    checkpoint = {
        'model_params': state.params,
        'enc_params': enc_params,
        'config': {
            'n_genes': n_genes,
            'n_targets': n_targets,
            'n_batches': n_batches,
            'backend': backend,
            'gibbs_steps': gibbs_steps,
            'block_size': block_size
        }
    }
    
    with open(run_dir / "model_checkpoint.pkl", 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Potts EBM training complete, results saved to {run_dir}")


# train MLP baseline (non-EBM)
def train_mlp_baseline(artifacts_dir, run_dir, epochs=30, batch_size=256, lr=1e-3, balance=False):
    """train a simple PyTorch MLP predictor using the same condition encoder"""
    print(f"\nTraining MLP Baseline (PyTorch)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

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
    ap = argparse.ArgumentParser(description="train energy-based models with benchmarking")
    ap.add_argument("--mode", choices=["potts", "mlp"], required=True,
                    help="model type to train")
    ap.add_argument("--backend", choices=["jax", "thrml", "torch"], default="jax",
                    help="backend: jax (pure JAX/Flax), thrml (thrml library), torch (PyTorch)")
    ap.add_argument("--artifacts", required=True,
                    help="path to preprocessed artifacts directory (eg artifacts/potts)")
    ap.add_argument("--run_dir", required=True,
                    help="output directory for this run (eg benchmarks/mlp_baseline_torch)")
    ap.add_argument("--epochs", type=int, default=30,
                    help="number of training epochs")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="batch size for training")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate")
    ap.add_argument("--balance", action="store_true",
                    help="use balanced sampling for imbalanced datasets")
    
    # EBM-specific args
    ap.add_argument("--gibbs_steps", type=int, default=5,
                    help="number of Gibbs steps for EBM sampling")
    ap.add_argument("--block_size", type=int, default=64,
                    help="block size for Gibbs sampling (for Potts EBM)")
    ap.add_argument("--max_genes_thrml", type=int, default=100,
                    help="max genes for thrml backend (subsamples if larger)")
    
    args = ap.parse_args()
    
    # verify GPU is available before training
    verify_gpu_available()

    if args.mode == "mlp":
        if args.backend != "torch":
            print(f"warning: MLP only supports torch backend, ignoring --backend={args.backend}")
        train_mlp_baseline(
            artifacts_dir=args.artifacts,
            run_dir=args.run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            balance=args.balance
        )
    else:  # potts
        train_potts_ebm(
            artifacts_dir=args.artifacts,
            run_dir=args.run_dir,
            backend=args.backend,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            gibbs_steps=args.gibbs_steps,
            block_size=args.block_size,
            balance=args.balance,
            max_genes_thrml=args.max_genes_thrml
        )
