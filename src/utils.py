"""
utility functions for training and benchmarking

includes GPU verification, metrics tracking, and helper functions
"""
import sys, os, csv, json
from pathlib import Path
import numpy as np
import torch
import jax


# GPU detection and configuration
def verify_gpu_available():
    """verify CUDA GPU is available for both PyTorch and JAX"""
    print("=" * 60)
    print("GPU VERIFICATION")
    print("=" * 60)
    
    # check PyTorch CUDA
    torch_cuda = torch.cuda.is_available()
    if torch_cuda:
        print(f"pytorch CUDA available")
        print(f"  device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("pytorch CUDA NOT available")
    
    # check JAX GPU
    try:
        jax_devices = jax.devices('gpu')
        jax_gpu = len(jax_devices) > 0
        if jax_gpu:
            print(f"JAX GPU available")
            print(f"  devices: {jax_devices}")
        else:
            print("JAX GPU NOT available")
            jax_gpu = False
    except RuntimeError:
        print("JAX GPU NOT available (no GPU backend)")
        jax_gpu = False
    
    print("=" * 60)
    
    # enforce GPU requirement
    if not (torch_cuda and jax_gpu):
        print("\nERROR: GPU not available for all frameworks")
        print("   this training script requires CUDA GPU acceleration")
        print("\ntroubleshooting:")
        if not torch_cuda:
            print("  - install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        if not jax_gpu:
            print("  - install JAX with CUDA 12: pip install 'jax[cuda12]'")
            print("  - verify NVIDIA drivers: nvidia-smi")
        sys.exit(1)
    
    print("all GPU checks passed - training will use CUDA\n")
    return torch_cuda, jax_gpu


def configure_jax_gpu():
    """configure JAX to use GPU memory efficiently"""
    # prevent JAX from pre-allocating all GPU memory
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    print("JAX GPU configuration applied")


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
        "mse": float(test_mse),
        "pcc": float(test_pcc)
    }, indent=2))

