#!/bin/bash
# training script for all 4 model benchmarks
# trains Potts EBM (JAX + thrml), Gaussian EBM (JAX only), and MLP baseline (PyTorch)

set -e  # exit on error

# configuration
EPOCHS=30
BATCH_SIZE=256
LR=0.001
LANGEVIN_STEPS=5
STEP_SIZE=0.05
BLOCK_SIZE=64

echo "starting 4 model training experiments"
echo "=========================================="

# 1 Potts EBM with JAX backend
echo ""
echo "[1/4] training Potts EBM (JAX backend)"
echo "warning: Potts EBM training loop needs full implementation"
# python src/train.py \
#     --mode potts \
#     --backend jax \
#     --artifacts artifacts/potts \
#     --run_dir benchmarks/potts_ebm_jax \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --langevin_steps $LANGEVIN_STEPS \
#     --block_size $BLOCK_SIZE

# 2 Potts EBM with thrml backend
echo ""
echo "[2/4] training Potts EBM (thrml backend)"
echo "warning: Potts EBM training loop needs full implementation"
# python src/train.py \
#     --mode potts \
#     --backend thrml \
#     --artifacts artifacts/potts \
#     --run_dir benchmarks/potts_ebm_thrml \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --langevin_steps $LANGEVIN_STEPS \
#     --block_size $BLOCK_SIZE

# 3 Gaussian EBM with JAX backend (thrml doesn't support continuous models)
echo ""
echo "[3/4] training Gaussian EBM (JAX backend)"
echo "warning: Gaussian EBM training loop needs full implementation"
# python src/train.py \
#     --mode gaussian \
#     --backend jax \
#     --artifacts artifacts/gaussian \
#     --run_dir benchmarks/gaussian_ebm_jax \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --langevin_steps $LANGEVIN_STEPS \
#     --step_size $STEP_SIZE

# 4 MLP baseline with PyTorch
echo ""
echo "[4/4] training MLP baseline (PyTorch)"
python src/train.py \
    --mode mlp \
    --backend torch \
    --artifacts artifacts/gaussian \
    --run_dir benchmarks/mlp_baseline_torch \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR

echo ""
echo "=========================================="
echo "training experiments complete"
echo ""
echo "generate benchmark visualizations:"
echo "   python benchmarks/convergance_curve.py --benchmarks benchmarks --outdir reports/figures"
echo "   python benchmarks/model_accuracy.py --benchmarks benchmarks --outdir reports/figures"
echo "   python benchmarks/energy_efficiency_curve.py --benchmarks benchmarks --outdir reports/figures"
