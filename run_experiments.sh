#!/bin/bash
# training script for all 3 model benchmarks
# trains Potts EBM (JAX + thrml) and MLP baseline (PyTorch)

set -e  # exit on error

# configuration
EPOCHS=100
EPOCHS_THRML=100  # thrml training epochs
BATCH_SIZE=256
LR=0.001
GIBBS_STEPS=5
GIBBS_STEPS_THRML=3  # fewer steps for faster thrml training
BLOCK_SIZE=64

echo "starting 3 model training experiments"
echo "=========================================="

# 1 MLP baseline with PyTorch
echo ""
echo "[1/3] training MLP baseline (PyTorch)"
python src/train.py \
    --mode mlp \
    --backend torch \
    --artifacts artifacts/potts \
    --run_dir benchmarks/mlp_baseline_torch \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR

# 2 Potts EBM with JAX backend
echo ""
echo "[2/3] training Potts EBM (JAX backend)"
python src/train.py \
    --mode potts \
    --backend jax \
    --artifacts artifacts/potts \
    --run_dir benchmarks/potts_ebm_jax \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --gibbs_steps $GIBBS_STEPS \
    --block_size $BLOCK_SIZE

# 3 Potts EBM with thrml backend (uses 2000 genes)
echo ""
echo "[3/3] training Potts EBM (thrml backend)"
python src/train.py \
    --mode potts \
    --backend thrml \
    --artifacts artifacts/potts \
    --run_dir benchmarks/potts_ebm_thrml \
    --epochs $EPOCHS_THRML \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --gibbs_steps $GIBBS_STEPS_THRML \
    --block_size $BLOCK_SIZE \
    --max_genes_thrml 2000


echo ""
echo "=========================================="
echo "training experiments complete"
echo ""
echo "generating visualizations..."
echo "=========================================="

# benchmark visualizations
echo ""
echo "creating convergence curves..."
python benchmarks/convergance_curve.py --benchmarks benchmarks --outdir reports/figures

echo ""
echo "creating model accuracy comparison..."
python benchmarks/model_accuracy.py --benchmarks benchmarks --outdir reports/figures

echo ""
echo "creating energy efficiency curves..."
python benchmarks/energy_efficiency_curve.py --benchmarks benchmarks --outdir reports/figures

# 3D visualizations
echo ""
echo "creating gene expression surface..."
python benchmarks/gene_expression_surface.py --artifacts artifacts/potts --split test --outdir reports/figures

echo ""
echo "creating energy landscape (JAX)..."
python benchmarks/energy_landscape.py --checkpoint benchmarks/potts_ebm_jax/model_checkpoint.pkl --outdir reports/figures

echo ""
echo "creating energy landscape (thrml)..."
python benchmarks/energy_landscape.py --checkpoint benchmarks/potts_ebm_thrml/model_checkpoint.pkl --outdir reports/figures

echo ""
echo "=========================================="
echo "all visualizations saved to reports/figures/"
echo "=========================================="
