#!/bin/bash
# training script for all model benchmarks
# run this to train all 5 models with benchmarking

set -e  # exit on error

# configuration
EPOCHS=30
BATCH_SIZE=256
LR=0.001

echo "starting model training experiments"
echo "=========================================="

# 1 MLP baseline on Gaussian data
echo ""
echo "[1/5] training MLP baseline (Gaussian)"
python src/train.py \
    --mode mlp \
    --artifacts artifacts/gaussian \
    --run_dir benchmarks/mlp_gaussian_baseline \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR

# 2 MLP baseline on Potts data
echo ""
echo "[2/5] training MLP baseline (Potts)"
python src/train.py \
    --mode mlp \
    --artifacts artifacts/potts \
    --run_dir benchmarks/mlp_potts_baseline \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR

# 3 Gaussian EBM (TODO: implement full training loop)
echo ""
echo "[3/5] training Gaussian EBM"
echo "warning: Gaussian EBM training needs full implementation"
# python src/train.py \
#     --mode gaussian \
#     --artifacts artifacts/gaussian \
#     --run_dir benchmarks/gaussian_ebm_gpu \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --langevin_steps 5 \
#     --step_size 0.05

# 4 Potts EBM (TODO: implement full training loop)
echo ""
echo "[4/5] training Potts EBM"
echo "warning: Potts EBM training needs full implementation"
# python src/train.py \
#     --mode potts \
#     --artifacts artifacts/potts \
#     --run_dir benchmarks/potts_ebm_gpu \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --lr $LR \
#     --langevin_steps 5 \
#     --block_size 64

# 5 additional model (e.g. thrml hardware or balanced sampling variant)
echo ""
echo "[5/5] training MLP with balanced sampling"
python src/train.py \
    --mode mlp \
    --artifacts artifacts/gaussian \
    --run_dir benchmarks/mlp_gaussian_balanced \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --balance

echo ""
echo "=========================================="
echo "training experiments complete"
echo ""
echo "generate benchmark visualizations:"
echo "   python benchmarks/convergance_curve.py --benchmarks benchmarks --outdir reports/figures"
echo "   python benchmarks/model_accuracy.py --benchmarks benchmarks --outdir reports/figures"
echo "   python benchmarks/energy_efficiency_curve.py --benchmarks benchmarks --outdir reports/figures"

