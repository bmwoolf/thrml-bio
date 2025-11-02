![Banner](assets/branching.png)

# thrml-bio

Energy-based modeling on the **Arc Virtual Cell** perturbation dataset, set up to compare a conventional GPU run to an energy-based (thrml) simulator. We train Potts-style EBMs on post-perturbation gene expression and evaluate convergence, accuracy, and efficiency. We also train a standard `torch` MLP on the GPU to test baseline.


## Requirements
- Python 3.10+
- Linux/macOS, NVIDIA GPU recommended (I use 12GB VRAM)
- `uv` env management


## Getting started
```bash
# create .venv and install from uv.lock
uv sync

# activate environment
source .venv/bin/activate
```


## Preprocessing commands
```bash
# Potts
python -m src.data.preprocess \
  --input path/to/arc.h5ad \      # fill out with your path to data
  --outdir artifacts/potts \
  --tau 0.8 \
  --n_hvg 2000 \
  --target_key target_gene \
  --batch_key batch
```

> **No large files checked in.** Preprocessed tensors are generated locally (see “Preprocessing”). `artifacts/` is gitignored.


## Training
...WIP