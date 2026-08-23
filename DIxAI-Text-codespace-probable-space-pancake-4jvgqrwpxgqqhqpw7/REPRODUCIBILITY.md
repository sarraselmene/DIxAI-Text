# Reproducibility Guide

This document provides instructions to reproduce the results presented in the paper **"Decision-Information Explanations (DIxAI)"**.

To generate the statistical bounds reported in the paper (Tables 1, 2, 3), we run the benchmarks across **10 independent random seeds** (0-9).

## 1. Environment Setup

### Hardware Requirements
- **GPU:** NVIDIA RTX 4070 (or equivalent) with >= 12GB VRAM recommended for ImageNet training.
- **CPU:** 8+ cores recommended for data loading.
- **RAM:** 32GB+.

### Software Dependencies
Install the required packages:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn scipy tqdm
```

### Random Seeds
All experiments use the following seeds for variance reporting (N=5):
- `42` (Primary)
- `123`
- `456`
- `789`
- `1011`

## 2. Reproducing Main Tables

We provide a master script `reproduce_paper_results.py` to execute the experiments corresponding to each table.

### Table 1: Tabular Baselines (Fidelity/Sparsity Pareto)
```bash
python reproduce_paper_results.py --table1
```
*Output: `experiments/results/tabular_baselines.csv`*

### Table 2: High-Dimensional Physics (Higgs Boson)
```bash
python reproduce_paper_results.py --table2
```
*Note: This benchmark runs on 1,000 test samples by default.*

### Table 3: Faithfulness Evaluation (Insertion/Deletion AUC)
```bash
python reproduce_paper_results.py --table3
```
*Computes AUC metrics for DIxAI vs. SHAP/LIME/GradCAM.*

### Table 5: ImageNet Benchmarks
```bash
python reproduce_paper_results.py --table5
```
*Requires ImageNet validation set in `data/imagenet_val`.*

### Table 6: Computational Characterization (Latency)
```bash
python reproduce_paper_results.py --latency
```
*Measures inference time on your local hardware.*

## 3. Training the Amortized Explainer

To train the `AmortizedExplainer` from scratch (as used in Section 5.14):

```bash
python experiments/main_train_amortized.py \
    --backbone resnet50 \
    --data-path /path/to/imagenet/val \
    --epochs 10 \
    --lambda-fidelity 5.0 \
    --lambda-sparsity 0.5 \
    --lambda-tv 0.1
```

Pre-trained checkpoints can be found (or placed) in the `checkpoints/` directory.

## 4. Troubleshooting
- **CUDA OOM:** Reduce `--batch-size` in `main_train_amortized.py` or the specific benchmark script.
- **Missing Data:** Ensure `data/` contains `imagenet_samples/` or the required tabular CSVs. The scripts include automatic downloaders for public datasets (Adult, MNIST).
