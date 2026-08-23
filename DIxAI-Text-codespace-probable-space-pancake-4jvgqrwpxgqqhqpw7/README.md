# DIxAI: Decision-Information Explanations

**Official PyTorch Implementation**
explanations by minimizing the mutual information between the input and the explanation mask while maximizing the mutual information between the masked input and the model's decision.

This repository contains the official PyTorch implementation and reproduction scripts for the experiments.

## 🚀 Key Features
- **Theoretical Guarantee:** Optimizes the Information Bottleneck objective $I(Y; Z) - \beta I(X; Z)$.
- **Architecture Agnostic:** Works with CNNs (ResNet, DenseNet, VGG), ViTs, and Tabular MLPs.
- **High-Fidelity:** Unlike attribution methods (SHAP, Grad-CAM), DIxAI produces executable masks that preserve model decisions with high probability.
- **Robustness:** Includes built-in mechanisms for total variation regularization and spatial priors.

## 📦 Installation

```bash
git clone https://github.com/haythemghazouani/decision_information_xai.git
cd decision_information_xai
pip install -e .
```

## 🧪 Reproducing Experiments

This repository includes the scripts used to generate the results for the manuscript.

### 1. Robustness & Stability (Tabular)
Reproduce the multi-seed stability analysis and baseline comparison on the Iris dataset:

```bash
# Run Baseline Comparison (DIxAI vs L2X vs Integrated Gradients)
python experiments/benchmark/bench_tabular_baselines.py

# Run Multi-Seed Statistical Analysis (N=10)
python experiments/benchmark/bench_robustness.py
```

### 2. Faithfulness Benchmarks (ROAR)
Run the Remove-And-Retrain (ROAR) evaluation to measure degradation curves:

```bash
python experiments/consistency/bench_roar.py
```

### 3. Sanity Checks
Verify model parameter randomization tests (Adebayo et al.):

```bash
python experiments/consistency/sanity_checks.py
```

### 4. Medical Domain Validation
Verify architecture compatibility with DenseNet-121 (CheXpert style):

```bash
python experiments/medical/bench_chexpert_transfer.py
```

## 📂 Project Structure

- `src/dixai`: Core library implementation.
  - `explainer.py`: Main `DecisionInformationExplainer` class.
  - `masks.py`: Learnable mask implementations (Gumbel-Softmax).
  - `objective.py`: Information Bottleneck loss functions.
- `experiments/`: Benchmark scripts.
  - `benchmark/`: Tabular and baseline comparisons.
  - `consistency/`: ROAR and Sanity Checks.
  - `medical/`: Medical imaging validation.
  - `vision/`: ImageNet/CIFAR visualizations.

## 📜 Citation

If you use this code in your research, please cite:

```bibtex
@article{ghazouani2026dixai,
  title={DIxAI: Decision-Information Conservation for Explainable AI},
  author={Ghazouani, Haythem},
  journal={arXiv preprint},
  year={2026}
}
```

## 📄 License
MIT License.
