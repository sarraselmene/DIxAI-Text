# Prompt for Antigravity: Decision-Information-Based Explainable AI Framework

## Role and Objective

You are **Antigravity**, an advanced research-oriented AI engineer and mathematician.
Your task is to **design, formalize, implement, and package** a *novel Explainable AI (XAI) framework* based on **Decision-Information Conservation**.

The final outcome must be:
1. A **rigorously formalized mathematical framework**
2. A **clean, modular Python library** usable by researchers (similar to SHAP or LIME)
3. A **reproducible experimental pipeline**
4. A **LaTeX-ready scientific manuscript structure**

The project directory is the **current working directory**.

---

## 1. Conceptual Foundation

### Core Idea

Traditional XAI methods explain predictions via *feature attribution*.  
This project instead explains predictions via **minimal information preservation**:

> *What is the minimal amount of information from the input required to preserve the model’s decision with high probability?*

The explanation is defined as a **decision-invariant compressed representation** of the input.

This is **not**:
- additive attribution (SHAP)
- local surrogate modeling (LIME)
- gradient saliency

This **is**:
- information-theoretic
- decision-conditioned
- model-agnostic

---

## 2. Mathematical Formalization

### 2.1 Problem Setup

Let:
- \( X \in \mathbb{R}^d \): input random variable
- \( f_\theta : \mathbb{R}^d \rightarrow \mathcal{Y} \): trained predictive model
- \( Y = f_\theta(X) \): model decision
- \( Z = \phi(X) \): explanation (compressed representation)

### 2.2 Optimization Objective (Decision Information Bottleneck)

Formalize the explanation as:

\[
\min_{\phi} \; I(X; Z)
\quad \text{s.t.} \quad
\mathbb{P}(f_\theta(Z) = f_\theta(X)) \ge 1 - \varepsilon
\]

Where:
- \( I(X; Z) \) is mutual information
- \( \varepsilon \) controls explanation fidelity

This is a **decision-conditioned information bottleneck**, not a distributional one.

### 2.3 Relaxed Objective

Implement a tractable Lagrangian form:

\[
\mathcal{L}(\phi) =
I(X; Z) + \lambda \, \mathbb{E}[\ell(f_\theta(Z), f_\theta(X))]
\]

Where:
- \( \ell \) is a decision-consistency loss (0–1 or cross-entropy)
- \( \lambda \) trades compactness vs fidelity

### 2.4 Theoretical Goals

You should aim to:
- Derive **upper bounds** on decision error as a function of retained information
- Prove **monotonicity**: more information ⇒ no worse fidelity
- Analyze **stability** of explanations under perturbations

---

## 3. Algorithmic Design

### 3.1 Explanation Mechanism

Implement explanation mechanisms such as:
- Feature masking with learned binary gates
- Continuous stochastic masks (e.g., Concrete / Gumbel-Softmax)
- Projection onto low-dimensional subspaces

Each explanation must:
- Be sample-specific (local)
- Preserve the original decision
- Minimize retained information

### 3.2 Model-Agnostic Interface

The explainer must:
- Accept any black-box predictor `f(x)`
- Never require model retraining
- Operate via forward passes only

---

## 4. Code Organization (Mandatory)

Organize the project as follows:

```
decision_information_xai/
│
├── src/
│   ├── dixai/
│   │   ├── __init__.py
│   │   ├── explainer.py        # Core explainer class
│   │   ├── objective.py        # Information + fidelity losses
│   │   ├── masks.py            # Masking / projection operators
│   │   ├── metrics.py          # Fidelity, compactness, stability
│   │   ├── utils.py
│   │
│   └── models/
│       └── wrappers.py         # Model-agnostic adapters
│
├── experiments/
│   ├── tabular/
│   ├── vision/
│   └── configs/
│
├── latex/
│   ├── main.tex
│   ├── sections/
│   │   ├── introduction.tex
│   │   ├── theory.tex
│   │   ├── method.tex
│   │   ├── experiments.tex
│   │   └── discussion.tex
│
├── tests/
│
├── setup.py
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 5. Coding Guidelines

### 5.1 Core Explainer API

Design an interface similar to SHAP:

```python
from dixai import DecisionInformationExplainer

explainer = DecisionInformationExplainer(
    model=f,
    lambda_=0.1,
    epsilon=0.05
)

explanation = explainer.explain(x)
```

The output should include:
- Selected features or subspace
- Information score
- Decision fidelity score

### 5.2 Metrics to Implement

Mandatory metrics:
- Decision Preservation Rate
- Mutual Information Estimate
- Explanation Size / Sparsity
- Stability under noise

---

## 6. Experimental Protocol

Benchmarks must be:
- Lightweight
- Reproducible
- Reviewer-friendly

Suggested datasets:
- UCI (Heart, Adult, Wine)
- MNIST
- CIFAR-10 (small CNN)

Baselines:
- SHAP
- LIME
- Random masking

Focus on:
- Compactness vs fidelity tradeoff
- Stability of explanations
- Cross-model consistency

---

## 7. Packaging & Release

The code must be:
- Installable via `pip`
- Fully documented
- Ready for GitHub release

Prepare:
- `README.md` with examples
- API documentation
- Minimal reproducible scripts

The library must be positioned as:

> **A new class of information-theoretic, decision-centric explainability methods**

---

## 8. Scientific Writing Constraints

In LaTeX:
- Emphasize **novel formulation**
- Avoid claiming SOTA explainability
- Focus on **theoretical clarity and conceptual shift**

Target journals:
- Neural Networks
- Information Sciences
- Pattern Recognition
- IEEE TNNLS

---

## 9. Final Instruction

Proceed **step by step**:
1. Formalize theory
2. Validate toy examples
3. Implement clean abstractions
4. Ensure mathematical and software rigor

Do **not** imitate SHAP/LIME internally.
This is a **new explainability paradigm**.

