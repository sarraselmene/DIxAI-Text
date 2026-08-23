# Human Interpretability Study Protocol
## "Evaluating Decision-Information Efficiency in Clinical & Natural Vision"

### 1. Objective
To quantify the improvements in **human decision efficiency** and **trust** when using DIxAI explanations compared to standard additive attribution methods (SHAP/Grad-CAM).

### 2. Experimental Design
**Type:** Within-subjects, randomized, counterbalanced design.
**Participants:** $N=20$ (10 AI researchers for ImageNet/CIFAR, 10 Radiologists for CheXpert).

### 3. Tasks
#### Task A: Forward Simulation (Simulatability)
- **Prompt:** "Based on this explanation alone (masked image), what class does the AI predict?"
- **Metric:** Human Prediction Accuracy (%).
- **Hypothesis:** DIxAI (Sufficiency-based) yields higher accuracy than SHAP (Importance-based) because it preserves the *context* needed for decision, not just disjoint features.

#### Task B: Spurious Feature Detection (Trust)
- **Prompt:** "The AI predicted 'Pneumonia'. The explanation highlights these regions. Do you Trust or Distrust this prediction?"
- **Stimuli:** Mixed set of correct predictions and "Clever Hans" predictions (e.g., markers, tubes).
- **Metric:** Trust Score, False Positive Trust Rate (on spurious/wrong predictions).

#### Task C: Comparative Preference (A/B Testing)
- **Prompt:** "Which explanation best helps you understand *why* the model made this decision?"
- **Choice:** [Method A] vs [Method B] (Randomized left/right).
- **Metric:** Preference Ratio (%).

### 4. Stimuli Generation
- **Source:** 50 ImageNet, 50 CheXpert samples.
- **Methods:** DIxAI (Ours), SHAP (Baseline 1), Grad-CAM++ (Baseline 2).
- **Format:** Side-by-side panels.

### 5. Statistical Analysis
- **Test:** Paired t-test or Wilcoxon Signed-Rank Test.
- **Hypothesis:** $p < 0.05$ for DIxAI superiority in Simulatability.

### 6. Implementation
Run `generate_survey.py` to create the evaluation set.
distribute results via PDF survey form or web interface.
