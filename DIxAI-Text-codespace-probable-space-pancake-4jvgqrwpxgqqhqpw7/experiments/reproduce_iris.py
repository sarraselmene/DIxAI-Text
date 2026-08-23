
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Try to import from installed package, otherwise fall back to src
try:
    from dixai import DecisionInformationExplainer, plot_feature_importance
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
    from dixai import DecisionInformationExplainer, plot_feature_importance

# 1. Setup Data
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# 2. Train Simple Model
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.log_softmax(self.fc2(x))

model = SimpleMLP()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

print("Training Model...")
for epoch in range(100):
    optimizer.zero_grad()
    out = model(X_train_t)
    loss = criterion(out, y_train_t)
    loss.backward()
    optimizer.step()
print("Training Complete.")

# 3. Explain Instance
explainer = DecisionInformationExplainer(model, lambda_fidelity=20.0, task='classification')

idx = 0
x_sample = X_test_t[idx]
print(f"\nExplaining instance {idx}")
print(f"Features: {x_sample.numpy()}")
print(f"Ground Truth Feature Names: {iris.feature_names}")

explanation = explainer.explain(x_sample, steps=1000, lr=0.1, verbose=False)

print("\n--- Explanation ---")
print(explanation)
print(f"Final Mask: {explanation.mask.numpy().round(2).flatten()}")

# 4. Visualization
plot_path = os.path.join(os.path.dirname(__file__), 'explanation_plot.png')
print(f"Saving plot to {plot_path}...")
class_orig_idx = torch.argmax(torch.exp(model(x_sample.unsqueeze(0)))).item()

plot_feature_importance(
    explanation, 
    feature_names=iris.feature_names, 
    title=f"DIXAI Explanation (True Class: {iris.target_names[class_orig_idx]})",
    save_path=plot_path,
    show=False
)

# 5. Check prediction with mask
model.eval()
with torch.no_grad():
    orig_pred = torch.exp(model(x_sample.unsqueeze(0)))
    masked_sample = x_sample * explanation.mask
    masked_pred = torch.exp(model(masked_sample.unsqueeze(0)))

print(f"\nOriginal Probabilities: {orig_pred.numpy().round(3).flatten()}")
print(f"Masked Probabilities:   {masked_pred.numpy().round(3).flatten()}")

class_orig = orig_pred.argmax().item()
class_mask = masked_pred.argmax().item()

if class_orig == class_mask:
    print("\nSUCCESS: Decision Preserved!")
else:
    print("\nFAILURE: Decision Changed.")

# 6. Global Analysis (Batch)
print("\n--- Global Analysis ---")
# Explain first 5 test instances (Reduced for speed)
batch_size = 5
x_batch = X_test_t[:batch_size]
print(f"Explaining batch of {batch_size} instances...")
# Reduced steps for demo speed
batch_explanations = explainer.explain_batch(x_batch, steps=200, verbose=False)

global_plot_path = os.path.join(os.path.dirname(__file__), 'global_importance_plot.png')
from dixai import plot_global_importance, plot_variation_curve
plot_global_importance(
    batch_explanations,
    feature_names=iris.feature_names,
    title="Global Feature Importance (Iris Test Subset)",
    save_path=global_plot_path,
    show=False
)

# 7. Stability / Variation Analysis
print("\n--- Variation Analysis ---")
lambdas = [0.1, 1.0, 10.0, 50.0] # Few points for speed
print(f"Sweeping lambda values: {lambdas}")
sweep_results = explainer.sweep_lambda(x_sample, lambdas, steps=200)

variation_plot_path = os.path.join(os.path.dirname(__file__), 'variation_plot.png')
plot_variation_curve(
    sweep_results,
    save_path=variation_plot_path,
    show=False
)
