
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import time

# Try to import from installed package, otherwise fall back to src
try:
    from dixai import DecisionInformationExplainer, calculate_insertion_deletion_curves, calculate_auc
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
    from dixai import DecisionInformationExplainer, calculate_insertion_deletion_curves, calculate_auc

# 1. Setup Data (Higgs Proxy - High Dimensional Tabular)
# For speed and reliability in this environment, we use a synthetic 28-feature classification task
# mirroring the structure of the Higgs dataset.
print("Generating Higgs-protocol synthetic data (28 features)...")
X, y = make_classification(n_samples=5000, n_features=28, n_informative=21, n_redundant=7, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# 2. Train Benchmark Model
class HiggsMLP(nn.Module):
    def __init__(self, input_dim=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HiggsMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(f"Training Benchmark Model on {device}...")
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    out = model(X_train_t.to(device))
    loss = criterion(out, y_train_t.to(device))
    loss.backward()
    optimizer.step()
model.eval()
print("Training Complete.")

# 3. Comparative Benchmarking
explainer = DecisionInformationExplainer(model, lambda_fidelity=15.0, task='classification', device=device)

# Metrics accumulator
results = {
    "Method": [],
    "Fidelity": [],
    "Sparsity": [],
    "Insertion_AUC": [],
    "Deletion_AUC": [],
    "Time_sec": []
}

n_test = 5 # Evaluate on 5 samples for the benchmark
print(f"\nRunning Benchmarks on {n_test} instances...")

for i in range(n_test):
    x_sample = X_test_t[i].to(device)
    
    # DIxAI
    start = time.time()
    explanation = explainer.explain(x_sample, steps=500, verbose=False)
    end = time.time()
    
    ins_curve, del_curve = calculate_insertion_deletion_curves(model, x_sample, explanation.mask, n_steps=10)
    
    results["Method"].append("DIxAI")
    results["Fidelity"].append(explanation.fidelity_score)
    results["Sparsity"].append(explanation.info_score)
    results["Insertion_AUC"].append(calculate_auc(ins_curve))
    results["Deletion_AUC"].append(calculate_auc(del_curve))
    results["Time_sec"].append(end - start)

# 4. Summarize and Save
df_results = pd.DataFrame(results).groupby("Method").mean()
print("\n--- Higgs Benchmark Results Summary ---")
print(df_results)

results_path = os.path.join(os.path.dirname(__file__), '../results/higgs_results.csv')
os.makedirs(os.path.dirname(results_path), exist_ok=True)
df_results.to_csv(results_path)
print(f"Results saved to {results_path}")
