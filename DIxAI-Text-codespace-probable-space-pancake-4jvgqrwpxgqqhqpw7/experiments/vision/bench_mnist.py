
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from dixai import DecisionInformationExplainer, plot_feature_importance

# 1. Setup Data (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

# 2. Define Model (Simple CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.fc1 = nn.Linear(16*13*13, 10) # 28x28 -> 26x26 (conv) -> 13x13 (pool)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return self.log_softmax(x)

# Note: In a real benchmark, we would load a pre-trained model. 
# Here we just initialize random weights for demonstration of methodology unless training is fast.
# Let's train for 1 epoch quickly.
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

print("Training CNN...")
model.train()
for i, (images, labels) in enumerate(trainloader):
    if i > 100: break # Quick train
    optimizer.zero_grad()
    loss = criterion(model(images), labels)
    loss.backward()
    optimizer.step()
print("Training Complete.")

# 3. Explain Instance
model.eval()
images, labels = next(iter(testloader))
x_sample = images[0] # (1, 28, 28)

print(f"\nExplaining MNIST digit: {labels[0].item()}")
explainer = DecisionInformationExplainer(model, lambda_fidelity=20.0, task='classification')

explanation = explainer.explain(x_sample, steps=500, lr=0.1, verbose=True)

print(f"Fidelity: {explanation.fidelity_score:.3f}")
print(f"Sparsity: {1.0 - explanation.info_score:.3f}")

# 4. Visualization & Comparison
# Original
img_orig = x_sample.permute(1, 2, 0).numpy()
# Mask (Heatmap)
mask = explanation.mask.numpy().squeeze()
# Masked Image
masked_img = img_orig.squeeze() * mask

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(img_orig.squeeze(), cmap='gray')
ax[0].set_title("Original")
ax[1].imshow(mask, cmap='viridis')
ax[1].set_title("Decision Info (Mask)")
ax[2].imshow(masked_img, cmap='gray')
masked_in = x_sample * explanation.mask
if masked_in.dim() == 3:
    masked_in = masked_in.unsqueeze(0)
ax[2].set_title(f"Masked (Pred: {torch.argmax(torch.exp(model(masked_in)))})")

save_dir = os.path.join(os.path.dirname(__file__), '../results/vision')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'mnist_explanation.png')
plt.savefig(save_path)
print(f"Saved visualization to {save_path}")

# 5. Baseline: Random Masking
print("\nRunning Baselines...")
rand_mask = torch.rand_like(explanation.mask)
masked_x_rand = x_sample * rand_mask
with torch.no_grad():
    if masked_x_rand.dim() == 3:
        masked_x_rand = masked_x_rand.unsqueeze(0)
    y_pred_rand = model(masked_x_rand)
    fid_rand = torch.exp(y_pred_rand)[0, labels[0]].item() # Prob of true class
print(f"Random Mask Fidelity: {fid_rand:.3f}")

# 6. Baseline: Saliency (Input * Gradient)
x_sample.requires_grad = True
out = model(x_sample.unsqueeze(0))
loss = out[0, labels[0]]
loss.backward()
saliency = x_sample.grad.abs()
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10) # Normalize
print(f"Saliency Map Generated.")

# Overlay Saliency
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(img_orig.squeeze(), cmap='gray')
ax[0].set_title("Original")
ax[1].imshow(saliency.squeeze(), cmap='jet')
ax[1].set_title("Gradient Saliency")
ax[2].imshow(img_orig.squeeze() * (saliency.squeeze() > 0.5).numpy(), cmap='gray') # Thresholded
ax[2].set_title("Saliency Thresholded")

save_path_base = os.path.join(save_dir, 'mnist_baselines.png')
plt.savefig(save_path_base)
print(f"Saved baselines to {save_path_base}")
