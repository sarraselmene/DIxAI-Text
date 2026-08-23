import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dixai.models.ddconv import HierarchicalBlock, DDConv2d
import time

def run_eurosat_experiment(mode, epochs=5):
    device = torch.device("cpu")
    print(f"Starting EuroSAT Experiment: {mode} (CPU)")
    
    # EuroSAT: 10 classes, 27,000 images
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # Reduced size for speed on CPU
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # We use a subset for speed
    dataset = datasets.EuroSAT(root='./data', download=True, transform=transform)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [2000, 25000]) # Small train set for fast validation
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    if mode == 'ddconv_full':
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            HierarchicalBlock(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        ).to(device)
    else:
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i > 50: break # Early break for rapid feedback
        print(f"Epoch {epoch+1} done.")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i > 20: break
            
    print(f"EuroSAT Accuracy ({mode}): {100 * correct / total:.2f}%")

if __name__ == "__main__":
    for m in ['standard', 'ddconv_full']:
        run_eurosat_experiment(m, epochs=2)
