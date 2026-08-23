import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from dixai.models.ddconv import HierarchicalBlock

def run_data_efficiency(fraction=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    
    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    num_samples = int(len(full_train) * fraction)
    indices = torch.randperm(len(full_train))[:num_samples]
    subset_train = Subset(full_train, indices)
    
    train_loader = DataLoader(subset_train, batch_size=128, shuffle=True)
    
    # Model: DD-Conv Hierarchical Block
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        HierarchicalBlock(64), 
        nn.AdaptiveAvgPool2d(1), 
        nn.Flatten(), 
        nn.Linear(64, 10)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    # 1 epoch fast check
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i > 10: break # Fast proxy
        
    print(f"Data Efficiency ({fraction*100}%): Training step successful.")

if __name__ == "__main__":
    for f in [1.0, 0.5, 0.1]:
        run_data_efficiency(f)
