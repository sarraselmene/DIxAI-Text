import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dixai.models.ddconv import HierarchicalBlock, DDConv2d
import time

class BaselineNet(nn.Module):
    def __init__(self, mode='standard', num_classes=100):
        super(BaselineNet, self).__init__()
        self.mode = mode
        
        # Initial features
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Comparison Block
        if mode == 'ddconv_full':
            self.block = HierarchicalBlock(64)
        elif mode == 'ddconv_only':
            self.block = nn.Sequential(DDConv2d(64, 64), nn.BatchNorm2d(64), nn.ReLU())
        elif mode == 'std_conv':
            self.block = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        elif mode == 'depthwise':
            self.block = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
                nn.Conv2d(64, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        else:
            raise ValueError("Unknown mode")
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def run_experiment(mode, epochs=5):
    print(f"Starting Experiment: {mode}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    
    model = BaselineNet(mode=mode).to(device)
    
    # Complexity analysis
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    total_time = time.time() - start_time
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Training Time: {total_time:.2f}s")
    
    return {
        "mode": mode,
        "params": num_params,
        "accuracy": accuracy,
        "time": total_time
    }

if __name__ == "__main__":
    results = []
    for m in ['std_conv', 'depthwise', 'ddconv_only', 'ddconv_full']:
        res = run_experiment(m, epochs=2) # Short run for verification
        results.append(res)
    
    print("\nFinal Results Table:")
    print("Mode | Params | Accuracy | Time")
    for r in results:
        print(f"{r['mode']} | {r['params']} | {r['accuracy']}% | {r['time']}s")
