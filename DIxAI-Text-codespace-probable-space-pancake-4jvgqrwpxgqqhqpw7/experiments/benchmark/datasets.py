import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
from typing import Tuple, Any

def get_tabular_dataset(name: str):
    """
    Loads and preprocesses tabular datasets.
    """
    if name == 'iris':
        data = load_iris()
        X, y = data.data, data.target
    elif name == 'adult':
        # Fetch from OpenML
        data = fetch_openml('adult', version=2, as_frame=True)
        X = data.frame.drop('class', axis=1)
        y = (data.frame['class'] == '>50K').astype(int).values
        # Simple encoding for demo (in practice needs better pipeline)
        X = pd.get_dummies(X).values
    elif name == 'german':
        data = fetch_openml('german_credit', version=1, as_frame=True)
        X = data.frame.drop('class', axis=1)
        y = (data.frame['class'] == 'good').astype(int).values
        X = pd.get_dummies(X).values
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long)
    )

def get_vision_dataloader(name: str, batch_size: int = 64, subset_size: int = None):
    """
    Returns dataloaders for vision datasets.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if name in ['mnist', 'fmnist'] else 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if name == 'mnist':
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif name == 'fmnist':
        train_set = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif name == 'cifar10':
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if subset_size:
        train_set = Subset(train_set, range(min(len(train_set), subset_size)))
        test_set = Subset(test_set, range(min(len(test_set), subset_size)))
        
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
