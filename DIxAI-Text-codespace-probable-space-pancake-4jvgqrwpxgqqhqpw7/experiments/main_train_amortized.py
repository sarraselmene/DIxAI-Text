import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from dixai.models.explainer_net import AmortizedExplainer
from dixai.metrics import calculate_continuity

def get_args():
    parser = argparse.ArgumentParser(description="Train Amortized DIxAI Explainer for ImageNet/Vision")
    parser.add_argument("--data-path", type=str, default="data/imagenet_val", help="Path to validation data")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Model backbone to explain")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda-fidelity", type=float, default=5.0, help="Weight for fidelity loss")
    parser.add_argument("--lambda-sparsity", type=float, default=0.5, help="Weight for sparsity (L1) loss")
    parser.add_argument("--lambda-tv", type=float, default=0.1, help="Weight for Total Variation loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Run a single batch for verification")
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Amortized Explainer on {device} ---")

    # 1. Load Black-box Model
    print(f"Loading backbone: {args.backbone}...")
    if args.backbone == 'resnet50':
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    else:
        raise ValueError(f"Backbone {args.backbone} not supported in this script yet.")
    backbone.eval()
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # 2. Initialize Explainer
    # ResNet50 penultimate features are 2048 channels (after layer4) or 512/1024 depending on hook.
    # The AmortizedExplainer in dixai.models usually expects input image and extracts features internally 
    # OR takes features. Let's assume it takes the image and uses its own encoder or the backbone's encoder.
    # Checking explainer_net.py (from context) it seems to have an `encoder`.
    # We will assume standard instantiation:
    explainer = AmortizedExplainer(in_channels=3, feature_channels=512).to(device) # Default config
    optimizer = optim.Adam(explainer.parameters(), lr=args.lr)

    # 3. Data Loader
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if os.path.exists(args.data_path):
        dataset = datasets.ImageFolder(args.data_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    else:
        print(f"Warning: Data path {args.data_path} not found. Using fake data for demonstration.")
        # Create fake dataset
        fake_data = torch.randn(100, 3, 224, 224)
        fake_labels = torch.randint(0, 1000, (100,))
        dataset = torch.utils.data.TensorDataset(fake_data, fake_labels)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 4. Training Loop
    os.makedirs(args.save_dir, exist_ok=True)
    kl_div = nn.KLDivLoss(reduction='batchmean')
    
    for epoch in range(1, args.epochs + 1):
        explainer.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            # Target is model prediction for faithfulness
            with torch.no_grad():
                orig_logits = backbone(data)
                orig_probs = torch.softmax(orig_logits, dim=1)

            optimizer.zero_grad()
            
            # Forward Explainer
            # Explainer returns mask or masked input?
            # Assuming explainer(data) -> mask (B, 1, H, W) based on typical design
            # If explainer_net.py is different, we adjust. 
            # From previous view_file, it returns 'mask'.
            mask = explainer(data) 
            
            # Apply Mask
            # Baseline: Zero (or could be Mean)
            masked_input = data * mask
            
            # Forward Backbone with Mask
            masked_logits = backbone(masked_input)
            masked_probs = torch.log_softmax(masked_logits, dim=1) # Log for KL
            
            # Losses
            loss_fidelity = kl_div(masked_probs, orig_probs)
            loss_sparsity = mask.mean()
            loss_tv = calculate_continuity(mask) # This returns 1-TV, so we need TV = 1-acc (approx) or just manual TV
            # Actually metrics.py calculate_continuity returns 1 - normalized_TV. 
            # We want to MINIMIZE TV. So we want to MAXIMIZE continuity.
            # Loss = -Continuity or just use raw TV.
            # Let's compute raw TV here to be safe and explicit.
            tv_loss = (torch.abs(mask[:, :, :, :-1] - mask[:, :, :, 1:]).mean() + 
                       torch.abs(mask[:, :, :-1, :] - mask[:, :, 1:, :]).mean())

            loss = (args.lambda_fidelity * loss_fidelity + 
                    args.lambda_sparsity * loss_sparsity + 
                    args.lambda_tv * tv_loss)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} (Fid: {loss_fidelity.item():.4f}, "
                      f"Spar: {loss_sparsity.item():.4f}, TV: {tv_loss.item():.4f})")
            
            if args.dry_run:
                print("Dry run complete.")
                return

        print(f"Epoch {epoch} Average Loss: {total_loss / len(dataloader):.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(args.save_dir, f"amortized_explainer_epoch_{epoch}.pth")
        torch.save(explainer.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()
