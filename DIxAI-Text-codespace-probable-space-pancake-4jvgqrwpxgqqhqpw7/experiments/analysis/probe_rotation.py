import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()

img_path = "d:/Haythem/Temp/AAAA-New Research/decision_information_xai/data/imagenet_samples/sample_0000.jpg"
img = Image.open(img_path).convert('RGB')
crop_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])
original_crop = crop_transform(img)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

print(f"{'Angle':<10} | {'Top Class':<15} | {'Conf':<10}")
print("-" * 40)

for angle in [0, 15, 30, 45]:
    rot_pil = original_crop.rotate(angle, fillcolor=(124, 116, 104))
    rot_tensor = normalize(to_tensor(rot_pil)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(rot_tensor)
        probs = torch.softmax(out, dim=-1)
        conf, pred = probs.max(dim=-1)
        print(f"{angle:<10} | {pred.item():<15} | {conf.item():.4f}")
