from datasets import load_dataset
import os
from PIL import Image

def download_samples():
    print("Fetching Mini-ImageNet samples from Hugging Face...")
    try:
        # Load a few samples from the validation or test split
        dataset = load_dataset("timm/mini-imagenet", split="test", streaming=True)
        
        sample_dir = "data/imagenet_samples"
        os.makedirs(sample_dir, exist_ok=True)
        
        count = 0
        limit = 1000 # Sample limit for validation
        
        for i, example in enumerate(dataset):
            if i >= limit:
                break
            
            img = example['image']
            label = example['label']
            
            save_path = os.path.join(sample_dir, f"sample_{i:04d}.jpg")
            img.convert("RGB").save(save_path)
            if i % 10 == 0:
                print(f"Saved {save_path} (Label: {label})")
            count += 1
            
        print(f"Successfully downloaded {count} samples.")
    except Exception as e:
        print(f"Error downloading samples: {e}")

if __name__ == "__main__":
    download_samples()
