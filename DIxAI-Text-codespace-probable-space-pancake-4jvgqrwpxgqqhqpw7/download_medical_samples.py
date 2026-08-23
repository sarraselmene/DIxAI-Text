from datasets import load_dataset
import os
from PIL import Image

def download_medical_samples():
    print("Fetching Chest X-Ray samples (Pneumonia Detection)...")
    try:
        import requests
        # Use direct raw URLs from a well-known COVID/Pneumonia research repo
        urls = [
            "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000001-1.jpg",
            "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000001-2.jpg",
            "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/000001-3.jpg"
        ]
        
        sample_dir = r"d:\Haythem\Temp\AAAA-New Research\decision_information_xai\data\medical_samples"
        os.makedirs(sample_dir, exist_ok=True)
        
        count = 0
        for i, url in enumerate(urls):
            save_path = os.path.join(sample_dir, f"medical_{i:03d}.jpg")
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(r.content)
                print(f"  Saved {save_path}")
                count += 1
            
        print(f"Successfully downloaded {count} medical samples.")
    except Exception as e:
        print(f"Error downloading medical samples: {e}")

if __name__ == "__main__":
    download_medical_samples()
