import requests
import os

def download_medical_samples_manual():
    urls = [
        ("https://upload.wikimedia.org/wikipedia/commons/e/e3/Chest_Xray_PA_3-8-2010.png", "chest_xray_normal_wikimedia.png"),
        ("https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/auntminnie-a-2020_01_28_20_51_20_51_0001.png", "chest_xray_pneumonia_1.png"),
        ("https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/auntminnie-b-2020_01_28_20_51_20_51_0001.png", "chest_xray_pneumonia_2.png")
    ]
    
    sample_dir = r"d:\Haythem\Temp\AAAA-New Research\decision_information_xai\data\medical_samples"
    os.makedirs(sample_dir, exist_ok=True)
    
    for url, filename in urls:
        save_path = os.path.join(sample_dir, filename)
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"  Success: {save_path}")
            else:
                print(f"  Failed (Status {response.status_code}): {url}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    download_medical_samples_manual()
