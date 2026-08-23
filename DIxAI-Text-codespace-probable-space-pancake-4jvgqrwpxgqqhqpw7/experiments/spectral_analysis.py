import torch
import numpy as np
import matplotlib.pyplot as plt
from dixai.models.ddconv import DDConv2d

def analyze_spectral_response():
    ddconv = DDConv2d(1, 1, kernel_size=11, directions=4)
    kernels = ddconv.kernels.detach().cpu().numpy() # (4, 1, 11, 11)
    
    titles = ['0° (Horizontal)', '90° (Vertical)', '45° (Diagonal)', '135° (Anti-diagonal)']
    
    plt.figure(figsize=(15, 8))
    for i in range(4):
        k = kernels[i, 0]
        
        # Spatial Domain
        plt.subplot(2, 4, i+1)
        plt.imshow(k, cmap='RdBu')
        plt.title(f"Spatial: {titles[i]}")
        plt.colorbar()
        
        # Frequency Domain (FFT2)
        fft_k = np.fft.fftshift(np.fft.fft2(k, s=(64, 64)))
        magnitude = np.abs(fft_k)
        
        plt.subplot(2, 4, i+5)
        plt.imshow(magnitude, cmap='viridis')
        plt.title(f"Spectral Response")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('d:\\Haythem\\Temp\\AAAA-New Research\\decision_information_xai\\experiments\\results\\ddconv_spectral.png')
    print("Spectral analysis saved to ddconv_spectral.png")

if __name__ == "__main__":
    analyze_spectral_response()
