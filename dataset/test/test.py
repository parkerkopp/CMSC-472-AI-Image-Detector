"""
test dataset.py.
all the images have been removed to save space for google drive
"""
#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
import sys
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, dataset_dir)
from dataset import DatasetFFT  

annotations_file = "../data/ai-generated-images-vs-real-images/train/labels.csv"
root_dir = "../data/ai-generated-images-vs-real-images/train/images"

# Load dataset
dataset = DatasetFFT(annotations_file=annotations_file, root_dir=root_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Grab a single sample
sample = next(iter(dataloader))
image = sample['image'][0]  # remove batch dim
image_fft = {k: v[0] for k, v in sample['image_fft'].items()}  # remove batch dim
label = sample['label'][0]

print("Image shape:", image.shape)  # (C, H, W)
print("Label:", label)

img_np = image.permute(1, 2, 0).numpy()
if img_np.shape[2] == 1:  # grayscale
    img_np = img_np[:, :, 0]

if img_np.dtype == 'float32' and img_np.max() > 1.0:
    img_np = img_np / 255.0

os.makedirs('images', exist_ok=True)
image_file = 'images/sample_image.png'
plt.imsave(image_file, img_np, cmap='gray' if img_np.ndim == 2 else None)
print(f"Saved {image_file}")

# Save FFT magnitude and phase for all channels
for i in range(image_fft['magnitude'].shape[0]):
    mag = image_fft['magnitude'][i].numpy()
    phase = image_fft['phase'][i].numpy()

    # Center 0 freq
    mag = np.fft.fftshift(mag)
    phase = np.fft.fftshift(phase)
    
    # Normalize to 0..1
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min())
    phase_norm = (phase - phase.min()) / (phase.max() - phase.min())
    
    fft_mag_file = f'images/sample_fft_magnitude_c{i}.png'
    fft_phase_file = f'images/sample_fft_phase_c{i}.png'
    
    plt.imsave(fft_mag_file, mag_norm, cmap='gray')
    plt.imsave(fft_phase_file, phase_norm, cmap='gray')
    
    print(f"Saved FFT magnitude channel {i} as {fft_mag_file}")
    print(f"Saved FFT phase channel {i} as {fft_phase_file}")
