#!/usr/bin/env python3
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

def load_image(img_path):
    """
    Loads and resizes an image as a numpy array.
    All images will be resized to 1024x1024 pixels.

    Args:
        img_path: Path to the image file

    Returns:
        numpy array as (1024 x 1024 x 1) in grayscale format.
    """
    # Force conversion to grayscale (L) and resize to fixed size of 1024x1024
    img = Image.open(img_path).convert('L')
    img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
    arr = np.asarray(img)
    
    # Ensure a channel axis is present
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    return arr

def fft(img):
    """ 
    Compute FFT per channel 
    
    returns dictionary with magnitude and angle as the keys, and the per-channel fft as the value
    """ 
    mag_channels, phase_channels = [], []
    for c in range(img.shape[2]):
        fft_img = np.fft.fft2(img[:, :, c])
        mag = np.abs(fft_img)
        phase = np.angle(fft_img)
        
        mag_channels.append(mag)
        phase_channels.append(phase)

    # Both attributes are stored as (H x W x C) where C is channels.
    # Perhaps helpful for direct comparison between fft and original image.
    # If not, then can change axis to 0 for (C x H x W).
    magnitude_image = np.stack(mag_channels, axis=-1) 
    phase_image = np.stack(phase_channels, axis=-1)

    return {'magnitude': magnitude_image, 'phase': phase_image}
 
class DatasetFFT(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = root_dir 
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = load_image(img_path)
        image_fft = fft(image)

        # Convert only the FFT outputs to tensors. We intentionally do NOT
        # include the original spatial-domain image in the returned sample.
        # Channels are converted to (C x H x W).
        image_fft = {k: torch.tensor(v, dtype=torch.float32).permute(2, 0, 1)
                     for k, v in image_fft.items()}

        label = self.img_labels.iloc[idx, 1]
        sample = {'image_fft': image_fft, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample 
        
