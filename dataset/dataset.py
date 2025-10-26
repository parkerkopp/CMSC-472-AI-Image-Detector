#!/usr/bin/env python3
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

def load_image(img_path):
    """
    Loads an image as a numpy array 

    returns numpy array as (H x W x C). 
    Grayscale images have 1 color channel, so (H x W x 1).
    All other image types are converted to RGB in the format_data.py script, and so have (H x W x 3).
    """
    img = Image.open(img_path)
    if img.mode == "L":
        img = np.expand_dims(np.asarray(img), axis=-1)
    return np.asarray(img)

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

        # Convert image and fft to tensors. 
        # Channels have to be changed to (C x H x W).
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) 
        image_fft = {k: torch.tensor(v, dtype=torch.float32).permute(2, 0, 1) 
                     for k, v in image_fft.items()}
        
        label = self.img_labels.iloc[idx, 1]
        sample = {'image': image, 'image_fft': image_fft, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample 
        
