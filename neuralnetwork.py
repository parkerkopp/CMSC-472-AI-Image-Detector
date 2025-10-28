import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data.is_this_ai_project.dataset.dataset import DatasetFFT

"""
Functions related to defining, training, and testing the neural network model and the dataset.
"""

#defines the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Process FFT magnitude
        self.fft_mag_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Process FFT phase
        self.fft_phase_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Combine features from FFT magnitude and phase (only Fourier domain)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 2, 256),  # *2 because we concatenate mag and phase features
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # binary classification
        )

    def forward(self, x):
        # Extract Fourier components from input dictionary
        fft_mag = x['image_fft']['magnitude']
        fft_phase = x['image_fft']['phase']

        # Process FFT magnitude and phase only
        mag_features = self.avgpool(self.fft_mag_conv(fft_mag))
        phase_features = self.avgpool(self.fft_phase_conv(fft_phase))

        # Concatenate features (batch x features)
        combined = torch.cat([
            mag_features.view(mag_features.size(0), -1),
            phase_features.view(phase_features.size(0), -1)
        ], dim=1)

        # Final classification
        return self.classifier(combined)

#train the model
def train(dataset, model, loss_fn, optimizer):
    # dataset -> should be a DataLoader that yields (input_dict, labels)
    size = len(dataset.dataset) if hasattr(dataset, 'dataset') else None
    running_loss = 0.0
    for batch_idx, (inp, labels) in enumerate(dataset):
        # Move to device
        labels = labels.to(device, dtype=torch.long)
        inp['image_fft']['magnitude'] = inp['image_fft']['magnitude'].to(device)
        inp['image_fft']['phase'] = inp['image_fft']['phase'].to(device)

        # Forward
        preds = model(inp)
        loss = loss_fn(preds, labels)
        running_loss += loss.item()

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            avg = running_loss / (batch_idx + 1)
            if size:
                seen = (batch_idx + 1) * inp['image_fft']['magnitude'].size(0)
                print(f"loss: {avg:>7f}  [{seen:>5d}/{size:>5d}]")
            else:
                print(f"loss: {avg:>7f}  [batch {batch_idx}]")
            
#tests the model and displays accuracy and loss          
def test(dataset, model, loss_fn):
    # dataset should be a DataLoader yielding (input_dict, labels)
    total = 0
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inp, labels in dataset:
            labels = labels.to(device, dtype=torch.long)
            inp['image_fft']['magnitude'] = inp['image_fft']['magnitude'].to(device)
            inp['image_fft']['phase'] = inp['image_fft']['phase'].to(device)

            outputs = model(inp)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    print(f"Test Error: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
     



"""
Main section of program to run the training and testing
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#Loads training and testing data with FFT
training_data = DatasetFFT(annotations_file='data\\is_this_ai_project\\dataset\\data\\ai-generated-images-vs-real-images\\train\\labels.csv',
                          root_dir='data\\is_this_ai_project\\dataset\\data\\ai-generated-images-vs-real-images\\train\\images')
test_data = DatasetFFT(annotations_file='data\\is_this_ai_project\\dataset\\data\\ai-generated-images-vs-real-images\\test\\labels.csv',
                       root_dir='data\\is_this_ai_project\\dataset\\data\\ai-generated-images-vs-real-images\\test\\images')
print(f"Dataset sizes - Training: {len(training_data)}, Test: {len(test_data)}")

# Collate function for batching FFT dataset samples
def fft_collate(batch):
    # batch: list of samples (dicts)
    mags = [b['image_fft']['magnitude'] for b in batch]
    phases = [b['image_fft']['phase'] for b in batch]
    labels = [b['label'] for b in batch]
    mag_batch = torch.stack(mags, dim=0)
    phase_batch = torch.stack(phases, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    inp = {'image_fft': {'magnitude': mag_batch, 'phase': phase_batch}}
    return inp, labels

# Create DataLoaders for batching and shuffling
batch_size = 64
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=fft_collate)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=fft_collate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)