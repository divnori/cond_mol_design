"""
Contains VAE classes and experiments

1. Build vanilla vae and visualize latent space
2. Hyperparameter sweep - parameters in __init__
3. Add esm embeddings
4. Experiment with loss functions
5. Compare to linear pca
"""

import dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# convolutional variational autoencoder
class VAE(nn.Module):

    def __init__(self, in_channels=3, hidden_channels=32, hidden_dim=1024, latent_dim=256, stride=5, kernel_size=4):

        # Conv2D layer - (C_in, C_out, kernel size)
        # input - (N, C_in, H, W) = (num data points, 3, 2048, 2048)
        self.enc_conv1 = nn.Conv2D(in_channels, hidden_channels)



if __name__ == "__main__":

    # data is a dict
    data = dataloader.load_data("dataset/images")

