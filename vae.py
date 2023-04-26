"""
Contains VAE classes and experiments

1. Build vanilla vae and visualize latent space
2. Hyperparameter sweep - parameters in __init__
3. Add esm embeddings
4. Experiment with loss functions
5. Compare to linear pca
"""
import argparse
import dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import random

# convolutional variational autoencoder
class VAE(nn.Module):

    def __init__(self, in_channels, latent_dim, 
                enc_stride, enc_kernel_size, 
                dec_stride, dec_kernel_size):

        super(VAE, self).__init__()

        hidden_dim = 128 * enc_kernel_size**4
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.enc_stride = enc_stride
        self.enc_kernel_size = enc_kernel_size
        self.dec_stride = dec_stride
        self.dec_kernel_size = dec_kernel_size
        self.hidden_dim = hidden_dim

        # Conv2D layer - (C_in, C_out, kernel size)
        # input - (N, C_in, H, W) = (num data points, 3, 2048, 2048)

        # output of encoder is (N, output channels, enc_kernel_size^2, enc_kernel_size^2)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, enc_kernel_size, enc_stride),
            nn.ReLU(),
            nn.Conv2d(32, 64, enc_kernel_size, enc_stride),
            nn.ReLU(),
            nn.Conv2d(64, 128, enc_kernel_size, enc_stride),
            nn.ReLU()
        )

        self.mu_fc = nn.Linear(hidden_dim, latent_dim) # predicting mu of gaussian
        self.logvar_fc = nn.Linear(hidden_dim, latent_dim) # predicting log var of gaussian
        self.latent_to_hidden_fc = nn.Linear(latent_dim, hidden_dim) # use before decoder

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, dec_kernel_size, dec_stride),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, dec_kernel_size, dec_stride),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, dec_kernel_size, dec_stride),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, dec_kernel_size, dec_stride),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, dec_kernel_size, dec_stride),
            nn.Sigmoid() # output between 0 and 1 since original images are normalized
        )

    def reparameterize(self, mu, logvar):
        # reparameterization trick - mu + sigma*epsilon where epsilon is a random noise vector

        batch_size, latent_dim = mu.size()
        epsilon = torch.randn(batch_size, latent_dim).to(device)
        sigma = torch.exp(0.5 * logvar)
        z = mu + sigma * epsilon
        return z

    def encode(self, x):
        # x is batch of images as a tensor of shape (N, 3, 2048, 2048)

        self.batch_size = x.shape[0]
        hidden_rep = self.encoder(x).view(-1, self.hidden_dim)
        mu = self.mu_fc(hidden_rep)
        logvar = self.logvar_fc(hidden_rep)
        return mu, logvar

    def decode(self, z):
        hidden_rep = self.latent_to_hidden_fc(z).view(self.batch_size, self.hidden_dim, 1, 1)
        output = self.decoder(hidden_rep)
        return output

    def forward(self, x):

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar

def kl_loss(original_x, recon_x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, original_x, size_average=False) # reconstruction loss
    kl_divergence = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp()) # regularization loss
    return bce + kl_divergence, bce, kl_divergence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--enc_stride', type=int, default=5)
    parser.add_argument('--enc_kernel_size', type=int, default=4)
    parser.add_argument('--dec_stride', type=int, default=4)
    parser.add_argument('--dec_kernel_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=int, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data is a dict
    data = dataloader.load_data("dataset/images")

    organelles = [] # ground truth in order
    gene_names = [] # ground truth in order
    images = []

    for gene_name, metadata in data.items():
        
        for image in data[gene_name]["image_arr"]:
            gene_names.append(gene_name)
            organelles.append(data[gene_name]["organelle"])
            images.append(image)

    # naive train-test split, replace with protein sequence similarity split once esm features are added
    distinct_genes = list(set(gene_names))
    genes_train = distinct_genes[:int(0.75*len(distinct_genes))]
    genes_test = gene_names[int(0.75*len(distinct_genes)):]
    first_test_index = gene_names.index(genes_test[-1])

    organelles_train = organelles[:first_test_index]
    gene_names_train = gene_names[:first_test_index]
    images_train = images[:first_test_index]

    organelles_test = organelles[first_test_index:]
    gene_names_test = organelles[first_test_index:]
    images_test = images[first_test_index:]

    model = VAE(in_channels=args.in_channels, latent_dim=args.latent_dim, 
                enc_stride=args.enc_stride, enc_kernel_size=args.enc_kernel_size, 
                dec_stride=args.dec_stride, dec_kernel_size=args.dec_kernel_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # train model
    for epoch in range(args.num_epochs):

        for i in range(0, len(images_train), args.batch_size):
            batch_number = i // args.batch_size
            print(f"Batch {batch_number} on epoch {epoch}.")
            
            try:
                batch = np.concatenate(images_train[i : i + args.batch_size], axis=0).reshape((args.batch_size, 3, 2048, 2048))
            except Exception as e:
                num_points = len(images_train[i : ])
                batch = np.concatenate(images_train[i : ], axis=0).reshape((num_points, 3, 2048, 2048))

            batch = torch.from_numpy(batch).to(device)
            output, mu, logvar = model(batch) #output size is (5, 3, 1024,1024)
            output = F.interpolate(output, size=(2048, 2048), mode='bilinear', align_corners=False)
            loss, bce, kl = kl_loss(batch, output, mu, logvar)
            print(f"Loss = {loss}, BCE = {bce}, KL-divergence = {kl}.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == args.num_epochs - 1:
            filename = f'model_checkpoints/vanilla_autoencoder_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, filename)

    # checkpoint = torch.load("model_checkpoints/vanilla_autoencoder_epoch_49.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # hold out test set
    for i in range(0, len(images_test), args.batch_size):
        batch_number = i // args.batch_size
        print(f"Batch {batch_number} on test set.")
        
        try:
            batch = np.concatenate(images_test[i : i + args.batch_size], axis=0).reshape((args.batch_size, 3, 2048, 2048))
        except Exception as e:
            num_points = len(images_test[i : ])
            batch = np.concatenate(images_test[i : ], axis=0).reshape((num_points, 3, 2048, 2048))

        batch = torch.from_numpy(batch).to(device)
        output, mu, logvar = model(batch) #output size is (5, 3, 1024,1024)
        output = F.interpolate(output, size=(2048, 2048), mode='bilinear', align_corners=False)

        batch = batch * 255
        output = output * 255

        print(batch.shape)
        print(torch.min(batch))
        print(torch.max(batch))

        print(output.shape)
        print(torch.min(output))
        print(torch.max(output))

        save_image(batch[0].data.cpu(), f'sample_image_gt_{i}.png')
        save_image(output[0].data.cpu(), f'sample_image_pred_{i}.png')