"""
Contains VAE classes and experiments
"""
import argparse
import dataloader
import numpy as np
import pickle
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

# convolutional variational autoencoder
class VAE(nn.Module):

    def __init__(self, in_channels, latent_dim, 
                enc_stride, enc_kernel_size, 
                dec_stride, dec_kernel_size):

        super(VAE, self).__init__()

        hidden_dim = 23104
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
            nn.Conv2d(in_channels, 16, enc_kernel_size, enc_stride),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(16, 32, enc_kernel_size, enc_stride),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(32, 64, enc_kernel_size, enc_stride),
            nn.ReLU(),
            nn.MaxPool2d(5, stride=3)
        )

        self.mu_fc = nn.Linear(hidden_dim, latent_dim) # predicting mu of gaussian
        self.logvar_fc = nn.Linear(hidden_dim, latent_dim) # predicting log var of gaussian
        self.latent_to_hidden_fc = nn.Linear(latent_dim, hidden_dim) # use before decoder

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, dec_kernel_size, dec_stride),
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
        hidden_rep = self.encoder(x).contiguous().view(self.batch_size, -1)
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

def bce_loss(original_x, recon_x):
    bce = F.binary_cross_entropy(recon_x, original_x, size_average=False) # reconstruction loss
    return bce

def mse_loss(original_x, recon_x):
    mse = F.mse_loss(recon_x, original_x, reduction='mean')
    return mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--in_channels', type=int, default=1) #3
    parser.add_argument('--latent_dim', type=int, default=2000)
    parser.add_argument('--enc_stride', type=int, default=2)
    parser.add_argument('--enc_kernel_size', type=int, default=5)
    parser.add_argument('--dec_stride', type=int, default=2)
    parser.add_argument('--dec_kernel_size', type=int, default=5)
    parser.add_argument('--learning_rate', type=int, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=500)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data is a dict
    # data = dataloader.load_data("dataset/images")

    with open('data.pickle', 'rb') as handle:
        data = pickle.load(handle)

    print("Loaded data.")

    organelles = [] # ground truth in order
    gene_names = [] # ground truth in order
    images = []

    for gene_name, metadata in data.items():
        
        for image in data[gene_name]["image_arr"]:
            gene_names.append(gene_name)
            organelles.append(data[gene_name]["organelle"])
            images.append(image)

    print("Have images.")

    # naive train-test split, replace with protein sequence similarity split once esm features are added
    distinct_genes = list(set(gene_names))
    genes_train = distinct_genes[:int(0.75*len(distinct_genes))]
    genes_test = distinct_genes[int(0.75*len(distinct_genes)):]
    first_test_index = gene_names.index(genes_test[0])

    organelles_train = organelles[:first_test_index]
    gene_names_train = gene_names[:first_test_index]
    images_train = images[:first_test_index]

    organelles_test = organelles[first_test_index:]
    gene_names_test = organelles[first_test_index:]
    images_test = images[first_test_index:]

    print("Split data.")

    model = VAE(in_channels=args.in_channels, latent_dim=args.latent_dim, 
                enc_stride=args.enc_stride, enc_kernel_size=args.enc_kernel_size, 
                dec_stride=args.dec_stride, dec_kernel_size=args.dec_kernel_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Train/Test Dataset Sizes: {len(organelles_train)}/{len(organelles_test)}")

    # train model
    for epoch in range(args.num_epochs):

        for i in range(0, len(images_train), args.batch_size):
            batch_number = i // args.batch_size
            print(f"Batch {batch_number} on epoch {epoch}.")
            
            try:
                reshaped_arrs = [array.reshape((1, 2048, 2048, 3)) for array in images_train[i : i+ args.batch_size]]
                batch = np.concatenate(reshaped_arrs, axis=0)
                batch = np.transpose(batch, (0,3,1,2))
                batch = batch[:,1:2,:,:] #getting green slice
            except Exception as e:
                reshaped_arrs = [array.reshape((1, 2048, 2048, 3)) for array in images_train[i : ]]
                batch = np.concatenate(reshaped_arrs, axis=0)
                batch = np.transpose(batch, (0,3,1,2))
                batch = batch[:,1:2,:,:] #getting green slice

            batch = np.where(batch > 0.1, 1, 0)

            if epoch % 10 == 0 and i % 100 == 0:
                images_original = np.transpose(batch, (0, 2, 3, 1))
                true_img = images_original[0]
                true_img = (true_img * 255).astype(np.uint8)
                print(true_img.shape)
                im = Image.fromarray(true_img[:,:,0]) #.convert('RGB')
                im.save(f'sample_imgs_green_train/batch_{epoch}_{i}.jpeg')

            batch = torch.from_numpy(batch).to(device).float()

            output, mu, logvar = model(batch) #output size is (5, 3, 1024,1024)
            output = F.interpolate(output, size=(2048, 2048), mode='bilinear', align_corners=False)
            #loss, bce, kl = kl_loss(batch, output, mu, logvar)
            #print(f"Loss = {loss}, BCE = {bce}, KL-divergence = {kl}.")
            # loss = mse_loss(batch, output)
            loss = bce_loss(batch, output)
            print(f"Loss = {loss}.")

            if epoch % 10 == 0 and i % 100 == 0:
                images_original = np.transpose(output.cpu().detach().numpy(), (0, 2, 3, 1))
                true_img = images_original[0]
                true_img = (true_img * 255).astype(np.uint8)
                print(true_img.shape)
                im = Image.fromarray(true_img[:,:,0]) #.convert('RGB')
                im.save(f'sample_imgs_green_train/output_{epoch}_{i}.jpeg')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == args.num_epochs - 1:
            filename = f'model_checkpoints/green_autoencoder_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, filename)

    # checkpoint = torch.load("model_checkpoints/vanilla_autoencoder_mseloss_100dp_epoch_499.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])

    # # cut off test set
    # images_test = images_test[:20]
    
    # # hold out test set
    # for i in range(0, len(images_test), args.batch_size):
    #     batch_number = i // args.batch_size
    #     print(f"Batch {batch_number} on test set.")
        
    #     try:
    #         reshaped_arrs = [array.reshape((1, 2048, 2048, 3)) for array in images_test[i : i + args.batch_size]]
    #         batch = np.concatenate(reshaped_arrs, axis=0)
    #         batch = np.transpose(batch, (0,3,1,2))
    #     except Exception as e:
    #         reshaped_arrs = [array.reshape((1, 2048, 2048, 3)) for array in images_test[i : ]]
    #         batch = np.concatenate(reshaped_arrs, axis=0)
    #         batch = np.transpose(batch, (0,3,1,2))

    #     batch = torch.from_numpy(batch).to(device)
    #     output, mu, logvar = model(batch) #output size is (5, 3, 1024,1024)
    #     output = F.interpolate(output, size=(2048, 2048), mode='bilinear', align_corners=False)

    #     images_original = np.transpose(batch.cpu().detach().numpy(), (0, 2, 3, 1))
    #     true_img = images_original[0]
    #     true_img = (true_img * 255).astype(np.uint8)
    #     print(true_img.shape)
    #     im = Image.fromarray(true_img).convert('RGB')
    #     im.save(f'sample_imgs_val/batch_{i}.jpeg')

    #     images_original = np.transpose(output.cpu().detach().numpy(), (0, 2, 3, 1))
    #     true_img = images_original[0]
    #     true_img = (true_img * 255).astype(np.uint8)
    #     print(true_img.shape)
    #     im = Image.fromarray(true_img).convert('RGB')
    #     im.save(f'sample_imgs_val/output_{i}.jpeg')