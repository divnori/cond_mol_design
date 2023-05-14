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

    def forward(self, x1, x2):
        mu1, logvar1 = self.encode(x1)
        mu2, logvar2 = self.encode(x2)
        z1 = self.reparameterize(mu1, logvar1)
        output1 = self.decode(z1)
        z2 = self.reparameterize(mu2, logvar2)
        output2 = self.decode(z2)
        return output1, output2, z1, z2

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

def contrastive_loss(original_x1, original_x2, recon_x1, recon_x2, latent_z1, latent_z2, ct1, ct2, org1, org2):
    mse1 = F.mse_loss(recon_x1, original_x1, reduction='mean')
    mse2 = F.mse_loss(recon_x2, original_x2, reduction='mean')
    euc_dist = torch.nn.functional.pairwise_distance(latent_z1, latent_z2)

    if ct1 == ct2:
        dct = 0
    else:
        dct = 1
    if org1 == org2:
        dorg = 0
    else:
        dorg = 1

    if dct == 0:
      cont1 = torch.mean(torch.pow(euc_dist, 2))  # distance squared
    else:  # dct == 1
      delta = 0.1 - euc_dist  # sort of reverse distance, m is 0.1
      delta = torch.clamp(delta, min=0.0, max=None)
      cont1 = torch.mean(torch.pow(delta, 2))  # mean over all rows

    if dorg == 0:
      cont2 = torch.mean(torch.pow(euc_dist, 2))  # distance squared
    else:  # dorg == 1
      delta = 0.1 - euc_dist  # sort of reverse distance
      delta = torch.clamp(delta, min=0.0, max=None)
      cont2 = torch.mean(torch.pow(delta, 2))  # mean over all rows

    return mse1 + mse2 + cont1 + cont2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--in_channels', type=int, default=3) 
    parser.add_argument('--latent_dim', type=int, default=2000)
    parser.add_argument('--enc_stride', type=int, default=2)
    parser.add_argument('--enc_kernel_size', type=int, default=5)
    parser.add_argument('--dec_stride', type=int, default=5)
    parser.add_argument('--dec_kernel_size', type=int, default=7)
    parser.add_argument('--learning_rate', type=int, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data is a dict
    # data = dataloader.load_data("dataset/images")

    with open('data.pickle', 'rb') as handle:
        data = pickle.load(handle)

    print("Loaded data.")

    organelles = [] # ground truth in order
    gene_names = [] # ground truth in order
    organs = [] # ground truth in order
    images = []

    for gene_name, metadata in data.items():
        
        for image in data[gene_name]["image_arr"]:
            gene_names.append(gene_name)
            organelles.append(data[gene_name]["organelle"])
            organs.append(data[gene_name]["organ"])
            images.append(image)

    first_test_index = int(len(images) * 0.75)

    organelles_train = organelles[:first_test_index]
    gene_names_train = gene_names[:first_test_index]
    organs_train = organs[:first_test_index]
    images_train = images[:first_test_index]

    organelles_test = organelles[first_test_index:]
    gene_names_test = organelles[first_test_index:]
    images_test = images[first_test_index:]
    organs_test = organs[first_test_index:]

    # even for contrastive loss
    if int(len(images_train)/2) != 0:
        images_train = images_train[:-1]
        gene_names_train = gene_names_train[:-1]
        organelles_train = organelles_train[:-1]
        organs_train = organs_train[:-1]

    if int(len(images_test)/2) != 0:
        images_test = images_test[:-1]
        gene_names_test = gene_names_test[:-1]
        organelles_test = organelles_test[:-1]
        organs_test = organs_test[:-1]

    train = list(zip(images_train, gene_names_train, organelles_train))
    random.shuffle(train)
    images_train, gene_names_train, organelles_train = zip(*train)
    test = list(zip(images_test, gene_names_test, organelles_test))
    random.shuffle(test)
    images_test, gene_names_test, organelles_test = zip(*test)

    print(f"Train/Test Dataset Sizes: {len(organelles_train)}/{len(organelles_test)}")

    model = VAE(in_channels=args.in_channels, latent_dim=args.latent_dim, 
                enc_stride=args.enc_stride, enc_kernel_size=args.enc_kernel_size, 
                dec_stride=args.dec_stride, dec_kernel_size=args.dec_kernel_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # # train model
    # losses = []
    # train_embeddings = []
    # for epoch in range(args.num_epochs):

    #     for i in range(0, len(images_train), args.batch_size):
    #         batch_number = i // args.batch_size
    #         print(f"Batch {batch_number} on epoch {epoch}.")
           
    #         reshaped_arrs = [array.reshape((1, 2048, 2048, 3)) for array in images_train[i : i+ args.batch_size]]
    #         batch = np.concatenate(reshaped_arrs, axis=0)
    #         batch = np.transpose(batch, (0,3,1,2))

    #         if epoch % 10 == 0 and i % 100 == 0:
    #             images_original = np.transpose(batch, (0, 2, 3, 1))
    #             true_img = images_original[0]
    #             true_img = (true_img * 255).astype(np.uint8)
    #             im = Image.fromarray(true_img).convert('RGB')
    #             im.save(f'sample_imgs_train/batch_{epoch}_{i}.jpeg')

    #         batch = torch.from_numpy(batch).to(device)

    #         y1, y2, z1, z2 = model(batch[0:1],batch[1:2]) #output size is (2, 3, 1024,1024)
    #         y1 = F.interpolate(y1, size=(2048, 2048), mode='bilinear', align_corners=False)
    #         y2 = F.interpolate(y2, size=(2048, 2048), mode='bilinear', align_corners=False)
    #         #loss, bce, kl = kl_loss(batch, output, mu, logvar)
    #         #print(f"Loss = {loss}, BCE = {bce}, KL-divergence = {kl}.")
    #         # loss = mse_loss(batch, output)
    #         # loss = bce_loss(batch, output)
    #         print(organs_train[i], organs_train[i+1], organelles_train[i], organelles_train[i+1])
    #         loss = contrastive_loss(batch[0:1],batch[1:2], y1, y2, z1, z2, organs_train[i], organs_train[i+1], organelles_train[i], organelles_train[i+1])
    #         print(f"Loss = {loss}.")
    #         losses.append(loss)

    #         if epoch == 9:
    #             train_embeddings.append({"organelle":organelles_train[i], "organ":organs_train[i], "embedding": z1.cpu().detach().numpy(), "gene": gene_names_train[i]})
    #             train_embeddings.append({"organelle":organelles_train[i+1], "organ":organs_train[i+1], "embedding": z2.cpu().detach().numpy(), "gene": gene_names_train[i+1]})

    #         if epoch % 9 == 0 and i % 100 == 0:
    #             images_predicted = np.transpose(y1.cpu().detach().numpy(), (0, 2, 3, 1))
    #             pred_img = images_predicted[0]
    #             pred_img = (pred_img * 255).astype(np.uint8)
    #             im = Image.fromarray(pred_img).convert('RGB')
    #             im.save(f'sample_imgs_train/output_{epoch}_{i}.jpeg')

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     if epoch % 9 == 0 and epoch != 0:
    #         filename = f'model_checkpoints/contrastive_autoencoder_epoch_{epoch}.pt'
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': loss,
    #             }, filename)
            
    #         with open('figure_data/loss_curve_data.pickle', 'wb') as handle:
    #             pickle.dump(losses, handle)

    #         with open('figure_data/train_embeddings.pickle', 'wb') as handle:
    #             pickle.dump(train_embeddings, handle)

    checkpoint = torch.load("model_checkpoints/contrastive_autoencoder_epoch_99.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    # # hold out test set
    # val_embeddings = []
    # for i in range(0, len(images_test), args.batch_size):
    #     try:
    #         batch_number = i // args.batch_size
    #         print(f"Batch {batch_number} on test set.")
            
    #         reshaped_arrs = [array.reshape((1, 2048, 2048, 3)) for array in images_test[i : i + args.batch_size]]
    #         batch = np.concatenate(reshaped_arrs, axis=0)
    #         batch = np.transpose(batch, (0,3,1,2))
    #         batch = torch.from_numpy(batch).to(device)

    #         y1, y2, z1, z2 = model(batch[0:1],batch[1:2]) #output size is (2, 3, 1024,1024)
    #         y1 = F.interpolate(y1, size=(2048, 2048), mode='bilinear', align_corners=False)
    #         y2 = F.interpolate(y2, size=(2048, 2048), mode='bilinear', align_corners=False)

    #         images_original = np.transpose(batch.cpu().detach().numpy(), (0, 2, 3, 1))
    #         true_img = images_original[0]
    #         true_img = (true_img * 255).astype(np.uint8)
    #         im = Image.fromarray(true_img).convert('RGB')
    #         im.save(f'sample_imgs_val/batch_{i}.jpeg')

    #         images_predicted = np.transpose(y1.cpu().detach().numpy(), (0, 2, 3, 1))
    #         pred_img = images_predicted[0]
    #         pred_img = (pred_img * 255).astype(np.uint8)
    #         im = Image.fromarray(pred_img).convert('RGB')
    #         im.save(f'sample_imgs_val/output_{i}.jpeg')

    #         val_embeddings.append({"organelle":organelles_test[i], "organ":organs_test[i], "embedding": z1.cpu().detach().numpy(), "gene": gene_names_test[i]})
    #         val_embeddings.append({"organelle":organelles_test[i+1], "organ":organs_test[i+1], "embedding": z2.cpu().detach().numpy(), "gene": gene_names_test[i+1]})
    #     except:
    #         pass

    # with open('figure_data/val_embeddings.pickle', 'wb') as handle:
    #     pickle.dump(val_embeddings, handle)


    image = Image.open("case_study.jpg")
    normalized_pixels = dataloader.normalize(image)[np.newaxis,:,:,:]
    batch = np.transpose(normalized_pixels, (0,3,1,2))
    batch = torch.from_numpy(batch).to(device)
    y1, y2, z1, z2 = model(batch,batch)

    with open('figure_data/case_study_emb.pickle', 'wb') as handle:
        pickle.dump(z1.cpu().detach().numpy(), handle)