"""
UMAP experiments
"""
#import dataloader
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.manifold import TSNE

# umap of raw images colored by gene ID 
# do images of the same protein exhibit similar features in the linear embedding space?

# with open('data.pickle', 'rb') as handle:
#         data = pickle.load(handle)

# data = dict(list(data.items())[0: 5])

# organelles = [] # ground truth in order
# gene_names = [] # ground truth in order
# images = []

# for gene_name, metadata in data.items():
    
#     for image in data[gene_name]["image_arr"]:
#         gene_names.append(gene_name)
#         organelles.append(data[gene_name]["organelle"])
#         images.append(np.expand_dims(image, axis=0))

# arr = np.mean(np.concatenate(images, axis=0),axis=3)
# new_shape = (arr.shape[0], 32, 32)
# row_ratio = arr.shape[1] // new_shape[1]
# col_ratio = arr.shape[2] // new_shape[2]
# arr_resized = np.resize(arr, (arr.shape[0], new_shape[1], row_ratio, new_shape[2], col_ratio)).mean(axis=(2, 4))
# print(arr_resized.shape)

# arr_flat = arr_resized.reshape((30, -1))
# df = pd.DataFrame(arr_flat)
# df["organelles"] = organelles

# df.to_pickle("linear_pca_data.pkl")

df = pd.read_pickle("linear_pca_data.pkl") 

labels = df['organelles']
df = df.drop('organelles', axis=1)
tsne_data = TSNE(n_components=2, learning_rate='auto',
                 init='random', perplexity=3).fit_transform(df.to_numpy())
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap='viridis')
plt.colorbar()
plt.savefig('tsne_plot.png')