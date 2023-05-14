import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from PIL import Image
import pandas as pd
import random
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

import math
import rdkit
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.rdmolfiles import SmilesMolSupplier

import dataloader

def loss_curve(loss_pickle):

    with open(loss_pickle, 'rb') as pickle_result:
        loss_list = pickle.load(pickle_result)

    loss_list = loss_list[:50]

    loss_list = [l.item() for l in loss_list]

    sns.set_style("darkgrid")
    plt.plot(np.array(loss_list))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("figures/train_loss_curve.png")


def pca_embeddings(df, color_criteria):

    pca = PCA(n_components=2)
    labels = df[color_criteria].tolist()
    labels = [l.lower() for l in labels]
    df = df.drop(columns=[color_criteria])

    df = (df - df.mean()) / df.std() # normalize
    pca_result = pca.fit_transform(df.values)
    pca_one = pca_result[:,0]
    pca_two = pca_result[:,1]
    
    plt.figure(figsize=(7,7))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Autoencoder Embeddings on Validation Set")
    sns_plot = sns.scatterplot(
        x=pca_one, y=pca_two,
        hue=labels,
        palette=sns.color_palette("tab10"),
        data=df,
        legend="full",
        s = 50
    )
    sns_plot.figure.savefig(f"figures/val_pca_{color_criteria}.png")

def umap_embeddings(df, color_criteria):
    reducer = umap.UMAP()
    labels = df[color_criteria].tolist()
    unique_labels = df[color_criteria].unique().tolist()
    unique_labels = [l.upper() for l in unique_labels]
    df = df.drop(columns=[color_criteria])
    scaled_data = StandardScaler().fit_transform(df)
    umap_embedding = reducer.fit_transform(scaled_data)

    # List of colors in the color palettes
    rgb_values = sns.color_palette("Set2", len(unique_labels))

    # Map continents to the colors
    color_map = dict(zip(unique_labels, rgb_values))
    print(color_map)

    plt.scatter(
        umap_embedding[:, 0],
        umap_embedding[:, 1],
        c = [color_map[l.upper()] for l in labels]
    )
    plt.savefig(f"val_umap_{color_criteria}.png")


def build_embedding_df(emb_pickle, color_criteria):
    with open(emb_pickle, 'rb') as pickle_result:
        embedding_list = pickle.load(pickle_result)

    criteria = []
    images = []

    columns = [f"feat{i}" for i in range(2000)]
    columns.extend([color_criteria])
    df = pd.DataFrame(columns=columns)

    for i in range(len(embedding_list)):
        d = embedding_list[i]
        label = d[color_criteria]
        embedding = d["embedding"][0].T
        row = embedding.tolist()
        row.extend([label])
        df.loc[i] = row
        if i % 100 == 0:
            print(i)

    df.to_pickle(f"figure_data/val_emb_{color_criteria}_df.pkl")
    return df

def draw_molecules(smi_file):
    # load molecules from file
    mols = SmilesMolSupplier(smi_file, sanitize=True, nameColumn=-1)

    n_samples = 20
    mols_list = [mol for mol in mols]
    mols_sampled = random.sample(mols_list, n_samples)  # sample 100 random molecules to visualize

    mols_per_row = int(math.sqrt(n_samples))            # make a square grid

    labels=list(range(n_samples))       # label structures with a number

    # draw the molecules (creates a PIL image)
    img = MolsToGridImage(mols=mols_sampled,
                        molsPerRow=mols_per_row,
                        legends=[str(i) for i in labels])

    img.save("figures/molecules_control.png")

if __name__ == "__main__":
    # loss_curve("figure_data/loss_curve_data.pickle")

    # df = build_embedding_df("figure_data/val_embeddings.pickle","organelle")
    
    # with open("figure_data/val_emb_organelle_df.pkl", 'rb') as pickle_result:
    #     df = pickle.load(pickle_result)

    # pca_embeddings(df, "organelle")
    pass