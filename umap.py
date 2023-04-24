"""
UMAP experiments
"""
import dataloader
import matplotlib.pyplot as plt
import os

# umap of raw images colored by gene ID 
# do images of the same protein exhibit similar features in the linear embedding space?

if __name__ == "__main__":
    dataloader.load_data("dataset/images")