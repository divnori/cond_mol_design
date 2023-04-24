import esm
from numpy import asarray
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch

def normalize(image):
    pixels = asarray(image)
    pixels = pixels.astype('float32')
    pixels /= 255.0
    return pixels

def get_esm2_emb(data,batch_converter):
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]


def load_data(image_dir):
    ground_truth = pd.read_csv("dataset/ground_truth_mini.csv")
    data = {}

    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    for image_path in os.listdir(image_dir):
        image = Image.open(f"{image_dir}/{image_path}")
        gene_name = image_path.split("_")[0]
        row = ground_truth[ground_truth["gene_name"] == gene_name].iloc[0]
        organelle = row["organelle"]
        protein_seq = row["antigen_sequence"]

        esm_embedding = get_esm2_emb([("protein1",protein_seq)],batch_converter,model)


        pixels = asarray(image)
        pixels = pixels.astype('float32')
        
        if gene_name not in data:
            data[gene_name] = {"organelle":organelle, 
                                "protein_seq":protein_seq, 
                                "image_arr":[pixels]}
        else:
            data[gene_name]["image_arr"].append(pixels)
    
    return data



