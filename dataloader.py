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

def get_esm2_emb(data, batch_converter, model):
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        model.to(torch.float16).cuda()
        results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    seq_rep = token_representations[0, 1 : token_representations.shape[1] - 1].mean(0)
    return seq_rep

def load_data(image_dir):
    ground_truth = pd.read_csv("dataset/ground_truth_mini.csv")
    data = {}

    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    for image_path in os.listdir(image_dir):
        image = Image.open(f"{image_dir}/{image_path}")
        normalized_pixels = normalize(image)
        gene_name = image_path.split("_")[0]

        # most images are 2048 x 2048 x 3 (rest are too low res)
        if normalized_pixels.shape[0] != 2048:
            continue

        if gene_name not in data:
            row = ground_truth[ground_truth["gene_name"] == gene_name].iloc[0]
            organelle = row["organelle"]
            protein_seq = row["antigen_sequence"]

            if type(protein_seq) != str:
                continue

            # calculate a size 1280 embedding for each protein sequence using ESM2 pretrained model
            esm_embedding = get_esm2_emb([("protein1",protein_seq)],batch_converter,model)

            data[gene_name] = {"organelle":organelle, 
                        "protein_seq":protein_seq,
                        "esm_embedding":esm_embedding, 
                        "image_arr":[normalized_pixels]}

        else:
            data[gene_name]["image_arr"].append(normalized_pixels)
    
    return data



