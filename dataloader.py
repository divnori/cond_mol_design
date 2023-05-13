import esm
from numpy import asarray
import numpy as np
import pandas as pd
import os
from PIL import Image
import pickle
import torch

organ_options = set(["Lung", "Skin", "Male", "Female", "Mesenchymal", "Brain", "Liver", "Kidney", "Gastrointestinal", "Myeloid"])
organelle_options = set(["nucleus","mitochondria","membrane","golgi","endoplasmic","cytosol","vesicles","centrosome"])
nucleus_alternatives = set(["nucleoli","nucleoli","nucleoplasm","nuclear"])


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

def process_organ(organ_str):
    words = organ_str.split(" ")
    organ = ""
    for word in words:
        if word in organ_options:
            organ = word
    if len(organ) == 0:
        return None
    else:
        return organ
    

def process_organelle(organelle_str):
    organelle_str = organelle_str.replace(";", " ")
    words = organelle_str.split(" ")
    organelle = ""
    for word in words:
        if word.lower() in organelle_options:
            organelle = word.lower()
        elif word.lower() in nucleus_alternatives:
            organelle = "nucleus"
    if len(organelle) == 0:
        return None
    else:
        return organelle

def load_data(image_dir):
    ground_truth = pd.read_csv("dataset/ground_truth.csv")
    data = {}

    # model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    # batch_converter = alphabet.get_batch_converter()
    # model.eval()  # disables dropout for deterministic results

    i = 0
    for image_path in os.listdir(image_dir):
        if i%100 == 0:
            print(i)
        i+=1
        image = Image.open(f"{image_dir}/{image_path}")
        normalized_pixels = normalize(image)
        gene_name = image_path.split("_")[0]

        # most images are 2048 x 2048 x 3 (rest are too low res)
        if normalized_pixels.shape[0] != 2048:
            continue

        if gene_name not in data:
            row = ground_truth[ground_truth["gene_name"] == gene_name].iloc[0]
            organelle = process_organelle(row["organelle"])
            organ = process_organ(row["organ"])
            protein_seq = row["antigen_sequence"]

            if type(protein_seq) != str:
                continue

            # calculate a size 1280 embedding for each protein sequence using ESM2 pretrained model
            # esm_embedding = get_esm2_emb([("protein1",protein_seq)],batch_converter,model)

            if organelle is not None and organ is not None:
                data[gene_name] = {"organelle":organelle,
                            "organ": organ,
                            "protein_seq":protein_seq,
                            "image_arr":[normalized_pixels]} #"esm_embedding":esm_embedding, 

        else:
            data[gene_name]["image_arr"].append(normalized_pixels)
    
    with open('data.pickle', 'wb') as handle:
        pickle.dump(data, handle)

    return data



