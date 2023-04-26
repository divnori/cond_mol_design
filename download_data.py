"""
Save images in folder dataset/images and generate ground_truth.csv
"""
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

def stem_hpa_image_url(image_url):
    # Take the stem of the image_url, meaning
    # remove the default 'blue_red_green.jpg' coloring
    stem_idx = image_url.find("blue_red_green.jpg")
    if stem_idx < 0:
        print("error in ", image_url)
    return image_url[:stem_idx]

def scrape_image_urls(hpa_gene_id, organelle, gene_name, version="v22", color="blue_red_green"):
    """
    Function written previously by Yitong Tseo (Uhler Group)
    """
    # Example xml and image url:
    # https://www.proteinatlas.org/ENSG00000147421.xml
    # https://images.proteinatlas.org/58586/1060_C8_6_blue_red_green.jpg
    url = f"https://{version}.proteinatlas.org/{hpa_gene_id}.xml"
    xml_data = requests.get(url).content
    soup = BeautifulSoup(xml_data, "xml")
    scraped_subcell_images = []
    for imageUrl in soup.findChildren("imageUrl", recursive=True):
        if (
            imageUrl.parent.name == "image"
            and imageUrl.parent.parent.name == "assayImage"
            and imageUrl.parent.parent.parent.name == "data"
            and imageUrl.parent.parent.parent.parent.name == "subAssay"
        ):
            image_url = f"{stem_hpa_image_url(imageUrl.text)}{color}.jpg"
            cell_line = imageUrl.parent.parent.parent.cellLine.text
            antibody_id = imageUrl.parent.parent.parent.parent.parent.parent["id"]
            antigen_seq = (
                imageUrl.parent.parent.parent.parent.parent.parent.antigenSequence.text
            )
            if version in ("v19", "v20", "v21", "v22"):
                organ = imageUrl.parent.parent.parent.cellLine["organ"]
                cellosaurus_id = imageUrl.parent.parent.parent.cellLine["cellosaurusID"]
            else:
                organ, cellosaurus_id = None, None
            scraped_subcell_images.append(
                {
                    "image_id": image_url.rsplit("/", 1)[-1],
                    "image_url": image_url,
                    "gene_id": hpa_gene_id,
                    "gene_name": gene_name,
                    "cell_line": cell_line.replace("-", "").replace(" ", ""),
                    "organ": organ,
                    "organelle": organelle, #where it localized
                    "cellosaurusID": cellosaurus_id,
                    "antibody_hpa_id": antibody_id,
                    "antigen_sequence": antigen_seq,
                    "version": version,
                }
            )
    return scraped_subcell_images


if __name__ == "__main__":

    subcell_df = pd.read_csv("dataset/subcellular_location.csv")
    subcell_df = subcell_df[subcell_df["Approved"].notnull()]

    # process downloaded csv into clean ground truth csv

    # scraped_images = []
    # i = 0
    # for row in tqdm(subcell_df.iterrows()):
    #     hpa_gene_id = row[1]["Gene"]
    #     gene_name = row[1]["Gene name"]
    #     organelle = row[1]["Approved"]
    #     scraped_images.extend(scrape_image_urls(hpa_gene_id, organelle, gene_name))
    #     print(f"Processed {hpa_gene_id} in {organelle}.")
    #     i+=1
    #     if i == 1000:
    #         break
    
    # image_df = pd.DataFrame.from_records(scraped_images)
    # image_df.to_csv("dataset/ground_truth.csv",index=False)

    # use clean ground truth csv to download images

    image_df = pd.read_csv("dataset/ground_truth.csv")

    for row in tqdm(image_df.iterrows()):
        image_url = row[1]["image_url"]
        image_name = row[1]["image_id"]
        gene_name = row[1]["gene_name"]
        img_data = requests.get(image_url).content
        with open(f"dataset/images/{gene_name}_{image_name}", 'wb') as handler:
            handler.write(img_data)