"""
Save images in folder dataset/images
Generate a csv with path to image | protein sequence | localization label 
"""

def scrape_image_urls(hpa_gene_id="ENSG00000147421", version="v22", color="blue_red_green"):
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
                    "cell_line": cell_line.replace("-", "").replace(" ", ""),
                    "organ": organ,
                    "cellosaurusID": cellosaurus_id,
                    "antibody_hpa_id": antibody_id,
                    "antigen_sequence": antigen_seq,
                    "version": version,
                }
            )
    return scraped_subcell_images


if __name__ == "__main__":

    