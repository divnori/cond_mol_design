# ImageToMolecule: Learning Protein Localization Images for Biologically-Specific Molecular Design

6.8301 Advances in Computer Vision: Final Project Spring 2023

In small molecule drug discovery, effective therapeutics must not only be chemically active but also tuned to biological environment. However,  current molecular generative models focus primarily on the former constraint. Here, I propose using protein localization images to learn a rich representation of biological context. The representation is learned by a convolutional variational autoencoder trained with contrastive loss. I then conditioned a graph-based molecular generative model on a selected protein's learned embedding to encourage biological specificity amongst the generated chemical space. The molecules generated with the conditioned model have lower hydrophobicity than the control molecules which is important to the selected biological context. These results indicate that protein localization image embeddings are biologically expressive enough to tune molecular design, bringing us one step closer to designing viable therapeutics using deep learning.

