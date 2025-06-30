import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import streamlit as st

def load_sam_model(checkpoint_path: str, device="cpu"):
    """
    Charge le modèle SAM ViT-B
    
    Args:
        checkpoint_path (str): Chemin vers le fichier de checkpoint SAM
        device (str): Device à utiliser ('cpu' ou 'cuda')
    
    Returns:
        sam_model: Modèle SAM chargé et prêt à l'utilisation
    """
    sam = sam_model_registry['vit_b'](checkpoint=checkpoint_path)
    sam.to(device)
    return sam

def generate_sam_mask(image: np.ndarray, sam_model, points_per_side=16):
    """
    Génère des masques SAM automatiquement sur une image
    
    Args:
        image (np.ndarray): Image en format RGB (H, W, 3)
        sam_model: Modèle SAM chargé
        points_per_side (int): Nombre de points par côté pour la génération automatique
    
    Returns:
        list: Liste des masques générés par SAM
    """
    mask_generator = SamAutomaticMaskGenerator(
        sam_model,
        points_per_side=points_per_side
    )
    
    masks = mask_generator.generate(image)
    return masks

def show_anns(anns):
    """
    Prépare la visualisation des annotations (masques)
    
    Args:
        anns: Liste des annotations/masques
    
    Returns:
        numpy.ndarray: Masque coloré pour overlay
    """
    if len(anns) == 0:
        return None
    
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], 
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    return img

def process_image_with_sam(image_pil, sam_model, points_per_side=16):
    """
    Traite une image PIL avec SAM et retourne les résultats
    
    Args:
        image_pil: Image PIL
        sam_model: Modèle SAM chargé
        points_per_side: Nombre de points par côté
    
    Returns:
        tuple: (image_numpy, masks, mask_overlay)
    """
    # Conversion PIL vers numpy RGB
    image_np = np.array(image_pil)
    if image_np.shape[-1] == 4:  # RGBA vers RGB
        image_np = image_np[:, :, :3]
    
    # Génération des masques
    masks = generate_sam_mask(image_np, sam_model, points_per_side)
    
    # Création de l'overlay des masques
    mask_overlay = show_anns(masks)
    
    return image_np, masks, mask_overlay