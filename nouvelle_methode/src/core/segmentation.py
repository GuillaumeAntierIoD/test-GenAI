import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import requests 
import os
import numpy as np

DEVICE = "cpu"
MODELS_DIR = "models"
MODEL_NAME = "sam_vit_b_01ec64.pth"
MODEL_CHECKPOINT = os.path.join(MODELS_DIR, MODEL_NAME)
MODEL_TYPE = "vit_b"
MODEL_URL = f"https://dl.fbaipublicfiles.com/segment_anything/{MODEL_NAME}"

def download_model_if_needed(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if not os.path.exists(model_path):
        print(f"Le modèle SAM n'a pas été trouvé. Téléchargement depuis {MODEL_URL}...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Modèle sauvegardé dans : {model_path}")
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors du téléchargement du modèle : {e}")
            return False
    return True

def load_sam_model():
    """Charge le modèle SAM avec les paramètres optimisés pour la vitesse."""
    if not download_model_if_needed(MODEL_CHECKPOINT):
        return None
    print(f"Chargement du modèle SAM depuis '{MODEL_CHECKPOINT}' sur le périphérique : {DEVICE}...")
    try:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CHECKPOINT)
        sam.to(device=DEVICE)
        
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=8, 
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        print("Modèle SAM chargé avec succès.")
        return mask_generator
    except Exception as e:
        print(f"Erreur lors du chargement du modèle SAM : {e}")
        return None

def segment_image(image_pil, mask_generator):
    """Prend une image PIL et le générateur de masques, et retourne la liste des masques."""
    if mask_generator is None:
        return None, None
    print("Traitement de l'image...")
    try:
        image_np = np.array(image_pil.convert("RGB"))
        masks = mask_generator.generate(image_np)
        print(f"Segmentation terminée. {len(masks)} masques détectés.")
        return image_np, masks
    except Exception as e:
        print(f"Une erreur est survenue lors de la segmentation : {e}")
        return None, None