
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
MODEL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def download_model_if_needed(model_path):
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
    if not download_model_if_needed(MODEL_CHECKPOINT):
        return None
    print(f"Chargement du modèle SAM sur le périphérique : {DEVICE}...")
    try:
        sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CHECKPOINT)
        sam.to(device=DEVICE)
        mask_generator = SamAutomaticMaskGenerator(sam)
        print("Modèle SAM chargé avec succès.")
        return mask_generator
    except Exception as e:
        print(f"Erreur lors du chargement du modèle SAM : {e}")
        return None

def segment_image(image_pil, mask_generator):
    if mask_generator is None:
        print("Le générateur de masques n'est pas disponible.")
        return None, None
    print(f"Traitement de l'image...")
    try:
        image_np = np.array(image_pil.convert("RGB"))
        masks = mask_generator.generate(image_np)
        print(f"Segmentation terminée. {len(masks)} masques détectés.")
        return image_np, masks
    except Exception as e:
        print(f"Une erreur est survenue lors de la segmentation : {e}")
        return None, None

if __name__ == '__main__':
    test_image_file = 'test_room.jpg'
    if not os.path.exists(test_image_file):
        print(f"Le fichier de test '{test_image_file}' n'existe pas. Veuillez en fournir un.")
    else:
        generator = load_sam_model()
        if generator:
            image_to_test = Image.open(test_image_file)
            image_array, generated_masks = segment_image(image_to_test, generator)
            if generated_masks:
                print("\n--- Résultat de la segmentation ---")
                print(f"Nombre de masques trouvés : {len(generated_masks)}")
                print(f"Aire du premier masque : {generated_masks[0]['area']} pixels")