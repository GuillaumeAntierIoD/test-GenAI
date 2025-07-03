# Fichier : src/core/depth_estimation.py
import os
import torch
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_midas_model():
    """
    Charge le modèle MiDaS depuis PyTorch Hub.
    Il est spécialisé dans l'estimation de la profondeur monoculaire.
    """
    print(f"Chargement du modèle MiDaS sur le périphérique : {DEVICE}...")
    try:
        # On choisit un modèle MiDaS performant
        model_type = "DPT_Large"
        midas_model = torch.hub.load("intel-isl/MiDaS", model_type)
        midas_model.to(DEVICE)
        midas_model.eval()

        # MiDaS requiert des transformations spécifiques sur l'image
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        midas_transform = transforms.dpt_transform if model_type == "DPT_Large" else transforms.small_transform
        
        print("Modèle MiDaS chargé avec succès.")
        return midas_model, midas_transform

    except Exception as e:
        print(f"Erreur lors du chargement du modèle MiDaS : {e}")
        return None, None

def estimate_depth(image_pil, midas_model, midas_transform):
    """
    Prend une image PIL et les composants MiDaS,
    et retourne la carte de profondeur sous forme d'array NumPy.
    """
    if midas_model is None or midas_transform is None:
        return None

    try:
        # On convertit l'image et on applique les transformations
        image_np = np.array(image_pil.convert("RGB"))
        input_tensor = midas_transform(image_np).to(DEVICE)

        with torch.no_grad():
            prediction = midas_model(input_tensor)
            
            # Redimensionne la prédiction à la taille de l'image originale
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        print("Carte de profondeur générée avec succès.")
        return depth_map

    except Exception as e:
        print(f"Une erreur est survenue lors de l'estimation de la profondeur : {e}")
        return None

if __name__ == '__main__':
    test_image_path = '../../images/test_room.jpg' 

    if not os.path.exists(test_image_path):
        print(f"Fichier de test non trouvé : {test_image_path}")
    else:
        model, transform = load_midas_model()
        if model and transform:
            image = Image.open(test_image_path)
            depth_map_result = estimate_depth(image, model, transform)
            
            if depth_map_result is not None:
                print("\n--- Résultat de l'estimation de profondeur ---")
                print(f"Dimensions de la carte de profondeur : {depth_map_result.shape}")
                print(f"Valeur min/max : {depth_map_result.min():.2f} / {depth_map_result.max():.2f}")