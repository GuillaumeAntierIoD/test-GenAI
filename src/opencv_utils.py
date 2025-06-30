import cv2
import numpy as np

def resize_object(obj_img: np.ndarray, scale: float):
    """
    Redimensionne l'objet selon un facteur d'échelle
    
    Args:
        obj_img: Image de l'objet (numpy array)
        scale: Facteur d'échelle (ex: 0.5 pour réduire de moitié)
    
    Returns:
        Image redimensionnée
    """
    if scale <= 0:
        raise ValueError("Le facteur d'échelle doit être positif")
    
    # Calculer nouvelles dimensions
    height, width = obj_img.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Redimensionner avec interpolation bicubique pour une meilleure qualité
    resized = cv2.resize(obj_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return resized


def insert_object(base_img: np.ndarray, obj_img: np.ndarray, mask: np.ndarray, position: tuple):
    """
    Insère l'objet dans l'image de base à la position donnée avec alpha blending
    
    Args:
        base_img: Image de fond (numpy array)
        obj_img: Image de l'objet à insérer (numpy array)
        mask: Masque binaire de l'objet (numpy array, valeurs 0-255)
        position: Tuple (x, y) pour le coin supérieur gauche de placement
    
    Returns:
        Image finale avec l'objet inséré
    """
    # Copier l'image de base pour ne pas la modifier
    result = base_img.copy()
    
    # Vérifier les dimensions
    obj_h, obj_w = obj_img.shape[:2]
    base_h, base_w = base_img.shape[:2]
    x, y = position
    
    # Vérifier que l'objet rentre dans l'image
    if x < 0 or y < 0 or x + obj_w > base_w or y + obj_h > base_h:
        # Ajuster les dimensions pour éviter de dépasser
        x = max(0, min(x, base_w - obj_w))
        y = max(0, min(y, base_h - obj_h))
        
        # Si l'objet est encore trop grand, le redimensionner
        if x + obj_w > base_w or y + obj_h > base_h:
            max_w = base_w - x
            max_h = base_h - y
            scale_w = max_w / obj_w
            scale_h = max_h / obj_h
            scale = min(scale_w, scale_h)
            
            obj_img = resize_object(obj_img, scale)
            mask = cv2.resize(mask, (obj_img.shape[1], obj_img.shape[0]))
            obj_h, obj_w = obj_img.shape[:2]
    
    # Normaliser le masque (0.0 à 1.0)
    if mask.dtype == np.uint8:
        mask_norm = mask.astype(np.float32) / 255.0
    else:
        mask_norm = mask.astype(np.float32)
    
    # S'assurer que le masque a la même dimension que l'objet
    if len(mask_norm.shape) == 2:
        mask_norm = np.stack([mask_norm] * 3, axis=2)
    
    # Définir la région d'intérêt dans l'image de base
    roi = result[y:y+obj_h, x:x+obj_w]
    
    # Alpha blending : résultat = objet * masque + fond * (1 - masque)
    blended = obj_img * mask_norm + roi * (1 - mask_norm)
    
    # Placer le résultat dans l'image finale
    result[y:y+obj_h, x:x+obj_w] = blended.astype(np.uint8)
    
    return result