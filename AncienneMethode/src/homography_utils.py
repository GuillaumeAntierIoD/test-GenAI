import cv2
import numpy as np
from typing import Tuple, List, Optional

def detect_keypoints_and_descriptors(image: np.ndarray, method='SIFT'):
    """
    Détecte les points clés et descripteurs dans une image
    
    Args:
        image: Image en niveaux de gris ou couleur
        method: Méthode de détection ('SIFT', 'ORB', 'SURF')
    
    Returns:
        tuple: (keypoints, descriptors)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    if method == 'SIFT':
        detector = cv2.SIFT_create()
    elif method == 'ORB':
        detector = cv2.ORB_create()
    else:
        detector = cv2.SIFT_create()  # Fallback
    
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors

def find_homography_from_plane(image: np.ndarray, depth_map: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Trouve l'homographie d'un plan à partir de la carte de profondeur et d'un masque
    
    Args:
        image: Image RGB
        depth_map: Carte de profondeur
        mask: Masque binaire du plan/surface
    
    Returns:
        Matrice d'homographie 3x3 ou None si échec
    """
    # Trouver les contours du masque
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # Prendre le plus grand contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximer le contour pour obtenir les coins
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Si on a approximativement 4 coins (quadrilatère)
    if len(approx) >= 4:
        # Prendre les 4 premiers coins
        corners = approx[:4].reshape(-1, 2).astype(np.float32)
        
        # Trier les coins (top-left, top-right, bottom-right, bottom-left)
        corners = sort_corners(corners)
        
        # Définir un rectangle de référence
        rect_width = 300
        rect_height = 200
        reference_corners = np.array([
            [0, 0],
            [rect_width, 0],
            [rect_width, rect_height],
            [0, rect_height]
        ], dtype=np.float32)
        
        # Calculer l'homographie
        homography, _ = cv2.findHomography(corners, reference_corners, cv2.RANSAC)
        return homography
    
    return None

def sort_corners(corners: np.ndarray) -> np.ndarray:
    """
    Trie les coins dans l'ordre : top-left, top-right, bottom-right, bottom-left
    """
    # Calculer les sommes et différences
    sums = corners.sum(axis=1)
    diffs = np.diff(corners, axis=1)
    
    # Top-left: plus petite somme
    tl = corners[np.argmin(sums)]
    # Bottom-right: plus grande somme
    br = corners[np.argmax(sums)]
    # Top-right: plus petite différence
    tr = corners[np.argmin(diffs)]
    # Bottom-left: plus grande différence
    bl = corners[np.argmax(diffs)]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

def estimate_plane_orientation(depth_map: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """
    Estime l'orientation d'un plan à partir de la carte de profondeur
    
    Args:
        depth_map: Carte de profondeur
        mask: Masque du plan
    
    Returns:
        tuple: (angle_x, angle_y) angles d'inclinaison en degrés
    """
    # Extraire les points du plan
    y_coords, x_coords = np.where(mask > 0)
    depth_values = depth_map[y_coords, x_coords]
    
    if len(depth_values) < 10:
        return 0.0, 0.0
    
    # Créer une matrice de points 3D
    points_3d = np.column_stack([x_coords, y_coords, depth_values])
    
    # Calculer le plan de régression
    centroid = np.mean(points_3d, axis=0)
    centered_points = points_3d - centroid
    
    # SVD pour trouver la normale du plan
    _, _, V = np.linalg.svd(centered_points)
    normal = V[-1, :]  # Dernier vecteur propre = normale
    
    # Calculer les angles
    angle_x = np.arctan2(normal[2], normal[1]) * 180 / np.pi
    angle_y = np.arctan2(normal[2], normal[0]) * 180 / np.pi
    
    return angle_x, angle_y

def apply_perspective_transform(obj_img: np.ndarray, homography: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Applique une transformation perspective à un objet
    
    Args:
        obj_img: Image de l'objet
        homography: Matrice d'homographie
        target_size: Taille cible (width, height)
    
    Returns:
        Image transformée
    """
    if homography is None:
        return obj_img
    
    # Appliquer la transformation perspective
    transformed = cv2.warpPerspective(obj_img, homography, target_size)
    return transformed

def calculate_optimal_scale_from_depth(depth_map: np.ndarray, position: Tuple[int, int], 
                                     reference_depth: float = None) -> float:
    """
    Calcule l'échelle optimale basée sur la profondeur
    
    Args:
        depth_map: Carte de profondeur
        position: Position (x, y) de placement
        reference_depth: Profondeur de référence (optionnel)
    
    Returns:
        Facteur d'échelle
    """
    x, y = position
    h, w = depth_map.shape
    
    # Vérifier les limites
    x = max(0, min(x, w-1))
    y = max(0, min(y, h-1))
    
    depth_at_position = depth_map[y, x]
    
    if reference_depth is None:
        # Utiliser la profondeur moyenne comme référence
        reference_depth = np.mean(depth_map[depth_map > 0])
    
    # Calculer l'échelle inversement proportionnelle à la profondeur
    # Plus l'objet est loin (profondeur élevée), plus il doit être petit
    if reference_depth > 0 and depth_at_position > 0:
        scale = reference_depth / depth_at_position
        # Limiter entre 0.1 et 2.0 pour éviter les extrêmes
        scale = max(0.1, min(2.0, scale))
    else:
        scale = 1.0
    
    return scale

def detect_placement_surface(depth_map: np.ndarray, masks: List[dict], 
                           min_area: int = 1000) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Détecte automatiquement une surface appropriée pour le placement
    
    Args:
        depth_map: Carte de profondeur
        masks: Liste des masques SAM
        min_area: Aire minimale requise
    
    Returns:
        tuple: (surface_mask, surface_info) ou None
    """
    best_surface = None
    best_score = 0
    
    for mask_dict in masks:
        mask = mask_dict['segmentation']
        area = mask_dict['area']
        
        if area < min_area:
            continue
        
        # Calculer la planéité de la surface
        y_coords, x_coords = np.where(mask)
        if len(y_coords) < 50:
            continue
        
        depths = depth_map[y_coords, x_coords]
        depth_std = np.std(depths)
        
        # Score basé sur l'aire et la planéité (plus la std est faible, mieux c'est)
        planarity_score = 1.0 / (1.0 + depth_std)
        area_score = min(area / 10000, 1.0)  # Normaliser l'aire
        total_score = planarity_score * area_score
        
        if total_score > best_score:
            best_score = total_score
            best_surface = (mask, mask_dict)
    
    return best_surface

def create_shadow_mask(obj_mask: np.ndarray, light_direction: Tuple[float, float] = (1, 1), 
                      shadow_intensity: float = 0.3) -> np.ndarray:
    """
    Crée un masque d'ombre basé sur la position de l'objet
    
    Args:
        obj_mask: Masque de l'objet
        light_direction: Direction de la lumière (x, y)
        shadow_intensity: Intensité de l'ombre (0-1)
    
    Returns:
        Masque d'ombre
    """
    # Créer une transformation pour l'ombre
    light_x, light_y = light_direction
    
    # Matrice de transformation pour l'ombre
    shadow_matrix = np.array([
        [1, light_x * 0.3, light_x * 5],
        [light_y * 0.1, 1, light_y * 5],
        [0, 0, 1]
    ], dtype=np.float32)
    
    h, w = obj_mask.shape
    shadow = cv2.warpPerspective(obj_mask, shadow_matrix, (w, h))
    
    # Appliquer un flou pour rendre l'ombre plus réaliste
    shadow = cv2.GaussianBlur(shadow, (15, 15), 0)
    
    # Réduire l'intensité
    shadow = (shadow * shadow_intensity).astype(np.uint8)
    
    return shadow

 