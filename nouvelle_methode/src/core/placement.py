import numpy as np
import cv2
from PIL import Image, ImageDraw

def find_floor_mask(masks, image_height):
    if not masks: return None
    min_area = (image_height * image_height) * 0.01
    large_masks = [m for m in masks if m['area'] > min_area]
    if not large_masks: return None
    floor_mask, max_area = None, 0
    for mask in large_masks:
        y_center = mask['bbox'][1] + mask['bbox'][3] / 2
        if y_center > (image_height / 2) and mask['area'] > max_area:
            max_area = mask['area']
            floor_mask = mask['segmentation']
    return floor_mask

def get_placement_point(floor_mask):
    if floor_mask is None: return None
    rows, cols = np.where(floor_mask)
    if len(rows) == 0: return None
    center_x = int(np.mean(cols))
    placement_y = int(np.min(rows) + (np.max(rows) - np.min(rows)) * 0.8)
    return (center_x, placement_y)

def get_placement_point(floor_mask):
    if floor_mask is None: return None
    rows, cols = np.where(floor_mask)
    if len(rows) == 0: return None
    center_x = int(np.mean(cols))
    placement_y = int(np.min(rows) + (np.max(rows) - np.min(rows)) * 0.8)
    return (center_x, placement_y)

def calculate_scale_from_depth(depth_map, placement_point, base_scale=1.0):
    """
    Calcule l'échelle uniquement à partir de la profondeur au point de placement.
    """
    if depth_map is None or placement_point is None:
        return base_scale

    x, y = placement_point
    
    if y < depth_map.shape[0] and x < depth_map.shape[1]:
        depth_value = depth_map[y, x]
        depth_min, depth_max = depth_map.min(), depth_map.max()

        if (depth_max - depth_min) == 0:
            return base_scale
        
        normalized_depth = 1 - ((depth_value - depth_min) / (depth_max - depth_min))
        depth_adjustment = 0.2 + (normalized_depth * 0.8)
        
        final_scale = base_scale * depth_adjustment
        return final_scale
        
    return base_scale

def insert_object(env_pil, obj_pil, position, scale, return_mask=False):
    """
    Insère une image (objet) dans une autre (environnement) de manière robuste,
    en utilisant la composition alpha pour éviter les erreurs de débordement.
    """
    obj_with_alpha = obj_pil.convert("RGBA")
    
    new_width = int(obj_with_alpha.width * scale)
    new_height = int(obj_with_alpha.height * scale)

    if new_width <= 0 or new_height <= 0:
        if return_mask:
            return env_pil.convert("RGB"), np.zeros((env_pil.height, env_pil.width), dtype=np.uint8)
        return env_pil.convert("RGB")
        
    resized_obj = obj_with_alpha.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    env_copy = env_pil.copy().convert("RGBA")
    
    paste_layer = Image.new("RGBA", env_copy.size)
    
    paste_x = position[0] - new_width // 2
    paste_y = position[1] - new_height // 2
    
    paste_layer.paste(resized_obj, (paste_x, paste_y))
    
    final_image = Image.alpha_composite(env_copy, paste_layer)
    
    if return_mask:
        object_mask = np.array(paste_layer)[:, :, 3]
        return final_image.convert("RGB"), object_mask
    
    return final_image.convert("RGB")

def apply_perspective_transform_to_fill(obj_pil, quad_points, target_size):
    """
    Déforme une image pour qu'elle remplisse un quadrilatère (quad_points)
    dans une image vide de la taille de l'arrière-plan ('target_size').
    """
    obj_rgba = obj_pil.convert("RGBA")
    obj_np = np.array(obj_rgba)
    h, w = obj_np.shape[:2]

    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(quad_points)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    warped_obj = cv2.warpPerspective(
        obj_np,
        matrix,
        (target_size[0], target_size[1])
    )

    return Image.fromarray(warped_obj)


def insert_object_with_perspective(bg_image, obj_image, quad_points, return_mask=False):
    """
    Insère un objet avec une transformation de perspective dans une image de fond.
    """
    transformed_layer = apply_perspective_transform_to_fill(
        obj_image, 
        quad_points, 
        bg_image.size 
    )

    bg_rgba = bg_image.copy().convert("RGBA")
    final_image = Image.alpha_composite(bg_rgba, transformed_layer)

    if return_mask:
        mask = transformed_layer.split()[-1]
        return final_image.convert('RGB'), mask
    
    return final_image.convert('RGB')

def debug_draw_perspective_bounds(bg_image, quad_points):
    """
    Fonction de débogage pour dessiner le quadrilatère de l'utilisateur
    et le rectangle calculé par la fonction de perspective.
    """
    debug_image = bg_image.copy()
    draw = ImageDraw.Draw(debug_image)

    # 1. Dessine en VERT le quadrilatère que vous avez cliqué.
    draw.polygon(quad_points, outline="lime", width=5)

    # 2. On recalcule la boîte de destination comme le fait votre fonction apply_perspective_to_object
    # Pour cela, on a besoin d'une taille source (peu importe laquelle, c'est pour la matrice)
    w, h = 100, 100 
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = np.float32(quad_points)
    
    try:
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, matrix)
        
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        
        min_x = int(np.min(x_coords))
        max_x = int(np.max(x_coords))
        min_y = int(np.min(y_coords))
        max_y = int(np.max(y_coords))
        
        # 3. Dessine en ROUGE le rectangle que le code CALCULE réellement.
        draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=5)
        
    except Exception as e:
        print(f"Erreur durant le calcul de la perspective pour le débogage : {e}")

    return debug_image