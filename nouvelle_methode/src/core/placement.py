import numpy as np
from PIL import Image

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

def calculate_scale_from_depth(depth_map, placement_point, base_scale=0.5):
    if depth_map is None or placement_point is None: return base_scale
    x, y = placement_point
    if y < depth_map.shape[0] and x < depth_map.shape[1]:
        depth_value = depth_map[y, x]
        depth_min, depth_max = depth_map.min(), depth_map.max()
        normalized_depth_scale = 1 - ((depth_value - depth_min) / (depth_max - depth_min + 1e-6))
        final_scale = base_scale * (0.2 + 0.8 * normalized_depth_scale)
        return final_scale
    return base_scale

def insert_object(env_pil, obj_pil, position, scale, return_mask=False):
    obj_with_alpha = obj_pil.convert("RGBA")
    original_width, original_height = obj_with_alpha.size
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_obj = obj_with_alpha.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    env_copy = env_pil.copy().convert("RGBA")
    
    paste_x = position[0] - new_width // 2
    paste_y = position[1] - new_height 
    
    env_copy.paste(resized_obj, (paste_x, paste_y), resized_obj)
    
    if return_mask:
        object_mask = np.zeros((env_copy.height, env_copy.width), dtype=np.uint8)
        alpha_mask = np.array(resized_obj)[:, :, 3] > 0
        object_mask[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = alpha_mask * 255
        return env_copy.convert("RGB"), object_mask

    return env_copy.convert("RGB")