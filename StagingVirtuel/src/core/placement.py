
from PIL import Image
import numpy as np

def placer_objet_drag_and_drop(bg_image, obj_image, placement_data, return_mask=False):
    """
    Insère un objet dans une image de fond en utilisant les données de placement
    (position, taille, angle) issues du canvas.
    """
    width = int(placement_data['width'] * placement_data.get('scaleX', 1))
    height = int(placement_data['height'] * placement_data.get('scaleY', 1))
    angle = placement_data.get('angle', 0)
    left = placement_data['left']
    top = placement_data['top']

    obj_rgba = obj_image.convert("RGBA")
    resized_obj = obj_rgba.resize((width, height), Image.Resampling.LANCZOS)
    rotated_obj = resized_obj.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    paste_layer = Image.new("RGBA", bg_image.size)
    
    paste_x = left - (rotated_obj.width - width) // 2
    paste_y = top - (rotated_obj.height - height) // 2
    
    paste_layer.paste(rotated_obj, (paste_x, paste_y), rotated_obj)

    bg_rgba = bg_image.copy().convert("RGBA")
    final_image = Image.alpha_composite(bg_rgba, paste_layer)

    if return_mask:
        object_mask = np.array(paste_layer)[:, :, 3]
        return final_image.convert("RGB"), object_mask
    
    return final_image.convert("RGB")