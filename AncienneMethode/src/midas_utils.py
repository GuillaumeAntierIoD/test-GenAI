import os
import sys
sys.path.append(os.path.abspath("src/MiDaS"))
import torch
import numpy as np
import cv2
from torchvision import transforms
from torchvision.transforms import Compose
from PIL import Image
from timm.models import create_model
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

def load_midas_model(device="cpu"):
    model_path = "models/dpt_large_384.pt"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle MiDaS non trouvé à : {model_path}")
    
    # Chargement du modèle
    model = DPTDepthModel(
        path=model_path,
        backbone="vitl16_384",
        non_negative=True
    )
    model.eval()
    model.to(device)

    # Pipeline de transformation
    transform = Compose([
        Resize(256, 256, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])

    return model, transform




def estimate_depth(image_pil: Image.Image, model, transform, device="cpu"):
    image_rgb = np.array(image_pil.convert("RGB"))
    input_tensor = transform(Image.fromarray(image_rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)

    depth = prediction.squeeze().cpu().numpy()
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

    return depth, depth_colored
