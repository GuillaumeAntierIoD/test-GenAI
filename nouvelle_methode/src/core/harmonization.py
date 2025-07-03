import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image
import numpy as np
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

def load_captioning_model():
    """Charge le modèle BLIP et son processeur pour la description d'images."""
    print("Chargement du modèle de description d'image (BLIP)...")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", torch_dtype=DTYPE
        ).to(DEVICE)
        print("Modèle BLIP chargé avec succès.")
        return processor, model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle BLIP : {e}")
        return None, None

def generate_caption(image_pil, processor, model):
    """Génère une description pour une image donnée."""
    if processor is None or model is None:
        return ""
    
    text = "a photography of"
    inputs = processor(images=image_pil.convert("RGB"), text=text, return_tensors="pt").to(DEVICE, DTYPE)
    
    generated_ids = model.generate(**inputs, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return generated_caption

def load_sd_inpainting_model():
    print(f"Chargement du modèle Stable Diffusion Inpainting sur le périphérique : {DEVICE}...")
    try:
        pipeline = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=DTYPE,
            variant="fp16"
        ).to(DEVICE)
        print("Modèle Stable Diffusion chargé avec succès.")
        return pipeline
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Stable Diffusion : {e}")
        return None

def create_inpainting_mask(object_mask, expansion_pixels=50):
    if object_mask is None: return None
    mask_uint8 = object_mask.astype(np.uint8) if object_mask.dtype != np.uint8 else object_mask
    kernel = np.ones((expansion_pixels, expansion_pixels), np.uint8)
    dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
    return Image.fromarray(dilated_mask)

def harmonize_image(composite_image_pil, inpainting_mask_pil, prompt, sd_pipeline, strength=0.85):
    if sd_pipeline is None: return None
    print("Début de l'harmonisation avec Stable Diffusion...")
    try:
        negative_prompt = "low quality, blurry, unrealistic, watermark, signature, text, ugly, deformed"
        harmonized_image = sd_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=composite_image_pil.convert("RGB"),
            mask_image=inpainting_mask_pil.convert("RGB"),
            strength=strength,
            num_inference_steps=25,
        ).images[0]
        print("Harmonisation terminée avec succès.")
        return harmonized_image
    except Exception as e:
        print(f"Une erreur est survenue lors de l'harmonisation : {e}")
        return None