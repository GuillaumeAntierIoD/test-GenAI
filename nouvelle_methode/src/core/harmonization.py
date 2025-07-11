import torch
from diffusers import AutoPipelineForInpainting, EulerDiscreteScheduler
from PIL import Image
import numpy as np
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Callable, Optional

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
    

def generate_adaptive_prompt(obj_caption, env_caption):
    """
    Génère un prompt adaptatif optimisé pour CLIP (77 tokens max)
    """
    
    # Détection du matériau (version simplifiée)
    material_keywords = {
        'metal': ['metal', 'iron', 'steel', 'aluminum', 'wrought'],
        'wood': ['wood', 'wooden', 'timber'],
        'stone': ['stone', 'concrete', 'brick'],
        'glass': ['glass', 'transparent']
    }
    
    # Détection du sol (version simplifiée)
    surface_keywords = {
        'paved': ['pavement', 'paved', 'stone', 'brick', 'tile'],
        'concrete': ['concrete', 'cement'],
        'gravel': ['gravel', 'pebble'],
        'grass': ['grass', 'lawn', 'garden']
    }
    
    # Détection matériau
    detected_material = 'generic'
    for material, keywords in material_keywords.items():
        if any(keyword in obj_caption.lower() for keyword in keywords):
            detected_material = material
            break
    
    # Détection surface
    detected_surface = 'ground'
    for surface, keywords in surface_keywords.items():
        if any(keyword in env_caption.lower() for keyword in keywords):
            detected_surface = surface
            break
    
    # Prompts courts et efficaces par matériau
    material_prompts = {
        'metal': 'weathered metal with realistic reflections and shadows',
        'wood': 'natural wood with grain texture and weathering',
        'stone': 'stone texture with natural weathering',
        'glass': 'clear glass with realistic reflections',
        'generic': 'natural surface with realistic shadows'
    }
    
    # Prompts de surface courts
    surface_prompts = {
        'paved': 'natural wear on pavement',
        'concrete': 'subtle concrete staining',
        'gravel': 'natural gravel displacement',
        'grass': 'grass wear patterns',
        'ground': 'natural ground wear'
    }
    
    # Construction du prompt court (viser ~60 tokens)
    prompt = f"{obj_caption} in {env_caption}. Perfect integration. {material_prompts[detected_material]}. {surface_prompts[detected_surface]}. Professional photography quality."
    
    return prompt

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
    """Charge le pipeline Stable Diffusion avec des optimisations mémoire."""
    print(f"Chargement du modèle Stable Diffusion Inpainting sur le périphérique : {DEVICE}...")
    try:
        model_id = "runwayml/stable-diffusion-inpainting"
        pipeline = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=DTYPE)

        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(DEVICE)

        if DEVICE == "cuda":
            pipeline.enable_attention_slicing()
            pipeline.enable_sequential_cpu_offload()

        print("Modèle Stable Diffusion chargé avec succès.")
        return pipeline
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Stable Diffusion : {e}")
        return None

def create_inpainting_mask(object_mask):
    """
    Crée un masque d'inpainting en dilatant et floutant le masque de l'objet.
    Accepte une image PIL ou un array NumPy en entrée.
    """

    if isinstance(object_mask, Image.Image):
        mask_np = np.array(object_mask)
    else:
        mask_np = object_mask

    if mask_np.ndim == 3:
        if mask_np.shape[2] == 4: 
            mask_np = mask_np[:, :, 3] 
        else: 
            mask_np = mask_np[:, :, 0] 

    
    mask_uint8 = mask_np.astype(np.uint8) if mask_np.dtype != np.uint8 else mask_np

    kernel = np.ones((15, 15), np.uint8) 
    dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)

    blurred_mask = cv2.GaussianBlur(dilated_mask, (31, 31), 0) 

    final_mask = Image.fromarray(blurred_mask)
    
    return final_mask

def harmonize_image(composite_image_pil, 
                    inpainting_mask_pil, 
                    prompt, 
                    sd_pipeline, 
                    strength=0.35, 
                    num_inference_steps=7, 
                    progress_callback: Optional[Callable] = None):
    
    if sd_pipeline is None: return None
    print(f"Début de l'harmonisation ({num_inference_steps} étapes)...")
    try:
        negative_prompt = "low quality, blurry, unrealistic, watermark, signature, text, ugly, deformed"
        
        with torch.inference_mode():
            harmonized_image = sd_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=composite_image_pil.convert("RGB"),
                mask_image=inpainting_mask_pil.convert("RGB"),
                strength=strength,
                num_inference_steps=num_inference_steps,
                callback_on_step_end=progress_callback
            ).images[0]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Harmonisation terminée avec succès.")
        return harmonized_image
    except Exception as e:
        print(f"Une erreur est survenue lors de l'harmonisation : {e}")
        return None