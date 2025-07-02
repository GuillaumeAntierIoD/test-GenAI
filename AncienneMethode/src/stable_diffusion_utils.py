import numpy as np
import torch
from PIL import Image
import cv2
from typing import Optional, Tuple
import io
import base64

try:
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
    from diffusers import DPMSolverMultistepScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

class StableDiffusionProcessor:
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-inpainting", device: str = "cpu"):
        """
        Initialise le processeur Stable Diffusion
        
        Args:
            model_id: ID du modèle Hugging Face
            device: Device à utiliser
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library not installed. Please install with: pip install diffusers")
        
        self.device = device
        self.model_id = model_id
        self.pipe = None
        self.img2img_pipe = None
        
    def load_inpainting_model(self):
        """Charge le modèle d'inpainting"""
        if self.pipe is None:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)
            
            if self.device == "cpu":
                self.pipe.enable_attention_slicing()
        
        return self.pipe
    
    def load_img2img_model(self):
        """Charge le modèle img2img"""
        if self.img2img_pipe is None:
            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.img2img_pipe = self.img2img_pipe.to(self.device)
            
            if self.device == "cpu":
                self.img2img_pipe.enable_attention_slicing()
        
        return self.img2img_pipe

def create_furniture_integration_prompt(furniture_type: str, room_type: str, lighting_conditions: str = "natural") -> str:
    """
    Crée un prompt optimisé pour l'intégration de meuble
    
    Args:
        furniture_type: Type de meuble (sofa, table, chair, etc.)
        room_type: Type de pièce (living room, bedroom, etc.)
        lighting_conditions: Conditions d'éclairage
    
    Returns:
        Prompt optimisé
    """
    base_prompt = f"A realistic {furniture_type} naturally placed in a {room_type}, "
    
    lighting_map = {
        "natural": "with natural daylight, soft shadows, warm ambiance",
        "artificial": "with artificial lighting, lamp shadows, cozy atmosphere",
        "mixed": "with mixed natural and artificial lighting, balanced shadows",
        "dim": "with dim ambient lighting, subtle shadows, intimate atmosphere"
    }
    
    lighting_desc = lighting_map.get(lighting_conditions, lighting_map["natural"])
    
    quality_prompt = (
        ", photorealistic, high quality, detailed textures, "
        "proper perspective, realistic proportions, seamless integration, "
        "professional interior design, 8k resolution"
    )
    
    negative_prompt = (
        "blurry, distorted, unrealistic proportions, floating objects, "
        "artificial looking, poor lighting, low quality, pixelated, "
        "cartoon, painting, drawing, sketch"
    )
    
    full_prompt = base_prompt + lighting_desc + quality_prompt
    
    return full_prompt, negative_prompt

def enhance_with_inpainting(image: np.ndarray, mask: np.ndarray, furniture_type: str, 
                          room_type: str, sd_processor: StableDiffusionProcessor,
                          strength: float = 0.8, guidance_scale: float = 7.5) -> np.ndarray:
    """
    Améliore l'intégration avec Stable Diffusion inpainting
    
    Args:
        image: Image avec l'objet placé
        mask: Masque de la zone à inpainter
        furniture_type: Type de meuble
        room_type: Type de pièce
        sd_processor: Processeur Stable Diffusion
        strength: Force de l'inpainting (0-1)
        guidance_scale: Échelle de guidance
    
    Returns:
        Image améliorée
    """
    pipe = sd_processor.load_inpainting_model()
    
    # Convertir en PIL
    image_pil = Image.fromarray(image).convert("RGB")
    mask_pil = Image.fromarray(mask).convert("L")
    
    # Créer le prompt
    prompt, negative_prompt = create_furniture_integration_prompt(furniture_type, room_type)
    
    # Génération
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=20,
            guidance_scale=guidance_scale,
            strength=strength
        ).images[0]
    
    return np.array(result)

def enhance_with_img2img(image: np.ndarray, furniture_type: str, room_type: str,
                        sd_processor: StableDiffusionProcessor,
                        strength: float = 0.3, guidance_scale: float = 7.5) -> np.ndarray:
    """
    Améliore l'image complète avec img2img
    
    Args:
        image: Image d'entrée
        furniture_type: Type de meuble
        room_type: Type de pièce
        sd_processor: Processeur Stable Diffusion
        strength: Force de la transformation
        guidance_scale: Échelle de guidance
    
    Returns:
        Image améliorée
    """
    pipe = sd_processor.load_img2img_model()
    
    # Convertir en PIL et redimensionner si nécessaire
    image_pil = Image.fromarray(image).convert("RGB")
    
    # Redimensionner pour SD (multiples de 8)
    w, h = image_pil.size
    w = (w // 8) * 8
    h = (h // 8) * 8
    image_pil = image_pil.resize((w, h))
    
    # Créer le prompt
    prompt, negative_prompt = create_furniture_integration_prompt(furniture_type, room_type)
    
    # Génération
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            num_inference_steps=20,
            guidance_scale=guidance_scale,
            strength=strength
        ).images[0]
    
    return np.array(result)

def create_inpainting_mask_from_object(obj_mask: np.ndarray, expansion_radius: int = 20) -> np.ndarray:
    """
    Crée un masque d'inpainting autour de l'objet
    
    Args:
        obj_mask: Masque de l'objet
        expansion_radius: Rayon d'expansion du masque
    
    Returns:
        Masque d'inpainting élargi
    """
    # Dilater le masque pour inclure les zones environnantes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expansion_radius*2, expansion_radius*2))
    expanded_mask = cv2.dilate(obj_mask, kernel, iterations=1)
    
    # Appliquer un flou gaussien pour des bords plus doux
    expanded_mask = cv2.GaussianBlur(expanded_mask, (5, 5), 0)
    
    return expanded_mask

def detect_lighting_conditions(image: np.ndarray) -> str:
    """
    Détecte automatiquement les conditions d'éclairage de l'image
    
    Args:
        image: Image d'entrée
    
    Returns:
        Type d'éclairage détecté
    """
    # Convertir en LAB pour analyser la luminosité
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    
    # Calculer des statistiques de luminosité
    mean_brightness = np.mean(l_channel)
    brightness_std = np.std(l_channel)
    
    # Analyser la distribution de la luminosité
    hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
    
    # Détecter le type d'éclairage basé sur les statistiques
    if mean_brightness < 80:
        return "dim"
    elif mean_brightness > 150 and brightness_std > 30:
        return "mixed"
    elif brightness_std < 20:
        return "artificial"
    else:
        return "natural"

def apply_color_harmony(image: np.ndarray, reference_colors: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Applique une harmonisation des couleurs pour une meilleure intégration
    
    Args:
        image: Image d'entrée
        reference_colors: Couleurs de référence (optionnel)
    
    Returns:
        Image avec harmonisation des couleurs
    """
    # Convertir en LAB pour manipulation des couleurs
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    if reference_colors is not None:
        # Utiliser les couleurs de référence pour l'harmonisation
        ref_lab = cv2.cvtColor(reference_colors.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)
        ref_mean_a = np.mean(ref_lab[:, :, 1])
        ref_mean_b = np.mean(ref_lab[:, :, 2])
        
        # Ajuster les canaux chromatiques
        lab[:, :, 1] = lab[:, :, 1] * 0.8 + ref_mean_a * 0.2
        lab[:, :, 2] = lab[:, :, 2] * 0.8 + ref_mean_b * 0.2
    else:
        # Harmonisation générale
        lab[:, :, 1] = lab[:, :, 1] * 0.9  # Réduire légèrement la saturation verte-rouge
        lab[:, :, 2] = lab[:, :, 2] * 0.9  # Réduire légèrement la saturation bleu-jaune
    
    # Convertir de retour en RGB
    lab = np.clip(lab, 0, 255)
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    return result

def post_process_integration(image: np.ndarray, obj_mask: np.ndarray) -> np.ndarray:
    """
    Post-traitement pour améliorer l'intégration visuelle
    
    Args:
        image: Image avec objet intégré
        obj_mask: Masque de l'objet
    
    Returns:
        Image post-traitée
    """
    result = image.copy()
    
    # Appliquer un léger flou aux bords de l'objet pour une meilleure intégration
    edge_mask = cv2.Canny(obj_mask, 50, 150)
    edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)
    
    # Créer une version légèrement floutée
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Mélanger aux bords
    edge_mask_norm = edge_mask.astype(np.float32) / 255.0
    if len(edge_mask_norm.shape) == 2:
        edge_mask_norm = np.stack([edge_mask_norm] * 3, axis=2)
    
    result = result * (1 - edge_mask_norm * 0.3) + blurred * (edge_mask_norm * 0.3)
    
    return result.astype(np.uint8)