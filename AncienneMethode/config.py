"""
Configuration file for the furniture placement application
"""

import os

# ======================= Model Paths =======================
SAM_PATH = "models/sam_vit_b_01ec64.pth"
MIDAS_PATH = "models/dpt_large_384.pt"

# ======================= Device Configuration =======================
DEVICE = "cpu"  # Change to "cuda" if you have a compatible GPU
SD_DEVICE = "cpu"  # Stable Diffusion device

# ======================= SAM Configuration =======================
POINTS_PER_SIDE = 16  # Number of points per side for SAM automatic mask generation
SAM_PRED_IOU_THRESH = 0.88  # IoU threshold for SAM predictions
SAM_STABILITY_SCORE_THRESH = 0.95  # Stability score threshold

# ======================= MiDaS Configuration =======================
MIDAS_MODEL_TYPE = "dpt_large_384"  # Model type for MiDaS
MIDAS_INPUT_SIZE = 384  # Input size for MiDaS

# ======================= Stable Diffusion Configuration =======================
ENABLE_SD = True  # Enable/disable Stable Diffusion features
SD_MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"
SD_IMG2IMG_MODEL_ID = "stabilityai/stable-diffusion-2-1"

# Default parameters for SD
DEFAULT_SD_STEPS = 20
DEFAULT_SD_GUIDANCE_SCALE = 7.5
DEFAULT_SD_STRENGTH = 0.3
DEFAULT_INPAINT_STRENGTH = 0.8

# ======================= OpenCV Configuration =======================
MIN_SURFACE_AREA = 1000  # Minimum area for surface detection
SHADOW_INTENSITY = 0.3  # Default shadow intensity
EXPANSION_RADIUS = 20  # Radius for inpainting mask expansion

# ======================= Color Harmony Configuration =======================
COLOR_HARMONY_STRENGTH = 0.2  # Strength of color harmony adjustment
BLUR_RADIUS = 3  # Radius for edge blurring in post-processing

# ======================= Performance Configuration =======================
ENABLE_ATTENTION_SLICING = True  # Enable attention slicing for SD (saves memory)
USE_HALF_PRECISION = False  # Use half precision for models (requires CUDA)

# ======================= UI Configuration =======================
DEFAULT_FURNITURE_TYPES = [
    "sofa", "chair", "table", "bookshelf", "lamp", 
    "cabinet", "bed", "desk", "wardrobe", "nightstand"
]

DEFAULT_ROOM_TYPES = [
    "living room", "bedroom", "dining room", "office", 
    "kitchen", "bathroom", "study", "hallway"
]

LIGHTING_CONDITIONS = [
    "natural", "artificial", "mixed", "dim"
]

# ======================= Quality Presets =======================
QUALITY_PRESETS = {
    "fast": {
        "sam_points_per_side": 8,
        "sd_steps": 10,
        "sd_guidance_scale": 5.0,
        "enable_post_processing": False
    },
    "balanced": {
        "sam_points_per_side": 16,
        "sd_steps": 20,
        "sd_guidance_scale": 7.5,
        "enable_post_processing": True
    },
    "high_quality": {
        "sam_points_per_side": 32,
        "sd_steps": 50,
        "sd_guidance_scale": 12.0,
        "enable_post_processing": True
    }
}

# ======================= Error Messages =======================
ERROR_MESSAGES = {
    "sam_model_not_found": f"Modèle SAM non trouvé : {SAM_PATH}. Téléchargez-le depuis https://github.com/facebookresearch/segment-anything",
    "midas_model_not_found": f"Modèle MiDaS non trouvé : {MIDAS_PATH}. Vérifiez le chemin du modèle.",
    "diffusers_not_installed": "La bibliothèque diffusers n'est pas installée. Installez avec : pip install diffusers",
    "cuda_not_available": "CUDA n'est pas disponible. Basculement vers CPU.",
    "insufficient_memory": "Mémoire insuffisante. Essayez de réduire la taille de l'image ou utilisez des paramètres plus légers."
}

# ======================= Helper Functions =======================
def get_model_path(model_name: str) -> str:
    """Get the full path for a model file"""
    model_paths = {
        "sam": SAM_PATH,
        "midas": MIDAS_PATH
    }
    return model_paths.get(model_name, "")

def check_models_exist() -> dict:
    """Check if required model files exist"""
    return {
        "sam": os.path.exists(SAM_PATH),
        "midas": os.path.exists(MIDAS_PATH)
    }

def get_quality_preset(preset_name: str) -> dict:
    """Get configuration for a quality preset"""
    return QUALITY_PRESETS.get(preset_name, QUALITY_PRESETS["balanced"])

def validate_device(device: str) -> str:
    """Validate and return appropriate device"""
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                print("CUDA requested but not available, falling back to CPU")
                return "cpu"
        except ImportError:
            print("PyTorch not available, using CPU")
            return "cpu"
    return "cpu"