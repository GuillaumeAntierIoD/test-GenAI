from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import numpy as np
import cv2

# === CONFIGURATION ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-inpainting"  # modèle Stable Diffusion inpainting

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# === 1. Charger l’image avec ton objet déjà collé ===
image_path = "objet_place.jpg"  # image avec objet déjà intégré (ex: OpenCV)
image = Image.open(image_path).convert("RGB")

# === 2. Créer un masque autour de l’objet (ex: un rectangle) ===
mask = np.zeros((image.height, image.width), dtype=np.uint8)
x, y, w, h = 300, 350, 220, 120  
cv2.rectangle(mask, (x-20, y-20), (x+w+20, y+h+20), 255, -1) 

# Convertir le masque en image binaire
mask_image = Image.fromarray(mask).convert("L")

# === 3. Prompt pour l'intégration ===
prompt = (
    "harmonize the inserted furniture with the room, realistic soft shadows under the object, "
    "consistent lighting, smooth edge blending, high quality photo"
)

# === 4. Génération avec inpainting ===
result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
).images[0]

# === 5. Sauvegarde / Affichage ===
result.save("result_integrated.jpg")
result.show()
