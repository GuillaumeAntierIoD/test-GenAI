import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
import time

# Étape 1 : Vérifier la disponibilité du GPU
if not torch.cuda.is_available():
    print("❌ Erreur : Ce script optimisé nécessite un GPU (CUDA).")
    exit()

device = "cuda"
# On passe en float16, plus standard et compatible pour économiser la mémoire.
torch_dtype = torch.float16 
print("✅ GPU détecté ! Configuration avancée pour VRAM limitée en cours...")

# Étape 2 : Charger les images (inchangé)
try:
    image = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
    mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")
except Exception as e:
    print(f"Erreur lors du chargement des images : {e}")
    exit()

# Étape 3 : Charger le pipeline
print("Chargement du modèle...")
pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev", 
    torch_dtype=torch_dtype
)

# Étape 4 : Activer les optimisations mémoire
# On active le déchargement du modèle principal sur le CPU.
pipe.enable_model_cpu_offload()
print("-> 'CPU Offloading' est activé.")

# NOUVELLE ÉTAPE : On active le découpage du VAE pour économiser encore plus de VRAM.
pipe.enable_vae_slicing()
print("-> 'VAE Slicing' est activé.")

# Étape 5 : Exécuter l'inférence
print("Début de la génération de l'image...")
start_time = time.time()

# --- Options à ajuster ---
# On réduit encore la résolution. C'est le changement le plus important !
output_height = 768
output_width = 512
# -------------------------

image_result = pipe(
    prompt="a white paper cup",
    image=image,
    mask_image=mask,
    height=output_height,
    width=output_width,
    guidance_scale=3.0,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator(device).manual_seed(0) 
).images[0]

end_time = time.time()
print(f"🎉 Image générée en {end_time - start_time:.2f} secondes !")

# Étape 6 : Sauvegarder l'image
output_filename = "flux-fill-dev-result-vram-super-optimized.png"
image_result.save(output_filename)
print(f"Image sauvegardée sous le nom : {output_filename}")