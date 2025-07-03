import torch
import streamlit as st
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import login
from dotenv import load_dotenv
import os
import random
import shutil
from huggingface_hub import HfFolder

HfFolder.delete_token()

# Chargement du token HuggingFace
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

st.set_page_config(layout="wide")
st.title("🪑 Meuble AI - Insertion dans un environnement avec Stable Diffusion")

# ----------------------
# INITIALISATION FLAGS
# ----------------------
if "pipe" not in st.session_state:
    st.session_state.pipe = None

if "prompt" not in st.session_state:
    st.session_state.prompt = ""

if "prompt_generated" not in st.session_state:
    st.session_state.prompt_generated = False

if "env_image" not in st.session_state:
    st.session_state.env_image = None

if "meub_image" not in st.session_state:
    st.session_state.meub_image = None

if "mask_image" not in st.session_state:
    st.session_state.mask_image = None

if "generation_done" not in st.session_state:
    st.session_state.generation_done = False

if "result_image" not in st.session_state:
    st.session_state.result_image = None

# ----------------------
# CHARGEMENT PIPELINE (1 seule fois)
# ----------------------
@st.cache_resource
def load_pipeline():
    """Version simplifiée avec nettoyage automatique"""
    
    def safe_load():
        try:
            return StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
        except Exception as e:
            if "diffusion_pytorch_model.safetensors" in str(e):
                # Nettoyage du cache et retry
                cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-inpainting")
                if os.path.exists(cache_path):
                    shutil.rmtree(cache_path)
                
                return StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    force_download=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
            else:
                raise e
    
    pipe = safe_load()
    pipe.enable_attention_slicing()
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    return pipe

# ----------------------
# UPLOAD IMAGE ENVIRONNEMENT
# ----------------------
env_col, meub_col = st.columns(2)
with env_col:
    env_image_file = st.file_uploader("📤 Image d'environnement", type=["png", "jpg", "jpeg"], key="env")
    if env_image_file:
        st.session_state.env_image = Image.open(env_image_file).convert("RGB")
        # Redimensionner pour optimiser les performances
        max_size = 512
        st.session_state.env_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        st.image(st.session_state.env_image, caption="Image d'environnement", use_column_width=True)
    else:
        st.info("Veuillez uploader une image d'environnement")
        st.stop()

# ----------------------
# UPLOAD IMAGE MEUBLE ET GÉNÉRATION PROMPT (optionnel)
# ----------------------
prompt_manual = True
with meub_col:
    meub_image_file = st.file_uploader("🪑 Image du meuble (optionnel)", type=["png", "jpg", "jpeg"], key="meub")
    if meub_image_file:
        st.session_state.meub_image = Image.open(meub_image_file).convert("RGB")
        st.image(st.session_state.meub_image, caption="Meuble fourni")
        if st.checkbox("🧠 Générer automatiquement un prompt à partir de l'image du meuble"):
            if not st.session_state.prompt_generated:
                with st.spinner("Génération du prompt avec BLIP..."):
                    try:
                        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                        inputs = processor(st.session_state.meub_image, return_tensors="pt")
                        out = model.generate(**inputs)
                        st.session_state.prompt = processor.decode(out[0], skip_special_tokens=True)
                        st.session_state.prompt_generated = True
                        st.success(f"Prompt généré : {st.session_state.prompt}")
                    except Exception as e:
                        st.error(f"Erreur lors de la génération du prompt : {e}")
                        st.session_state.prompt = "a modern wooden chair"
            prompt_manual = False

if prompt_manual:
    st.session_state.prompt = st.text_input("✍️ Prompt pour le meuble", value=st.session_state.prompt or "a modern wooden chair")

# ----------------------
# DESSIN DU MASQUE - CORRECTION MAJEURE
# ----------------------
st.subheader("🎯 Sélection de la zone à remplir (dessinez sur l'image)")
st.info("💡 Dessinez en blanc sur les zones où vous voulez placer le meuble")

if st.session_state.env_image:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.8)",  # Blanc avec transparence
        stroke_width=30,
        stroke_color="#FFFFFF",  # Blanc pour le masque
        background_image=st.session_state.env_image,
        update_streamlit=True,
        height=min(st.session_state.env_image.height, 512),
        width=min(st.session_state.env_image.width, 512),
        drawing_mode="freedraw",
        key="canvas_mask"
    )

    if canvas_result.image_data is not None:
        canvas_array = canvas_result.image_data
        
        alpha_channel = canvas_array[:, :, 3]  # Canal alpha
        mask_array = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)
        
        mask_image = Image.fromarray(mask_array, mode='L')
        
        mask_image = mask_image.resize(st.session_state.env_image.size, Image.Resampling.NEAREST)
        
        st.session_state.mask_image = mask_image
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.mask_image, caption="Masque généré (blanc = zone à remplir)", use_column_width=True)
        with col2:
            # Aperçu du masque superposé
            preview = st.session_state.env_image.copy()
            preview.paste(Image.new('RGB', preview.size, (255, 0, 0)), mask=st.session_state.mask_image)
            st.image(preview, caption="Aperçu (rouge = zone à remplir)", use_column_width=True)

# ----------------------
# GÉNÉRATION D'IMAGE - CORRECTIONS
# ----------------------
if st.button("🚀 Générer avec Stable Diffusion"):
    if not st.session_state.prompt.strip():
        st.error("Veuillez entrer un prompt.")
    elif st.session_state.mask_image is None:
        st.error("Veuillez dessiner un masque sur l'image.")
    else:
        # Vérifier que le masque n'est pas vide
        mask_array = np.array(st.session_state.mask_image)
        if np.sum(mask_array) == 0:
            st.error("Le masque est vide. Veuillez dessiner sur l'image.")
        else:
            try:
                # Charger le pipeline
                if st.session_state.pipe is None:
                    with st.spinner("Chargement du modèle..."):
                        st.session_state.pipe = load_pipeline()
                
                with st.spinner("Génération en cours avec Stable Diffusion..."):
                    generator = torch.Generator().manual_seed(random.randint(0, 2**32 - 1))

                    # Paramètres optimisés pour l'inpainting
                    result = st.session_state.pipe(
                        prompt=st.session_state.prompt,
                        image=st.session_state.env_image,
                        mask_image=st.session_state.mask_image,
                        guidance_scale=7.5,
                        num_inference_steps=50,  
                        strength=0.99,  
                        generator=generator
                    ).images[0]
                    
                    st.session_state.result_image = result
                    st.session_state.generation_done = True
                    
            except Exception as e:
                st.error(f"Erreur lors de la génération : {e}")

# ----------------------
# AFFICHAGE RÉSULTAT
# ----------------------
if st.session_state.generation_done and st.session_state.result_image:
    st.subheader("✨ Résultat")
    
    # Comparaison avant/après
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.env_image, caption="Image originale", use_column_width=True)
    with col2:
        st.image(st.session_state.result_image, caption="Résultat avec inpainting", use_column_width=True)
    
    # Bouton de téléchargement
    if st.button("💾 Sauvegarder le résultat"):
        st.session_state.result_image.save("inpainting_result.png")
        st.success("Image sauvegardée sous 'inpainting_result.png'")

# ----------------------
# INFORMATIONS DE DEBUG
# ----------------------
if st.checkbox("🔧 Afficher les informations de debug"):
    st.write("**États des variables :**")
    st.write(f"- Prompt: {st.session_state.prompt}")
    st.write(f"- Image environnement chargée: {st.session_state.env_image is not None}")
    st.write(f"- Masque créé: {st.session_state.mask_image is not None}")
    if st.session_state.mask_image:
        mask_stats = np.array(st.session_state.mask_image)
        st.write(f"- Pixels blancs dans le masque: {np.sum(mask_stats > 0)}")
        st.write(f"- Taille du masque: {st.session_state.mask_image.size}")
    st.write(f"- Pipeline chargé: {st.session_state.pipe is not None}")
    st.write(f"- Génération terminée: {st.session_state.generation_done}")