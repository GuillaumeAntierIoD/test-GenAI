import torch
import streamlit as st
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import requests
import io
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import login
login("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")


st.set_page_config(layout="wide")
st.title("🪑 Meuble AI - Insertion dans un environnement avec Flux")

# -- Session state to hold uploaded images
if "mask_image" not in st.session_state:
    st.session_state.mask_image = None

# -- Upload environment image
env_col, meub_col = st.columns(2)
with env_col:
    env_image_file = st.file_uploader("📤 Image d'environnement", type=["png", "jpg", "jpeg"], key="env")
    if env_image_file:
        env_image = Image.open(env_image_file).convert("RGB")
        st.image(env_image, caption="Image d'environnement", use_column_width=True)
    else:
        st.stop()

# -- Upload furniture image (optional)
prompt_manual = True
prompt = ""
with meub_col:
    meub_image_file = st.file_uploader("🪑 Image du meuble (optionnel)", type=["png", "jpg", "jpeg"], key="meub")
    if meub_image_file:
        meub_image = Image.open(meub_image_file).convert("RGB")
        st.image(meub_image, caption="Meuble fourni")
        if st.checkbox("🧠 Générer automatiquement un prompt à partir de l'image du meuble"):
            prompt_manual = False
            with st.spinner("Génération du prompt avec BLIP..."):
                processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                inputs = processor(meub_image, return_tensors="pt")
                out = model.generate(**inputs)
                prompt = processor.decode(out[0], skip_special_tokens=True)
                st.success(f"Prompt généré : {prompt}")

if prompt_manual:
    prompt = st.text_input("✍️ Prompt pour le meuble", value="a modern wooden chair")

# -- Mask drawing
st.subheader("🎯 Sélection de la zone à remplir (dessinez sur l'image)")
canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",
    stroke_width=30,
    stroke_color="#FF0000",
    background_image=env_image,
    update_streamlit=True,
    height=env_image.height if env_image else 512,
    width=env_image.width if env_image else 512,
    drawing_mode="freedraw",
    key="canvas_mask"
)

# -- Convert mask
if canvas_result.image_data is not None:
    mask_image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
    st.session_state.mask_image = mask_image
    st.image(mask_image, caption="Mask généré", use_column_width=True)

# -- Generation Trigger
if st.button("🚀 Générer avec Flux"):
    if not prompt:
        st.error("Veuillez entrer un prompt.")
    elif not st.session_state.mask_image:
        st.error("Veuillez dessiner un masque sur l'image.")
    else:
        with st.spinner("Génération en cours avec Flux..."):
            # Charger les images et le modèle
            pipe = FluxFillPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Fill-dev",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

            image_input = env_image
            mask_input = st.session_state.mask_image

            result = pipe(
                prompt=prompt,
                image=image_input,
                mask_image=mask_input,
                guidance_scale=30,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.manual_seed(0)
            ).images[0]

            st.image(result, caption="Résultat généré", use_column_width=True)
            result.save("flux_result.png")
            st.success("Image enregistrée sous flux_result.png")
