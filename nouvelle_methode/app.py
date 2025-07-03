# Fichier : app.py

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.cm as cm
from rembg import remove
import time

MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
os.environ['TORCH_HOME'] = MODELS_PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from core.segmentation import load_sam_model, segment_image
from core.depth_estimation import load_midas_model, estimate_depth
from core.placement import find_floor_mask, get_placement_point, calculate_scale_from_depth, insert_object
from core.harmonization import load_sd_inpainting_model, create_inpainting_mask, harmonize_image, load_captioning_model, generate_caption

def show_masks_on_image(image_np, masks):
    if not masks:
        fig, ax = plt.subplots(); ax.imshow(image_np); ax.axis("off"); return fig
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)
    for mask in sorted_masks:
        m = mask['segmentation']
        color = np.concatenate([np.random.random(3), [0.55]])
        ax.imshow(np.dstack((m, m, m, m)) * color)
    ax.axis("off"); plt.tight_layout(); return fig

def normalize_depth_map(depth_map):
    return (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

st.set_page_config(layout="wide", page_title="Projet de Staging Virtuel")
st.title("🛋️ Projet de Staging Virtuel IA")
st.markdown("Uploadez une image de pièce, puis une image d'objet pour l'intégrer, et enfin, rendez le tout photoréaliste.")
st.divider()

@st.cache_resource
def get_models():
    """Charge tous les modèles nécessaires (SAM, MiDaS, SD et BLIP)."""
    sam_generator = load_sam_model()
    midas_model, midas_transform = load_midas_model()
    sd_pipeline = load_sd_inpainting_model()
    caption_processor, caption_model = load_captioning_model()
    return sam_generator, (midas_model, midas_transform), sd_pipeline, (caption_processor, caption_model)

with st.spinner("Veuillez patienter, chargement des modèles d'IA (cela peut prendre du temps la première fois)..."):
    sam_mask_generator, midas_components, sd_pipeline, captioning_components = get_models()

col1_upload, col2_upload = st.columns(2)
with col1_upload:
    env_file = st.file_uploader("1. Choisissez une image de pièce...", type=["jpg", "png"])
with col2_upload:
    obj_file = st.file_uploader("2. Choisissez un objet...", type=["png", "jpg"])

if env_file:
    env_pil = Image.open(env_file)
    if 'analysis_done' not in st.session_state or st.session_state.env_file_name != env_file.name:
        with st.spinner("Analyse de la pièce en cours... 🧠"):
            start_time_analysis = time.perf_counter()
            st.session_state.image_np, st.session_state.masks = segment_image(env_pil, sam_mask_generator)
            st.session_state.depth_map = estimate_depth(env_pil, midas_components[0], midas_components[1])
            st.session_state.env_caption = generate_caption(env_pil, captioning_components[0], captioning_components[1])
            end_time_analysis = time.perf_counter()
            st.session_state.analysis_duration = end_time_analysis - start_time_analysis
            st.session_state.analysis_done = True
            st.session_state.env_file_name = env_file.name
    
    st.success(f"Analyse de la scène terminée en {st.session_state.analysis_duration:.2f} secondes.")
    
    if obj_file:
        if 'obj_file_name' not in st.session_state or st.session_state.obj_file_name != obj_file.name:
            st.session_state.obj_pil = Image.open(obj_file)
            st.session_state.obj_file_name = obj_file.name
            # CORRIGÉ : On utilise st.session_state.obj_pil qui existe, et non la variable locale "obj_pil"
            st.session_state.obj_caption = generate_caption(st.session_state.obj_pil, captioning_components[0], captioning_components[1])
            if 'obj_no_bg' in st.session_state: del st.session_state['obj_no_bg']
            if 'composite_image' in st.session_state: del st.session_state['composite_image']
            if 'final_image' in st.session_state: del st.session_state['final_image']

        st.subheader("Étape 1 : Préparation et Placement")
        col_param, col_result = st.columns([1, 2])
        
        with col_param:
            image_to_display = st.session_state.get('obj_no_bg', st.session_state.obj_pil)
            st.image(image_to_display, caption="Objet à placer", width=200)
            if st.button("✨ Supprimer le fond"):
                with st.spinner("Suppression du fond..."):
                    st.session_state.obj_no_bg = remove(st.session_state.obj_pil)
                st.rerun()
            base_scale_slider = st.slider("Ajuster la taille de base", 0.1, 2.0, 0.5, 0.05)
            if st.button("Placer dans la scène", use_container_width=True, type="primary"):
                with st.spinner("Intégration en cours..."):
                    floor = find_floor_mask(st.session_state.masks, st.session_state.image_np.shape[0])
                    if floor is not None:
                        point = get_placement_point(floor)
                        if point is not None:
                            scale = calculate_scale_from_depth(st.session_state.depth_map, point, base_scale=base_scale_slider)
                            st.session_state.composite_image, st.session_state.object_mask = insert_object(env_pil, image_to_display, point, scale, return_mask=True)
        
        with col_result:
            if 'composite_image' in st.session_state:
                st.image(st.session_state.composite_image, caption="Résultat du collage", use_container_width=True)
            else:
                st.info("Ajustez les paramètres et cliquez sur 'Placer dans la scène'.")

        st.divider()

        if 'composite_image' in st.session_state:
            st.subheader("Étape 2 : Harmonisation Photoréaliste (IA)")

            col_opts1, col_opts2 = st.columns(2)
            with col_opts1:
                sd_steps = st.slider("Qualité vs Vitesse (Étapes)", min_value=10, max_value=50, value=20, step=1)
            with col_opts2:
                sd_strength = st.slider("Force de l'harmonisation", min_value=0.5, max_value=1.0, value=0.85, step=0.05)

            auto_prompt = f"{st.session_state.get('obj_caption', 'an object')} in {st.session_state.get('env_caption', 'a room')}, photorealistic, realistic shadows, 8k, high quality"
            prompt_text = st.text_area("Prompt généré automatiquement (vous pouvez le modifier) :", auto_prompt, height=100)

            if st.button("🚀 Lancer l'harmonisation", use_container_width=True, type="primary"):
                if sd_pipeline is None:
                    st.error("Le modèle d'harmonisation (Stable Diffusion) n'a pas pu être chargé.")
                else:
                    with st.spinner(f"L'IA redessine l'image en {sd_steps} étapes... Un peu de patience !"):
                        start_time_harmonization = time.perf_counter()
                        inpainting_mask = create_inpainting_mask(st.session_state.object_mask)

                        final_image = harmonize_image(
                            st.session_state.composite_image,
                            inpainting_mask,
                            prompt_text,
                            sd_pipeline,
                            strength=sd_strength,
                            num_inference_steps=sd_steps
                        )
                        end_time_harmonization = time.perf_counter()
                        st.session_state.harmonization_duration = end_time_harmonization - start_time_harmonization
                        if final_image:
                            st.session_state.final_image = final_image

            if 'final_image' in st.session_state:
                duration_text = f"Généré en {st.session_state.harmonization_duration:.2f}s"
                st.image(st.session_state.final_image, caption=f"✨ Résultat Final Harmonisé ({duration_text}) ✨", use_container_width=True)
