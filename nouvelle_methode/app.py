import streamlit as st
from PIL import Image
import numpy as np
import sys
import os
import torch
from rembg import remove
from streamlit_drawable_canvas import st_canvas
import time
import cv2

# --- Configuration des chemins et imports (nettoyés) ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from core.placement import insert_object_with_perspective
from core.harmonization import load_sd_inpainting_model, create_inpainting_mask, harmonize_image, load_captioning_model, generate_caption, generate_adaptive_prompt

# --- Interface Streamlit ---
st.set_page_config(layout="wide", page_title="Projet de Staging Virtuel")
st.title("🛋️ Projet de Staging Virtuel par Perspective")

# --- Initialisation des modèles (à la demande) ---
if 'models' not in st.session_state:
    st.session_state.models = {}
def get_model(model_name, load_function):
    """Charge un modèle et le met en cache dans la session Streamlit."""
    if model_name not in st.session_state.models:
        with st.spinner(f"Chargement du modèle {model_name}..."):
            st.session_state.models[model_name] = load_function()
    return st.session_state.models[model_name]

# --- UI (Barre latérale) ---
st.sidebar.header("Commandes")
env_file = st.sidebar.file_uploader("1. Image de l'environnement", type=["jpg", "png"])
obj_file = st.sidebar.file_uploader("2. Image de l'objet", type=["png", "jpg"])

# --- Logique principale (simplifiée) ---
if not env_file:
    st.info("Bienvenue ! Veuillez uploader une image d'environnement pour commencer.")
else:
    env_pil = Image.open(env_file)
    if st.session_state.get('env_file_name') != env_file.name:
        keys_to_clear = ['obj_file_name', 'obj_pil', 'obj_no_bg', 'composite_image', 'final_image', 'object_mask', 'perspective_points']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.env_file_name = env_file.name
    
    if 'canvas_key_counter' not in st.session_state:
        st.session_state.canvas_key_counter = 0

    if obj_file:
        st.sidebar.divider()
        st.sidebar.subheader("Objet à placer")
        if st.session_state.get('obj_file_name') != obj_file.name:
            st.session_state.obj_file_name = obj_file.name
            original_obj_pil = Image.open(obj_file)
            with st.spinner("Suppression du fond et rognage..."):
                img_no_bg = remove(original_obj_pil)
                bbox = img_no_bg.getbbox()
                if bbox:
                    st.session_state.obj_no_bg = img_no_bg.crop(bbox)
                else:
                    st.session_state.obj_no_bg = img_no_bg
        
            for key in ['composite_image', 'perspective_points']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        obj_to_display = st.session_state.get('obj_no_bg')
        if obj_to_display: st.sidebar.image(obj_to_display)
    
    if obj_file:
        st.subheader("Définissez la perspective avec 4 points")
        st.markdown("""
        **Instructions :** Cliquez sur 4 points pour définir les coins de votre objet.
        *Respectez cet ordre pour un placement correct :*
        1. 🔴 **Haut-gauche**
        2. 🟡 **Haut-droit**
        3. 🟢 **Bas-droit**
        4. 🔵 **Bas-gauche**
        """)
        
        background_image_to_show = st.session_state.get('composite_image', env_pil)

        canvas_result = st_canvas(
            background_image=background_image_to_show,
            height=env_pil.height,
            width=env_pil.width,
            drawing_mode="point",
            point_display_radius=8,
            update_streamlit=True,
            key=f"canvas_{st.session_state.canvas_key_counter}",
        )
        
        if canvas_result.json_data and canvas_result.json_data.get("objects"):
            points = canvas_result.json_data["objects"]
            
            point_count = len(points)
            if point_count == 1: st.info("🔴 Point 1 placé. Cliquez pour le point 2 (coin haut-droit)")
            elif point_count == 2: st.info("🟡 Point 2 placé. Cliquez pour le point 3 (coin bas-droit)")
            elif point_count == 3: st.info("🟢 Point 3 placé. Cliquez pour le point 4 (coin bas-gauche)")

            if len(points) >= 4:
                last_4_points = points[-4:]
                current_quad = [(int(p["left"]), int(p["top"])) for p in last_4_points]
                
                if (current_quad != st.session_state.get("perspective_points")):
                    st.session_state.perspective_points = current_quad
                    obj_to_place = st.session_state.get('obj_no_bg')
                    if obj_to_place:
                        composite_image, object_mask = insert_object_with_perspective(
                            env_pil, obj_to_place, current_quad, return_mask=True)
                        st.session_state.composite_image = composite_image
                        st.session_state.object_mask = object_mask
                    st.rerun()
                
                st.success("✅ 4 points placés ! L'objet est positionné.")
                if st.button("🔄 Recommencer le placement"):
                    st.session_state.perspective_points = []
                    if 'composite_image' in st.session_state: del st.session_state['composite_image']
                    if 'object_mask' in st.session_state: del st.session_state['object_mask']
                    st.session_state.canvas_key_counter += 1
                    st.rerun()

        # --- Étape 3 : Harmonisation (si un collage existe) ---
        if 'composite_image' in st.session_state:
            st.sidebar.divider()
            st.sidebar.subheader("Étape 3 : Harmonisation (IA)")
                
            if st.sidebar.button("🚀 Lancer l'harmonisation", use_container_width=True):
        
                st.sidebar.markdown("Début de l'harmonisation...")
                if 'env_caption' not in st.session_state:
                    captioning_components = get_model('captioning', load_captioning_model)
                    st.session_state.env_caption = generate_caption(env_pil, captioning_components[0], captioning_components[1])
                if 'obj_caption' not in st.session_state:
                    captioning_components = get_model('captioning', load_captioning_model)
                    obj_to_place = st.session_state.get('obj_no_bg', st.session_state.get('obj_pil'))
                    st.session_state.obj_caption = generate_caption(obj_to_place, captioning_components[0], captioning_components[1])

                auto_prompt = generate_adaptive_prompt(
                    st.session_state.get('obj_caption', 'gate'), 
                    st.session_state.get('env_caption', 'outdoor entrance')
                )    
                if 'captioning' in st.session_state.models:
                        del st.session_state.models['captioning']
                if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                progress_bar = st.sidebar.progress(0)
                progress_text = st.sidebar.empty()

                base_inference_steps = 10 
                strength_value = 0.4
                actual_steps_to_run = int(base_inference_steps * strength_value) 

                def update_progress(pipe, step, timestep, kwargs):
                    percentage = int(((step + 1) / actual_steps_to_run) * 100)
                    progress_bar.progress(percentage)
                    progress_text.text(f"Étape {step + 1}/{actual_steps_to_run} en cours...")
                    return kwargs

                with st.spinner("Chargement des modèles et préparation..."):
                    sd_pipeline = get_model('sd_pipeline', load_sd_inpainting_model)
                    
                    inpainting_mask = create_inpainting_mask(st.session_state.object_mask)

                final_image = harmonize_image(
                    st.session_state.composite_image, 
                    inpainting_mask, 
                    auto_prompt, 
                    sd_pipeline, 
                    strength=strength_value,
                    num_inference_steps=base_inference_steps,
                    progress_callback=update_progress 
                )
                
                progress_bar.progress(100)
                progress_text.success("Harmonisation terminée avec succès !") 
                
                st.session_state.final_image = final_image

                if 'sd_pipeline' in st.session_state.models:
                    del st.session_state.models['sd_pipeline']
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                import time
                time.sleep(2) 

                st.rerun()

        if 'final_image' in st.session_state and 'composite_image' in st.session_state:
            st.subheader("Comparaison Avant/Après")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Avant harmonisation**")
                st.image(st.session_state.composite_image)
            
            with col2:
                st.markdown("**Après harmonisation**")
                st.image(st.session_state.final_image)
            
            # Bouton pour télécharger le résultat
            if st.button("💾 Télécharger le résultat"):
                # Convertir en bytes pour le téléchargement
                import io
                buf = io.BytesIO()
                st.session_state.final_image.save(buf, format="PNG")
                st.download_button(
                    label="Télécharger l'image finale",
                    data=buf.getvalue(),
                    file_name="staging_virtuel_resultat.png",
                    mime="image/png"
                )
    
    elif not obj_file and st.session_state.get('analysis_done'):
        st.image(env_pil)
        st.info("Veuillez uploader une image d'objet dans la barre de gauche pour continuer.")

    elif env_file and not st.session_state.get('analysis_done'):
        st.image(env_pil)
        st.info("Cliquez sur 'Analyser la scène' dans la barre de gauche pour commencer.")