import streamlit as st
from PIL import Image
import numpy as np
import sys
import os
import io
import torch
from rembg import remove
import base64
from streamlit_drawable_canvas import st_canvas
import time

# --- Configuration des chemins et imports ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from core.placement import placer_objet_drag_and_drop 
from core.harmonization import load_sd_inpainting_model, create_inpainting_mask, harmonize_image, load_captioning_model, generate_caption, generate_adaptive_prompt

# --- Interface Streamlit (inchangée) ---
st.set_page_config(layout="wide", page_title="Projet de Staging Virtuel")
col1, col2 = st.columns([1, 5]) 
with col1:
    st.image("IoD_solutions_logo_500px-3.png")
with col2:
    st.title("Démonstrateur Staging Virtuel")

# --- Initialisation des modèles (inchangée) ---
if 'models' not in st.session_state:
    st.session_state.models = {}
def get_model(model_name, load_function):
    if model_name not in st.session_state.models:
        with st.spinner(f"Chargement du modèle {model_name}..."):
            st.session_state.models[model_name] = load_function()
    return st.session_state.models[model_name]

def pil_to_base64(image: Image.Image) -> str:
    """
    Convertit une image PIL en une chaîne de caractères base64 pour l'affichage
    dans le canvas.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# --- UI (Barre latérale, inchangée) ---
st.sidebar.header("Commandes")
env_file = st.sidebar.file_uploader("1. Image de l'environnement", type=["jpg", "png"])
obj_file = st.sidebar.file_uploader("2. Image de l'objet", type=["png", "jpg"])


# --- Logique principale ---
if not env_file or not obj_file:
    st.info("Bienvenue ! Veuillez uploader une image d'environnement et d'objet pour commencer.")
else:
    env_pil = Image.open(env_file)

    if st.session_state.get('obj_file_name') != obj_file.name:
        st.session_state.obj_file_name = obj_file.name
        original_obj_pil = Image.open(obj_file)
        with st.spinner("Suppression du fond de l'objet..."):
            img_no_bg = remove(original_obj_pil)
            bbox = img_no_bg.getbbox()
            st.session_state.obj_no_bg = img_no_bg.crop(bbox) if bbox else img_no_bg
        if 'composite_image' in st.session_state:
            del st.session_state['composite_image']
        st.rerun()

    obj_to_display = st.session_state.get('obj_no_bg')
    if obj_to_display:
        st.sidebar.divider()
        st.sidebar.subheader("Objet à placer")
        st.sidebar.image(obj_to_display)

    st.subheader("Placez votre objet")
    st.info("Utilisez les poignées pour déplacer, redimensionner et faire pivoter l'objet.")

    obj_base64_src = pil_to_base64(obj_to_display)

    initial_drawing = {
            "objects": [{
                "type": "image",
                "left": 100,
                "top": 100,
                "width": obj_to_display.width * 0.5, 
                "height": obj_to_display.height * 0.5,
                "src": obj_base64_src
            }]
        }

    canvas_result = st_canvas(
            background_image=env_pil,
            height=env_pil.height,
            width=env_pil.width,
            initial_drawing=initial_drawing,
            drawing_mode="transform",
            update_streamlit=True,
            key="canvas_dd"
        )

    if st.button("✅ Valider le placement"):
        if canvas_result.json_data and canvas_result.json_data.get("objects"):
            with st.spinner("Composition de l'image..."):
                placement_data = canvas_result.json_data["objects"][0]
                
                composite_image, object_mask = placer_objet_drag_and_drop(
                    env_pil, obj_to_display, placement_data, return_mask=True
                )
                
                st.session_state.composite_image = composite_image
                st.session_state.object_mask = object_mask
                st.rerun()
        else:
            st.warning("Veuillez d'abord placer l'objet sur l'image.")

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
                print(auto_prompt)
                if 'captioning' in st.session_state.models:
                        del st.session_state.models['captioning']
                if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                progress_bar = st.sidebar.progress(0)
                progress_text = st.sidebar.empty()

                base_inference_steps = 10 
                strength_value = 0.45
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