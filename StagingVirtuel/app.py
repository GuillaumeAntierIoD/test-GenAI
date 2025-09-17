import streamlit as st
from PIL import Image, ImageOps
import requests
import numpy as np
import sys
import os
from dotenv import load_dotenv
import io
import torch
from rembg import remove
import base64
from streamlit_drawable_canvas import st_canvas
import time

# --- Configuration et Chargement des Clés API ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Erreur : La clé API 'GEMINI_API_KEY' n'a pas été trouvée...")
    st.stop()

# URL de l'API et constantes
FURNITURE_API_URL = "https://furniture-api.fly.dev/v1/products"
FURNITURE_PER_PAGE = 10
ALLOWED_CATEGORIES = [
  'sofa', 'chair', 'stool', 'table', 'desk', 'kitchen',
  'vanitory', 'matress', 'wardrove', 'tv table', 'garden'
]

# Ajout du chemin vers les modules personnalisés
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from core.placement import placer_objet_drag_and_drop
from core.harmonization import load_captioning_model, generate_caption, generate_adaptive_prompt, harmonize_image_with_gemini

# --- NOUVELLE APPROCHE : Fonctions de rappel (Callbacks) ---

def reset_app_state():
    """Réinitialise toute l'application."""
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]

def go_to_page(page_increment):
    """Change de page dans la bibliothèque."""
    st.session_state.furniture_page += page_increment

def set_category():
    """Met à jour la catégorie et réinitialise la page."""
    if st.session_state.category_select != st.session_state.current_category:
        st.session_state.current_category = st.session_state.category_select
        st.session_state.furniture_page = 0

def add_furniture_to_canvas(furniture_obj):
    """Ajoute un meuble à la liste des objets du canevas."""
    with st.spinner(f"Préparation de '{furniture_obj['name']}'..."):
        image = process_object_image(furniture_obj['image_path'])
        if image:
            unique_key = f"{furniture_obj['id']}_{int(time.time())}"
            st.session_state.canvas_objects.append({
                "key": unique_key, "name": furniture_obj['name'], "image": image,
            })
            if 'composite_image' in st.session_state:
                del st.session_state['composite_image']

def remove_furniture_from_canvas(key_to_remove):
    """Supprime un meuble de la liste."""
    st.session_state.canvas_objects = [
        obj for obj in st.session_state.canvas_objects if obj['key'] != key_to_remove
    ]
    if 'composite_image' in st.session_state:
        del st.session_state['composite_image']

# --- Fonctions Utilitaires (inchangées) ---

@st.cache_data
def fetch_furniture_data(page=0, category=None):
    limit = FURNITURE_PER_PAGE
    offset = page * limit
    params = {"limit": limit, "offset": offset}
    if category and category != "Toutes": params["category"] = category
    try:
        response = requests.get(FURNITURE_API_URL, params=params)
        response.raise_for_status()
        data = response.json().get("data", [])
        return data if isinstance(data, list) else []
    except (requests.exceptions.RequestException, ValueError):
        return []

@st.cache_data
def process_object_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        img_no_bg = remove(response.content)
        image = Image.open(io.BytesIO(img_no_bg))
        bbox = image.getbbox()
        return image.crop(bbox) if bbox else image
    return None

def get_model(model_name, load_function):
    if model_name not in st.session_state.get('models', {}):
        if 'models' not in st.session_state: st.session_state.models = {}
        with st.spinner(f"Chargement du modèle {model_name}..."):
            st.session_state.models[model_name] = load_function()
    return st.session_state.models[model_name]

def pil_to_base64(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


# --- Interface Streamlit ---
st.set_page_config(layout="wide", page_title="Projet de Staging Virtuel")

# Initialisation des états de session
if 'furniture_page' not in st.session_state: st.session_state.furniture_page = 0
if 'current_category' not in st.session_state: st.session_state.current_category = "Toutes"
if 'canvas_objects' not in st.session_state: st.session_state.canvas_objects = []

# Entête
c1, c2 = st.columns([1, 5]); c1.image("IoD_solutions_logo_500px-3.png", width=150); c2.title("Démonstrateur Staging Virtuel")

# --- Barre Latérale (Sidebar) ---
st.sidebar.header("Commandes")
st.sidebar.button("Recommencer depuis le début 🔄", on_click=reset_app_state, use_container_width=True, type="primary")
env_file = st.sidebar.file_uploader("1. Choisissez l'image de l'environnement", type=["jpg", "png"])
st.sidebar.divider()

st.sidebar.subheader("2. Bibliothèque de meubles 🛋️")
st.sidebar.selectbox("Filtrer par catégorie", ["Toutes"] + ALLOWED_CATEGORIES, key="category_select", on_change=set_category)

c1, c2, c3 = st.sidebar.columns([2, 1, 2])
c1.button("⬅️ Précédent", on_click=go_to_page, args=(-1,), use_container_width=True, disabled=st.session_state.furniture_page == 0)
c3.button("Suivant ➡️", on_click=go_to_page, args=(1,), use_container_width=True)
c2.write(f"Page {st.session_state.furniture_page + 1}")

furniture_items = fetch_furniture_data(st.session_state.furniture_page, st.session_state.current_category)
if furniture_items:
    with st.sidebar.container():
        for item in furniture_items:
            c1, c2 = st.columns([1, 2])
            c1.image(item['image_path'], use_column_width='always')
            c2.write(f"**{item['name']}**")
            c2.button("Ajouter +", key=item['id'], on_click=add_furniture_to_canvas, args=(item,), use_container_width=True)
else:
    st.sidebar.warning("Aucun meuble trouvé.")

st.sidebar.divider()
st.sidebar.subheader("Objets sur la scène")
if st.session_state.canvas_objects:
    for obj in st.session_state.canvas_objects:
        c1, c2 = st.sidebar.columns([3, 1])
        c1.text(obj['name'])
        c2.button("❌", key=f"del_{obj['key']}", on_click=remove_furniture_from_canvas, args=(obj['key'],), help="Supprimer")
else:
    st.sidebar.info("Aucun objet ajouté.")


# --- Affichage principal ---
if not env_file:
    st.info("👋 Bienvenue ! Veuillez commencer par uploader une image d'environnement.")
elif not st.session_state.canvas_objects:
    st.image(Image.open(env_file), caption="Aperçu de l'environnement")
    st.info("🖼️ Maintenant, ajoutez un ou plusieurs meubles depuis la bibliothèque.")
else:
    env_pil = Image.open(env_file)
    if 'composite_image' not in st.session_state:
        st.subheader("Placez vos objets sur la scène")
        initial_drawing = {"objects": [{"type": "image", "left": 100+i*30, "top": 100+i*30, "width": o['image'].width, "height": o['image'].height, "src": pil_to_base64(o['image'])} for i, o in enumerate(st.session_state.canvas_objects)]}
        
        # --- CORRECTION : Utilisation de l'argument nommé 'background_image' ---
        canvas_result = st_canvas(
            background_image=env_pil, 
            height=env_pil.height, 
            width=env_pil.width, 
            initial_drawing=initial_drawing, 
            drawing_mode="transform", 
            key="canvas"
        )
        
        if st.button("✅ Valider le placement"):
            if canvas_result.json_data and canvas_result.json_data["objects"]:
                with st.spinner("Composition de l'image..."):
                    final_comp = env_pil.copy()
                    for i, p in enumerate(canvas_result.json_data["objects"]):
                        final_comp = placer_objet_drag_and_drop(final_comp, st.session_state.canvas_objects[i]['image'], p)
                    st.session_state.composite_image = final_comp
                    st.rerun()
    else:
        st.subheader("Comparaison Avant / Après Harmonisation")
        c1, c2 = st.columns(2)
        c1.image(st.session_state.composite_image, caption="Avant harmonisation", use_column_width=True)
        
        with c2:
            if 'final_image' in st.session_state:
                st.image(st.session_state.final_image, caption="Après harmonisation", use_column_width=True)
            else:
                st.info("Lancez l'harmonisation pour voir le résultat.")
        
        st.divider()
        if st.button("✏️ Modifier le placement"):
            del st.session_state.composite_image
            if 'final_image' in st.session_state: del st.session_state.final_image
            st.rerun()
        if 'final_image' in st.session_state:
            buf = io.BytesIO(); st.session_state.final_image.save(buf, format="PNG")
            st.download_button("💾 Télécharger", buf.getvalue(), "resultat.png", "image/png")
        
        if 'final_image' not in st.session_state and 'run_harmonization' not in st.session_state:
            st.sidebar.divider()
            st.sidebar.subheader("Étape 3 : Harmonisation (IA)")
            if st.sidebar.button("🚀 Lancer l'harmonisation", use_container_width=True):
                st.session_state.run_harmonization = True
                st.rerun()

    if st.session_state.get('run_harmonization'):
        del st.session_state.run_harmonization

        with st.spinner("Harmonisation avec l'API Gemini en cours..."):
            st.sidebar.info("Génération des descriptions...")

            prompt = (
                "You are a photorealistic image harmonization expert. "
                "Your task is to integrate the object(s) from the provided image into the background scene. "
                "You MUST NOT change the shape, style, or material of the original object(s). "
                "Your ONLY job is to adjust the lighting, color temperature, and reflections on the object "
                "to perfectly match the environment. "
                "Most importantly, you MUST add realistic, soft contact shadows on the floor beneath the object(s) "
                "to make it look grounded. The final output must be only the complete, harmonized image."
            )

            st.sidebar.info("Appel à l'API Gemini...")

            final_image = harmonize_image_with_gemini(st.session_state.composite_image, prompt, GEMINI_API_KEY)

            if final_image:
                st.session_state.final_image = final_image
                st.success("Harmonisation terminée !")
                time.sleep(2)
                st.rerun()

            else:
                st.error("L'harmonisation a échoué.")
                if st.button("Réessayer l'harmonisation"):
                    st.session_state.run_harmonization = True
                    st.rerun()
            