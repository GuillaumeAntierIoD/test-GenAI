import torch
from diffusers import AutoPipelineForInpainting, EulerDiscreteScheduler
from PIL import Image
import numpy as np
import google.generativeai as genai
from google.api_core import exceptions
import io
import streamlit as st

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

from PIL import Image

def make_image_square(pil_image: Image.Image, fill_color=(255, 255, 255, 255)) -> Image.Image:
    """
    Prend une image PIL et la transforme en une image carrée en ajoutant des bandes
    de couleur (padding). L'image originale est centrée.

    Args:
        pil_image: L'image d'entrée au format PIL.
        fill_color: La couleur de fond pour les bandes (blanc par défaut).

    Returns:
        Une nouvelle image PIL carrée.
    """
    pil_image = pil_image.convert('RGBA')
    
    width, height = pil_image.size
    max_dim = max(width, height)

    new_image = Image.new('RGBA', (max_dim, max_dim), fill_color)

    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2

    new_image.paste(pil_image, (paste_x, paste_y), pil_image)

    return new_image

def harmonize_image_with_gemini(composite_image: Image.Image, prompt: str, api_key: str):
    """
    Appelle l'API Gemini (Gemini 2.5 Flash Image) pour réaliser l'harmonisation.
    """
    try:
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel('gemini-2.5-flash-image-preview')

        response = model.generate_content([prompt, composite_image])

        if not response.candidates:
            print("Erreur : La réponse de l'API ne contient aucun candidat.")
            return None

        final_image = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_data = part.inline_data.data
                final_image = Image.open(io.BytesIO(image_data))
                break 

        if final_image:
            return final_image
        else:
            try:
                error_text = response.text
                print(f"DEBUG: Gemini a répondu avec du texte uniquement : '{error_text}'")
                st.error(f"L'IA a retourné un message mais pas d'image : {error_text}")
            except Exception:
                st.error("L'API a retourné une réponse non valide qui ne contient pas d'image.")
            
            return None

    except exceptions.ResourceExhausted as e:
        print(f"Quota dépassé : {e}")
        st.error("Le service d'harmonisation est surchargé. Vous avez dépassé le quota de requêtes. Veuillez réessayer dans une minute.")
        return None
    
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors de l'appel à l'API Gemini : {e}")
        st.error("Une erreur inattendue est survenue avec le service d'harmonisation.")
        return None
    
def replace_furniture_with_gemini(
    original_image: Image.Image,
    new_furniture_image: Image.Image,
    object_to_replace_text: str, 
    new_furniture_name: str, 
    api_key: str
):
    """
    Appelle l'API Gemini pour remplacer un meuble dans une image par un autre.
    Cette version est robuste et cherche l'image dans toutes les parties de la réponse.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-image-preview')

        prompt = (
            f"En tant qu'expert en design d'intérieur et en retouche photo photoréaliste :\n\n"
            f"**Tâche :** À partir de la première image (la pièce), **supprime complètement** '{object_to_replace_text}' "
            f"et **insère** à la place le nouveau meuble présenté dans la seconde image (un '{new_furniture_name}').\n\n"
            f"**Exigences Clés :**\n"
            f"1.  **Géométrie et Forme :** Le nouveau meuble doit **adopter sa propre forme et géométrie unique**, comme visible dans sa photo de référence, sans s'appuyer sur la forme de l'objet remplacé. Ajuste son échelle, sa perspective, son éclairage et ses couleurs pour correspondre à l'environnement de la pièce.\n" # <-- CLÉ
            f"2.  **Ombres :** Génère des ombres réalistes sous le nouveau meuble, cohérentes avec la source de lumière existante de la pièce, reflétant sa nouvelle forme.\n"
            f"3.  **Préservation :** Ne modifie absolument rien d'autre dans l'image de la pièce.\n"
            f"4.  **Sortie :** Produis une unique image photoréaliste du résultat.\n"
            f"**ACTION FINALE : Ne réponds PAS par du texte. Le seul et unique résultat doit être le fichier image final.**"
        )
        response = model.generate_content([
            prompt,
            original_image.convert("RGB"), 
            new_furniture_image.convert("RGB")
        ])

        if not response.candidates:
            st.error("L'API n'a retourné aucun candidat valide.")
            return None

        final_image = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_data = part.inline_data.data
                final_image = Image.open(io.BytesIO(image_data))
                break 

        if final_image:
            return final_image
        else:
            try:
                error_text = response.text
                print(f"DEBUG: Gemini a répondu avec du texte uniquement : '{error_text}'")
                st.error(f"L'IA a retourné un message mais pas d'image : {error_text}")
            except Exception:
                st.error("L'API a retourné une réponse non valide qui ne contient pas d'image.")
            
            return None

    except exceptions.ResourceExhausted as e:
        st.error(f"Quota dépassé : {e}. Veuillez réessayer dans une minute.")
        return None
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue : {e}")
        return None