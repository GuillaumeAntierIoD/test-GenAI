import torch
from diffusers import AutoPipelineForInpainting, EulerDiscreteScheduler
from PIL import Image
import numpy as np
import google.generativeai as genai
from google.api_core import exceptions
import io
import os
import cv2
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Callable, Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

def load_captioning_model():
    """Charge le modèle BLIP et son processeur pour la description d'images."""
    print("Chargement du modèle de description d'image (BLIP)...")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", torch_dtype=DTYPE
        ).to(DEVICE)
        print("Modèle BLIP chargé avec succès.")
        return processor, model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle BLIP : {e}")
        return None, None
    

def generate_adaptive_prompt(obj_caption: str, env_caption: str) -> str:
    """
    Génère un prompt descriptif pour l'API Gemini.
    """
    prompt = (
        f"En utilisant l'image fournie qui montre un '{obj_caption}' placé dans un '{env_caption}', "
        f"modifie l'image pour que le '{obj_caption}' s'intègre parfaitement. "
        "Ajuste son éclairage, ses ombres, sa balance des couleurs et sa texture pour qu'ils correspondent au style et à l'ambiance de la pièce. "
        "Le résultat doit être photoréaliste, comme si l'objet avait toujours été là. Ne modifie rien d'autre dans la pièce."
    )
    return prompt

def generate_caption(image_pil, processor, model):
    """Génère une description pour une image donnée."""
    if processor is None or model is None:
        return ""
    
    text = "a photography of"
    inputs = processor(images=image_pil.convert("RGB"), text=text, return_tensors="pt").to(DEVICE, DTYPE)
    
    generated_ids = model.generate(**inputs, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return generated_caption

def harmonize_image_with_gemini(composite_image: Image.Image, prompt: str, api_key: str):
    """
    Appelle l'API Gemini (Gemini 2.5 Flash Image) pour réaliser l'harmonisation.
    """
    try:
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel('gemini-2.5-flash-image-preview')

        response = model.generate_content([prompt, composite_image])

        print(response)

        if not response.candidates:
            print("Erreur : La réponse de l'API ne contient aucun candidat.")
            return None

        image_part = response.candidates[0].content.parts[0]
        if image_part.inline_data:
            image_data = image_part.inline_data.data
            final_image = Image.open(io.BytesIO(image_data))
            return final_image
        else:
            print("Erreur : L'API n'a pas retourné de données d'image.")
            return None

    except exceptions.ResourceExhausted as e:
        print(f"Quota dépassé : {e}")
        st.error("Le service d'harmonisation est surchargé. Vous avez dépassé le quota de requêtes. Veuillez réessayer dans une minute.")
        return None
    
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors de l'appel à l'API Gemini : {e}")
        st.error("Une erreur inattendue est survenue avec le service d'harmonisation.")
        return None