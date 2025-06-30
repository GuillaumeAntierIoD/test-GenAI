import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import torch
import sys

sys.path.append(os.path.abspath("MiDaS"))

# Import des utilitaires SAM
from src.sam_utils import load_sam_model, process_image_with_sam
from src.midas_utils import load_midas_model, estimate_depth
from src.opencv_utils import resize_object, insert_object

# ======================= Configuration =======================
SAM_PATH = "models/sam_vit_b_01ec64.pth"
MIDAS_PATH = "models/dpt_large_384.pt"
DEVICE = "cpu"
POINTS_PER_SIDE = 16

# ======================= Chargement du modèle (mise en cache) =======================
@st.cache_resource
def get_sam_model():
    """Charge le modèle SAM avec mise en cache Streamlit"""
    if not os.path.exists(SAM_PATH):
        st.error(f"Modèle SAM non trouvé : {SAM_PATH}")
        st.stop()
    
    with st.spinner("Chargement du modèle SAM..."):
        sam_model = load_sam_model(SAM_PATH, DEVICE)
    
    return sam_model

@st.cache_resource
def get_midas_model():
    """Charge le modèle MiDaS avec mise en cache Streamlit"""
    try:
        midas_model, midas_transform = load_midas_model(DEVICE)
        return midas_model, midas_transform
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    
# ======================= Interface principale =======================

st.title("Mise en situation de produit - Démo")

st.markdown("""
### Guide utilisateur
1. Téléchargez une photo de votre pièce.
2. Téléchargez la photo du meuble ou fenêtre.
3. Cliquez sur "Placer l'objet" et patientez.
4. Découvrez le rendu final avec l'objet intégré.
""")

# Upload des images
env_img = st.file_uploader("Uploader la photo d'environnement (ex: salon)", type=["png", "jpg", "jpeg"])
obj_img = st.file_uploader("Uploader la photo du meuble/fenêtre", type=["png", "jpg", "jpeg"])

if env_img and obj_img:
    env_img_pil = Image.open(env_img)
    obj_img_pil = Image.open(obj_img)

    # Affichage des images uploadées
    col1, col2 = st.columns(2)
    with col1:
        st.image(env_img_pil, caption="Photo environnement", use_container_width=True)
    with col2:
        st.image(obj_img_pil, caption="Image meuble/fenêtre", use_container_width=True)

    if st.button("Placer l'objet"):
    # Chargement des modèles
        sam_model = get_sam_model()
        midas_model, midas_transform = get_midas_model()

         # ================= OBJET : Masque + Profondeur ===================
        st.subheader("Analyse du meuble (masque + profondeur)")

        # Étape 1 : traitement SAM une seule fois et stockage dans session_state
        if "obj_masks" not in st.session_state:
            with st.spinner("Segmentation SAM sur l'objet..."):
                obj_img_pil = obj_img_pil.convert("RGBA")
                background = Image.new("RGBA", obj_img_pil.size, (200, 200, 200, 255))
                obj_img_with_bg = Image.alpha_composite(background, obj_img_pil).convert("RGB")

                obj_np, obj_masks, obj_mask_overlay = process_image_with_sam(
                    obj_img_with_bg, sam_model, POINTS_PER_SIDE
                )

                # Stockage
                st.session_state.obj_np = obj_np
                st.session_state.obj_masks = obj_masks
                st.session_state.obj_img_with_bg = obj_img_with_bg
                st.session_state.obj_img_pil = obj_img_pil
                st.session_state.obj_mask_overlay = obj_mask_overlay

        st.image(st.session_state.obj_mask_overlay, caption="Masque SAM de l'objet")

        # Étape 2 : Sélection des masques dans un formulaire
        st.subheader("Sélection des masques pour l'objet (canapé)")

        selected_mask_indices = []
        with st.form("mask_selection_form"):
            cols = st.columns(3)
            for i, mask_dict in enumerate(st.session_state.obj_masks):
                mask = mask_dict["segmentation"]
                col = cols[i % 3]
                with col:
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(np.array(st.session_state.obj_img_pil))
                    ax.imshow(mask, alpha=0.5, cmap='jet')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
                    if st.checkbox(f"Sélectionner le masque #{i}", key=f"mask_{i}"):
                        selected_mask_indices.append(i)

            submitted = st.form_submit_button("Valider les masques")

        # Étape 3 : Carte de profondeur uniquement si validation du formulaire
        if submitted and selected_mask_indices:
            combined_mask = np.zeros_like(st.session_state.obj_masks[0]["segmentation"], dtype=bool)
            for idx in selected_mask_indices:
                combined_mask = combined_mask | st.session_state.obj_masks[idx]["segmentation"]

            st.image(combined_mask.astype(np.uint8)*255, caption="Masque combiné sélectionné")

            # Génération de la depth map après validation des masques
            with st.spinner("Profondeur du meuble via MiDaS..."):
                input_sample = {"image": np.array(st.session_state.obj_img_with_bg)}
                input_tensor = midas_transform(input_sample)["image"]
                if isinstance(input_tensor, np.ndarray):
                    input_tensor = torch.from_numpy(input_tensor)

                input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    prediction = midas_model(input_tensor)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=st.session_state.obj_np.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze().cpu().numpy()

                obj_mask = combined_mask.astype(np.uint8)
                obj_depth = prediction * obj_mask

                depth_vis_obj = (obj_depth - obj_depth.min()) / (obj_depth.max() - obj_depth.min() + 1e-8)
                st.image(depth_vis_obj, caption="Carte de profondeur de l'objet (filtrée)")
        elif submitted:
            st.warning("Veuillez sélectionner au moins un masque pour l'objet.")



        with st.spinner("Traitement avec SAM en cours..."):
            start_time = time.time()
     
            # Traitement SAM
            env_np, env_masks, env_mask_overlay = process_image_with_sam(
                env_img_pil, sam_model, POINTS_PER_SIDE)
            
            processing_time = time.time() - start_time
            
            # Résultats segmentation
            st.success(f"Segmentation terminée en {processing_time:.2f} secondes")
            st.info(f"Nombre de segments détectés : {len(env_masks)}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Image originale")
                st.image(env_img_pil, use_container_width=True)
            with col2:
                st.subheader("Segmentation SAM")
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(env_np)
                if env_mask_overlay is not None:
                    ax.imshow(env_mask_overlay, alpha=0.6)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()


        # 🔄 Étape MiDaS : estimation de la profondeur
        with st.spinner("Analyse de profondeur avec MiDaS..."):
            try:
                midas_model, midas_transform = get_midas_model()

                # Préparation de l'image
                input_image = np.array(env_img_pil.convert("RGB"))
                input_sample = {"image": input_image}
                input_tensor = midas_transform(input_sample)["image"]

                # Convertir en tensor PyTorch si besoin
                if isinstance(input_tensor, np.ndarray):
                    input_tensor = torch.from_numpy(input_tensor)

                input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

                # Estimation de la profondeur
                with torch.no_grad():
                    prediction = midas_model(input_tensor)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=input_image.shape[:2],  # (H, W)
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                depth_map = prediction.cpu().numpy()

                # Visualisation
                depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)  # Normaliser
                depth_vis = (depth_vis * 255).astype("uint8")

                st.success("Profondeur estimée avec MiDaS")

                st.subheader("Carte de profondeur")
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(depth_map, cmap='plasma')  # colormap plasma
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # barre de couleur

                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Erreur lors de l'analyse MiDaS : {e}")
        
        
        st.subheader("📌 Placement de l'objet avec OpenCV")

        try:
            selected_mask = env_masks[mask_index]
            
            # Centroid du masque sélectionné
            ys, xs = np.where(selected_mask)
            if len(xs) == 0 or len(ys) == 0:
                raise ValueError("Le segment sélectionné est vide.")
            
            x_center = int(xs.mean())
            y_center = int(ys.mean())

            depth_value = depth_map[y_center, x_center]
            norm_depth = (depth_value - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            scale = max(0.2, 1.5 - norm_depth * 1.2)

            st.write(f"🔍 Profondeur à (x={x_center}, y={y_center}) : `{depth_value:.2f}`")
            st.write(f"📐 Facteur d'échelle calculé : `{scale:.2f}`")

            # Préparer l'objet en tenant compte du canal alpha s’il existe
            obj_rgba = obj_img_pil.convert("RGBA")
            obj_np = np.array(obj_rgba)

            # Extraire alpha comme masque
            obj_rgb = obj_np[:, :, :3]
            obj_alpha = obj_np[:, :, 3]
            obj_mask = (obj_alpha > 0).astype(np.uint8) * 255

            obj_resized = resize_object(obj_rgb, scale=scale)
            mask_resized = resize_object(obj_mask, scale=scale)

            result_img = insert_object(env_np, obj_resized, mask_resized, position=(x_center, y_center))

            st.success("Objet inséré avec succès 🧩")
            st.image(result_img, caption="✅ Résultat avec l'objet placé", use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors du placement de l'objet : {e}")





        # Placeholder suite du pipeline
        st.markdown("---")
        st.subheader("Prochaines étapes du pipeline :")
        st.write("🎨 Génération finale avec Stable Diffusion")
        st.image(env_img_pil, caption="Résultat final (placeholder - sera remplacé par le pipeline complet)")


else:
    if not env_img:
        st.info("👆 Veuillez uploader une photo d'environnement")
    if not obj_img:
        st.info("👆 Veuillez uploader une photo du meuble/fenêtre")