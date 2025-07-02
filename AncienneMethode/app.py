import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import torch
import sys

sys.path.append(os.path.abspath("MiDaS"))

# Import des utilitaires existants
from src.sam_utils import load_sam_model, process_image_with_sam
from src.midas_utils import load_midas_model, estimate_depth
from src.opencv_utils import resize_object, insert_object
from src.click_placement import interactive_placement_section, place_object_at_coordinates
from src.homography_utils import (
    find_homography_from_plane, 
    calculate_optimal_scale_from_depth,
    detect_placement_surface,
    create_shadow_mask,
    apply_perspective_transform
)

from src.stable_diffusion_utils import (
    StableDiffusionProcessor,
    enhance_with_img2img,
    enhance_with_inpainting,
    create_inpainting_mask_from_object,
    detect_lighting_conditions,
    apply_color_harmony,
    post_process_integration
)

# ======================= Configuration =======================
SAM_PATH = "models/sam_vit_b_01ec64.pth"
MIDAS_PATH = "models/dpt_large_384.pt"
DEVICE = "cpu"
POINTS_PER_SIDE = 16

ENABLE_SD = True  # Set to False if you don't have diffusers installed
SD_DEVICE = "cpu"  # Change to "cuda" if you have GPU

# ======================= Initialisation des variables de session =======================
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'masks_selected' not in st.session_state:
    st.session_state.masks_selected = False
if 'obj_processed' not in st.session_state:
    st.session_state.obj_processed = False
if 'env_processed' not in st.session_state:
    st.session_state.env_processed = False

# ======================= Chargement des modèles (mise en cache) =======================
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

@st.cache_resource
def get_sd_processor():
    """Charge le processeur Stable Diffusion avec mise en cache"""
    if not ENABLE_SD:
        return None
    try:
        return StableDiffusionProcessor(device=SD_DEVICE)
    except ImportError:
        st.warning("Stable Diffusion non disponible. Installez diffusers pour activer cette fonctionnalité.")
        return None

# ======================= Fonctions utilitaires =======================
def apply_mask_to_image(image, mask):
    """
    Applique un masque à une image pour créer une image RGBA
    """
    if len(image.shape) == 3 and image.shape[2] == 3:  # RGB
        # Ajouter le canal alpha basé sur le masque
        alpha = (mask * 255).astype(np.uint8)
        rgba = np.dstack([image, alpha])
        return rgba
    elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
        # Modifier le canal alpha existant
        image[:, :, 3] = (mask * 255).astype(np.uint8)
        return image
    else:
        raise ValueError(f"Format d'image non supporté: {image.shape}")
    
def manual_surface_selection_interface(env_np, env_masks):
    """Interface pour la sélection manuelle de la zone de placement"""
    st.subheader("🎯 Sélection de la zone de placement")
    
    # Options de sélection
    selection_method = st.radio(
        "Méthode de sélection de la zone :",
        ["Automatique", "Manuel par masques", "Manuel par coordonnées"],
        index=0
    )
    
    if selection_method == "Automatique":
        return None, "auto"
    
    elif selection_method == "Manuel par masques":
        st.write("Sélectionnez la zone où vous voulez placer votre meuble :")
        
        # Afficher les masques disponibles
        cols = st.columns(3)
        selected_surface_idx = None
        
        for i, mask_dict in enumerate(env_masks):
            mask = mask_dict["segmentation"]
            col = cols[i % 3]
            
            with col:
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(env_np)
                ax.imshow(mask, alpha=0.5, cmap='Reds')
                ax.axis('off')
                ax.set_title(f"Zone #{i}\n(aire: {mask_dict['area']})")
                st.pyplot(fig)
                plt.close(fig)
                
                if st.button(f"Sélectionner zone #{i}", key=f"surface_{i}"):
                    selected_surface_idx = i
                    st.session_state.selected_surface_mask = mask
                    st.session_state.surface_selected = True
                    st.success(f"Zone #{i} sélectionnée !")
                    st.rerun()
        
        return selected_surface_idx, "manual_mask"
    
    else:  # Manuel par coordonnées
        st.write("Spécifiez manuellement les coordonnées de placement :")
        
        col1, col2 = st.columns(2)
        with col1:
            x_manual = st.slider("Position X", 0, env_np.shape[1], env_np.shape[1]//2, key="manual_x")
            y_manual = st.slider("Position Y", 0, env_np.shape[0], env_np.shape[0]//2, key="manual_y")
        
        with col2:
            zone_width = st.slider("Largeur de la zone", 50, 1000, 150, key="zone_width")
            zone_height = st.slider("Hauteur de la zone", 50, 1000, 100, key="zone_height")
        
        # Créer un masque rectangulaire basé sur les coordonnées
        manual_mask = np.zeros(env_np.shape[:2], dtype=bool)
        
        x_start = max(0, x_manual - zone_width // 2)
        x_end = min(env_np.shape[1], x_manual + zone_width // 2)
        y_start = max(0, y_manual - zone_height // 2)
        y_end = min(env_np.shape[0], y_manual + zone_height // 2)
        
        manual_mask[y_start:y_end, x_start:x_end] = True
        
        # Prévisualisation de la zone sélectionnée
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(env_np)
        ax.imshow(manual_mask, alpha=0.5, cmap='Reds')
        ax.set_title("Zone de placement manuelle")
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)
        
        if st.button("Confirmer cette zone"):
            st.session_state.selected_surface_mask = manual_mask
            st.session_state.surface_selected = True
            st.success("Zone manuelle confirmée !")
            st.rerun()
        
        return manual_mask, "manual_coords"

# ======================= Interface principale =======================

st.title("Mise en situation de produit - Démo")

st.markdown("""
### Guide utilisateur
1. Téléchargez une photo de votre pièce.
2. Téléchargez la photo du meuble ou fenêtre.
3. Cliquez sur "Traiter les images" et patientez.
4. Sélectionnez les masques du meuble et validez.
5. Le placement se fera automatiquement après validation.
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

    # Bouton pour commencer le traitement
    if st.button("Traiter les images") and not st.session_state.analysis_done:
        sam_model = get_sam_model()
        midas_model, midas_transform = get_midas_model()
        
        st.session_state.analysis_done = True
        st.rerun()

    # ================= TRAITEMENT DES IMAGES ===================
    if st.session_state.analysis_done:
        
        # Chargement des modèles
        if 'sam_model' not in st.session_state:
            st.session_state.sam_model = get_sam_model()
            st.session_state.midas_model, st.session_state.midas_transform = get_midas_model()

        # ================= OBJET : Masque + Profondeur ===================
        if not st.session_state.obj_processed:
            st.subheader("Analyse du meuble (masque + profondeur)")
            
            with st.spinner("Segmentation SAM sur l'objet..."):
                obj_img_pil_processed = obj_img_pil.convert("RGBA")
                background = Image.new("RGBA", obj_img_pil_processed.size, (200, 200, 200, 255))
                obj_img_with_bg = Image.alpha_composite(background, obj_img_pil_processed).convert("RGB")

                obj_np, obj_masks, obj_mask_overlay = process_image_with_sam(
                    obj_img_with_bg, st.session_state.sam_model, POINTS_PER_SIDE
                )

                # Stockage dans session_state
                st.session_state.obj_np = obj_np
                st.session_state.obj_masks = obj_masks
                st.session_state.obj_img_with_bg = obj_img_with_bg
                st.session_state.obj_img_pil = obj_img_pil_processed
                st.session_state.obj_mask_overlay = obj_mask_overlay
                st.session_state.obj_processed = True

        # Affichage des masques de l'objet
        if st.session_state.obj_processed:
            st.image(st.session_state.obj_mask_overlay, caption="Masque SAM de l'objet")

            # ================= SÉLECTION DES MASQUES ===================
            if not st.session_state.masks_selected:
                st.subheader("Sélection des masques pour l'objet")

                with st.form("mask_selection_form"):
                    st.write("Sélectionnez les masques qui correspondent au meuble :")
                    
                    cols = st.columns(3)
                    selected_masks = []
                    
                    for i, mask_dict in enumerate(st.session_state.obj_masks):
                        mask = mask_dict["segmentation"]
                        col = cols[i % 3]
                        with col:
                            fig, ax = plt.subplots(figsize=(3, 3))
                            ax.imshow(np.array(st.session_state.obj_img_pil))
                            ax.imshow(mask, alpha=0.5, cmap='jet')
                            ax.axis('off')
                            ax.set_title(f"Masque #{i}")
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Checkbox pour sélection
                            if st.checkbox(f"Sélectionner #{i}", key=f"mask_{i}"):
                                selected_masks.append(i)

                    # Bouton de validation dans le formulaire
                    submitted = st.form_submit_button("Valider les masques sélectionnés")
                    
                    if submitted:
                        if selected_masks:
                            st.session_state.selected_mask_indices = selected_masks
                            st.session_state.masks_selected = True
                            st.success(f"Masques sélectionnés : {selected_masks}")
                            st.rerun()
                        else:
                            st.error("Veuillez sélectionner au moins un masque.")

            # ================= TRAITEMENT APRÈS SÉLECTION DES MASQUES ===================
            if st.session_state.masks_selected and 'selected_mask_indices' in st.session_state:
                
                # Créer le masque combiné
                if 'combined_mask' not in st.session_state:
                    combined_mask = np.zeros_like(st.session_state.obj_masks[0]["segmentation"], dtype=bool)
                    for idx in st.session_state.selected_mask_indices:
                        combined_mask = combined_mask | st.session_state.obj_masks[idx]["segmentation"]
                    st.session_state.combined_mask = combined_mask

                st.image(st.session_state.combined_mask.astype(np.uint8)*255, caption="Masque combiné sélectionné")

                # Génération de la depth map pour l'objet
                if 'obj_depth' not in st.session_state:
                    with st.spinner("Profondeur du meuble via MiDaS..."):
                        input_sample = {"image": np.array(st.session_state.obj_img_with_bg)}
                        input_tensor = st.session_state.midas_transform(input_sample)["image"]
                        if isinstance(input_tensor, np.ndarray):
                            input_tensor = torch.from_numpy(input_tensor)

                        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

                        with torch.no_grad():
                            prediction = st.session_state.midas_model(input_tensor)
                            prediction = torch.nn.functional.interpolate(
                                prediction.unsqueeze(1),
                                size=st.session_state.obj_np.shape[:2],
                                mode="bicubic",
                                align_corners=False,
                            ).squeeze().cpu().numpy()

                        obj_mask = st.session_state.combined_mask.astype(np.uint8)
                        obj_depth = prediction * obj_mask
                        st.session_state.obj_depth = obj_depth

                        depth_vis_obj = (obj_depth - obj_depth.min()) / (obj_depth.max() - obj_depth.min() + 1e-8)
                    
                    st.image(depth_vis_obj, caption="Carte de profondeur de l'objet (filtrée)")

                # ================= TRAITEMENT DE L'ENVIRONNEMENT ===================
                if not st.session_state.env_processed:
                    with st.spinner("Traitement de l'environnement..."):
                        start_time = time.time()
                 
                        # Traitement SAM de l'environnement
                        env_np, env_masks, env_mask_overlay = process_image_with_sam(
                            env_img_pil, st.session_state.sam_model, POINTS_PER_SIDE)
                        
                        processing_time = time.time() - start_time
                        
                        # Stockage
                        st.session_state.env_np = env_np
                        st.session_state.env_masks = env_masks
                        st.session_state.env_mask_overlay = env_mask_overlay
                        st.session_state.env_processed = True

                        # Résultats segmentation
                        st.success(f"Segmentation terminée en {processing_time:.2f} secondes")
                        st.info(f"Nombre de segments détectés : {len(env_masks)}")

                # Affichage des résultats de l'environnement
                if st.session_state.env_processed:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Image originale")
                        st.image(env_img_pil, use_container_width=True)
                    with col2:
                        st.subheader("Segmentation SAM")
                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.imshow(st.session_state.env_np)
                        if st.session_state.env_mask_overlay is not None:
                            ax.imshow(st.session_state.env_mask_overlay, alpha=0.6)
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()

                    # ================= ESTIMATION DE PROFONDEUR DE L'ENVIRONNEMENT ===================
                    if 'depth_map' not in st.session_state:
                        with st.spinner("Analyse de profondeur avec MiDaS..."):
                            try:
                                # Préparation de l'image
                                input_image = np.array(env_img_pil.convert("RGB"))
                                input_sample = {"image": input_image}
                                input_tensor = st.session_state.midas_transform(input_sample)["image"]

                                # Convertir en tensor PyTorch si besoin
                                if isinstance(input_tensor, np.ndarray):
                                    input_tensor = torch.from_numpy(input_tensor)

                                input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

                                # Estimation de la profondeur
                                with torch.no_grad():
                                    prediction = st.session_state.midas_model(input_tensor)
                                    prediction = torch.nn.functional.interpolate(
                                        prediction.unsqueeze(1),
                                        size=input_image.shape[:2],  # (H, W)
                                        mode="bicubic",
                                        align_corners=False,
                                    ).squeeze()

                                depth_map = prediction.cpu().numpy()
                                st.session_state.depth_map = depth_map

                                # Visualisation
                                depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
                                depth_vis = (depth_vis * 255).astype("uint8")

                                st.success("Profondeur estimée avec MiDaS")

                            except Exception as e:
                                st.error(f"Erreur lors de l'analyse MiDaS : {e}")

                    # Affichage de la carte de profondeur
                    if 'depth_map' in st.session_state:
                        st.subheader("Carte de profondeur")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(st.session_state.depth_map, cmap='plasma')
                        ax.axis('off')
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        st.pyplot(fig)
                        plt.close(fig)

                        # ======================= SÉLECTION DE LA ZONE DE PLACEMENT =======================
                        st.markdown("---")
                        
                        # Interface de sélection de zone
                        surface_result, method = manual_surface_selection_interface(
                            st.session_state.env_np, st.session_state.env_masks
                        )
                        
                        # Traitement selon la méthode choisie
                        surface_mask = None
                        surface_info = None
                        
                        if method == "auto":
                            # Détection automatique comme avant
                            placement_surface = detect_placement_surface(st.session_state.depth_map, st.session_state.env_masks)
                            if placement_surface is not None:
                                surface_mask, surface_info = placement_surface
                                st.success(f"Surface de placement détectée automatiquement (aire: {surface_info['area']} pixels)")
                            else:
                                st.warning("Aucune surface automatique trouvée. Utilisez la sélection manuelle.")
                        
                        elif method == "manual_mask" and st.session_state.surface_selected:
                            surface_mask = st.session_state.selected_surface_mask
                            surface_info = {"area": np.sum(surface_mask)}
                            st.success(f"Surface sélectionnée manuellement (aire: {surface_info['area']} pixels)")
                        
                        elif method == "manual_coords" and st.session_state.surface_selected:
                            surface_mask = st.session_state.selected_surface_mask
                            surface_info = {"area": np.sum(surface_mask)}
                            st.success(f"Zone manuelle définie (aire: {surface_info['area']} pixels)")

                        # ======================= PLACEMENT SI UNE ZONE EST SÉLECTIONNÉE =======================
                        if surface_mask is not None:
                            # Afficher la surface sélectionnée
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.imshow(st.session_state.env_np)
                            ax.imshow(surface_mask, alpha=0.5, cmap='Reds')
                            ax.set_title("Zone de placement sélectionnée")
                            ax.axis('off')
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Interface pour ajustement
                            col1, col2 = st.columns(2)
                            with col1:
                                furniture_type = st.selectbox("Type de meuble", 
                                    ["sofa", "chair", "table", "bookshelf", "lamp", "cabinet"])
                                room_type = st.selectbox("Type de pièce",
                                    ["living room", "bedroom", "dining room", "office", "kitchen"])
                            with col2:
                                add_shadows = st.checkbox("Ajouter des ombres", value=True)
                                manual_scale = st.slider("Échelle manuelle", 0.1, 2.0, 1.0, 0.1)
                            
                            # Calcul automatique de la position dans la zone sélectionnée
                            y_coords, x_coords = np.where(surface_mask)
                            x_center = int(x_coords.mean()) if len(x_coords) > 0 else st.session_state.env_np.shape[1]//2
                            y_center = int(y_coords.mean()) if len(y_coords) > 0 else st.session_state.env_np.shape[0]//2
                            
                            # Calcul de l'échelle optimale basée sur la profondeur
                            optimal_scale = calculate_optimal_scale_from_depth(
                                st.session_state.depth_map, (x_center, y_center)
                            ) * manual_scale
                            
                            st.info(f"Position calculée: ({x_center}, {y_center}), Échelle finale: {optimal_scale:.2f}")
                            
                            # Bouton pour effectuer le placement
                            if st.button("🎯 Effectuer le placement"):
                                try:
                                    # Préparer l'objet avec masque sélectionné
                                    obj_rgba = st.session_state.obj_img_pil.convert("RGBA")
                                    obj_np = np.array(obj_rgba)
                                    obj_rgb = obj_np[:, :, :3]
                                    
                                    # Utiliser le masque combiné pour créer le masque alpha
                                    obj_mask_alpha = st.session_state.combined_mask.astype(np.uint8) * 255
                                    
                                    # Calculer l'homographie si possible
                                    homography = find_homography_from_plane(st.session_state.env_np, st.session_state.depth_map, surface_mask.astype(np.uint8))
                                    
                                    if homography is not None:
                                        st.success("Homographie calculée avec succès")
                                        obj_transformed = apply_perspective_transform(
                                            obj_rgb, homography, (obj_rgb.shape[1], obj_rgb.shape[0])
                                        )
                                        mask_transformed = apply_perspective_transform(
                                            obj_mask_alpha, homography, (obj_mask_alpha.shape[1], obj_mask_alpha.shape[0])
                                        )
                                    else:
                                        st.info("Utilisation du placement simple sans homographie")
                                        obj_transformed = obj_rgb
                                        mask_transformed = obj_mask_alpha
                                    
                                    # Redimensionner selon l'échelle optimale
                                    obj_resized = resize_object(obj_transformed, scale=optimal_scale)
                                    mask_resized = resize_object(mask_transformed, scale=optimal_scale)
                                    
                                    # Créer des ombres si demandé
                                    shadow_mask = None
                                    if add_shadows:
                                        lighting = detect_lighting_conditions(st.session_state.env_np)
                                        st.info(f"Conditions d'éclairage détectées: {lighting}")
                                        
                                        shadow_mask = create_shadow_mask(
                                            mask_resized, 
                                            light_direction=(1, 1),
                                            shadow_intensity=0.3
                                        )
                                    
                                    # Insérer l'objet dans l'environnement
                                    result_img = insert_object(st.session_state.env_np, obj_resized, mask_resized, 
                                                            position=(x_center - obj_resized.shape[1]//2, 
                                                                    y_center - obj_resized.shape[0]//2))
                                    
                                    # Ajouter l'ombre si elle existe
                                    if shadow_mask is not None:
                                        shadow_overlay = np.zeros_like(result_img)
                                        shadow_overlay[:, :] = [20, 20, 20]
                                        
                                        shadow_area = result_img[y_center:y_center+shadow_mask.shape[0], 
                                                            x_center:x_center+shadow_mask.shape[1]]
                                        if shadow_area.shape[:2] == shadow_mask.shape:
                                            shadow_norm = shadow_mask.astype(np.float32) / 255.0
                                            if len(shadow_norm.shape) == 2:
                                                shadow_norm = np.stack([shadow_norm] * 3, axis=2)
                                            
                                            shadow_blended = shadow_area * (1 - shadow_norm * 0.5) + shadow_overlay[:shadow_area.shape[0], :shadow_area.shape[1]] * (shadow_norm * 0.5)
                                            result_img[y_center:y_center+shadow_mask.shape[0], 
                                                    x_center:x_center+shadow_mask.shape[1]] = shadow_blended.astype(np.uint8)
                                    
                                    # Post-traitement
                                    full_obj_mask = np.zeros(result_img.shape[:2], dtype=np.uint8)
                                    y_start = max(0, y_center - obj_resized.shape[0]//2)
                                    y_end = min(result_img.shape[0], y_start + obj_resized.shape[0])
                                    x_start = max(0, x_center - obj_resized.shape[1]//2)
                                    x_end = min(result_img.shape[1], x_start + obj_resized.shape[1])
                                    
                                    mask_h = y_end - y_start
                                    mask_w = x_end - x_start
                                    if mask_h > 0 and mask_w > 0:
                                        full_obj_mask[y_start:y_end, x_start:x_end] = mask_resized[:mask_h, :mask_w]
                                    
                                    result_img = post_process_integration(result_img, full_obj_mask)
                                    
                                    # Stocker le résultat
                                    st.session_state.result_img = result_img
                                    st.session_state.full_obj_mask = full_obj_mask
                                    st.session_state.placement_done = True
                                    
                                    st.success("Objet placé avec succès ! 🎯")

                                except Exception as e:
                                    st.error(f"Erreur lors du placement intelligent: {str(e)}")

                            # Afficher le résultat si le placement est fait
                            if st.session_state.get('placement_done', False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(st.session_state.env_np, caption="Image originale", use_container_width=True)
                                with col2:
                                    st.image(st.session_state.result_img, caption="Avec objet placé", use_container_width=True)

                                # ======================== STABLE DIFFUSION ENHANCEMENT ========================
                                st.markdown("---")
                                st.subheader("🎨 Amélioration avec Stable Diffusion")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    use_sd = st.checkbox("Activer Stable Diffusion", value=False)
                                    enhancement_type = st.selectbox("Type d'amélioration", 
                                        ["img2img", "inpainting", "both"])
                                with col2:
                                    sd_strength = st.slider("Force SD", 0.1, 1.0, 0.3, 0.1)
                                    guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
                                with col3:
                                    apply_color_harmony_option = st.checkbox("Harmonisation couleurs", value=True)
                                    
                                if use_sd and st.button("🚀 Améliorer avec Stable Diffusion"):
                                    sd_processor = get_sd_processor()
                                    
                                    if sd_processor is not None:
                                        with st.spinner("Génération Stable Diffusion en cours..."):
                                            try:
                                                final_result = st.session_state.result_img
                                                
                                                if enhancement_type in ["img2img", "both"]:
                                                    st.info("Amélioration img2img...")
                                                    enhanced_img = enhance_with_img2img(
                                                        st.session_state.result_img, furniture_type, room_type, 
                                                        sd_processor, strength=sd_strength, 
                                                        guidance_scale=guidance_scale
                                                    )
                                                    final_result = enhanced_img
                                                    st.subheader("Résultat img2img")
                                                    st.image(enhanced_img, caption="Amélioré avec img2img", use_container_width=True)
                                                
                                                if enhancement_type in ["inpainting", "both"]:
                                                    st.info("Amélioration inpainting...")
                                                    inpaint_mask = create_inpainting_mask_from_object(st.session_state.full_obj_mask, expansion_radius=30)
                                                    
                                                    final_result = enhance_with_inpainting(
                                                        final_result, inpaint_mask, furniture_type, 
                                                        room_type, sd_processor, strength=sd_strength,
                                                        guidance_scale=guidance_scale
                                                    )
                                                    st.subheader("Résultat inpainting")
                                                    st.image(final_result, caption="Amélioré avec inpainting", use_container_width=True)
                                                
                                                if apply_color_harmony_option:
                                                    final_result = apply_color_harmony(final_result)
                                                    st.subheader("Résultat final avec harmonisation")
                                                    st.image(final_result, caption="✨ Résultat final optimisé", use_container_width=True)
                                                
                                                st.success("Amélioration Stable Diffusion terminée ! ✨")
                                                
                                            except Exception as e:
                                                st.error(f"Erreur lors de l'amélioration SD: {str(e)}")
                                                st.image(st.session_state.result_img, caption="Résultat sans SD", use_container_width=True)
                                    else:
                                        st.warning("Stable Diffusion non disponible.")
                                        st.image(st.session_state.result_img, caption="Résultat sans SD", use_container_width=True)
                                
                                else:
                                    # Afficher le résultat final
                                    st.markdown("---")
                                    st.subheader("🎯 Résultat final")
                                    final_display = st.session_state.result_img
                                    if apply_color_harmony_option:
                                        final_display = apply_color_harmony(final_display)
                                    st.image(final_display, caption="✅ Résultat final", use_container_width=True)

                        else:
                            st.warning("Aucune surface de placement appropriée détectée.")
                            st.info("Essayez avec une image ayant des surfaces planes plus définies (sol, table, etc.)")

# Bouton de reset
if st.session_state.get('analysis_done', False):
    st.markdown("---")
    if st.button("🔄 Recommencer avec de nouvelles images"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

else:
    if not env_img:
        st.info("👆 Veuillez uploader une photo d'environnement")
    if not obj_img:
        st.info("👆 Veuillez uploader une photo du meuble/fenêtre")

# Footer avec informations
st.markdown("---")
st.markdown("""
**🚀 Fonctionnalités disponibles :**
- Segmentation automatique avec SAM
- Estimation de profondeur avec MiDaS  
- Correction de perspective basée sur la profondeur
- Placement interactif par clic
- Amélioration avec Stable Diffusion (optionnel)
- Génération d'ombres réalistes
""")