from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit as st
import numpy as np
from PIL import Image

def apply_mask_to_image(image_array, mask):
    """Applique un masque à une image en rendant le fond transparent"""
    if len(image_array.shape) == 3:
        # Ajouter canal alpha
        result = np.zeros((*image_array.shape[:2], 4), dtype=np.uint8)
        result[:, :, :3] = image_array
        result[:, :, 3] = mask.astype(np.uint8) * 255
    else:
        result = image_array.copy()
        result[:, :, 3] = mask.astype(np.uint8) * 255
    
    return result

def interactive_placement_section():
    """Section de placement interactif avec clic sur l'image"""
    
    st.markdown("---")
    st.subheader("Placement de l'objet")
    st.write("Cliquez directement sur l'image pour placer votre objet :")
    
    # Affichage de l'image avec capture des coordonnées du clic
    with st.container():
        # Utiliser streamlit-image-coordinates pour capturer les clics
        coordinates = streamlit_image_coordinates(
            st.session_state.env_np,
            key="image_click",
            width=600  # Largeur d'affichage
        )
        
        # Si l'utilisateur a cliqué sur l'image
        if coordinates is not None:
            # Récupérer les coordonnées du clic
            click_x = int(coordinates["x"])
            click_y = int(coordinates["y"])
            
            # Calculer les coordonnées réelles en fonction du redimensionnement d'affichage
            display_width = 600
            actual_height, actual_width = st.session_state.env_np.shape[:2]
            scale_factor = actual_width / display_width
            
            real_x = int(click_x * scale_factor)
            real_y = int(click_y * scale_factor)
            
            # S'assurer que les coordonnées sont dans les limites
            real_x = max(0, min(real_x, actual_width - 1))
            real_y = max(0, min(real_y, actual_height - 1))
            
            st.success(f"Point cliqué : ({real_x}, {real_y})")
            
            # Calcul automatique de l'échelle basée sur la profondeur
            depth_value = st.session_state.env_depth_map[real_y, real_x]
            norm_depth = (depth_value - st.session_state.env_depth_map.min()) / \
                        (st.session_state.env_depth_map.max() - st.session_state.env_depth_map.min() + 1e-8)
            auto_scale = max(0.1, 1.0 - norm_depth * 0.8)
            
            # Contrôles pour affiner le placement
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Position :** ({real_x}, {real_y})")
                st.write(f"**Profondeur :** {depth_value:.2f}")
                
            with col2:
                # Permettre d'ajuster l'échelle
                scale = st.slider("Ajuster l'échelle", 0.1, 2.0, auto_scale, 0.05)
                
                # Bouton de confirmation du placement
                if st.button("🎯 Confirmer le placement", type="primary"):
                    place_object_at_coordinates(real_x, real_y, scale)

def place_object_at_coordinates(x, y, scale):
    """Place l'objet aux coordonnées spécifiées"""
    try:
        # Préparer l'objet avec le masque appliqué
        obj_rgb = np.array(st.session_state.obj_img_processed)
        obj_rgba = apply_mask_to_image(obj_rgb, st.session_state.combined_mask)
        
        # Redimensionner l'objet
        obj_pil = Image.fromarray(obj_rgba, 'RGBA')
        new_size = (int(obj_pil.width * scale), int(obj_pil.height * scale))
        obj_resized = obj_pil.resize(new_size, Image.Resampling.LANCZOS)
        
        # Créer l'image finale
        result_img = Image.fromarray(st.session_state.env_np).convert('RGBA')
        
        # Calculer la position de collage (centré sur le point cliqué)
        paste_x = max(0, x - obj_resized.width // 2)
        paste_y = max(0, y - obj_resized.height // 2)
        
        # Coller l'objet
        result_img.paste(obj_resized, (paste_x, paste_y), obj_resized)
        
        st.success("🎉 Objet placé avec succès!")
        st.image(result_img, caption="Résultat final", use_container_width=True)
        
        # Option de téléchargement
        from io import BytesIO
        buf = BytesIO()
        result_img.convert('RGB').save(buf, format='PNG')
        st.download_button(
            label="📥 Télécharger le résultat",
            data=buf.getvalue(),
            file_name="objet_place.png",
            mime="image/png"
        )
        
    except Exception as e:
        st.error(f"Erreur lors du placement : {e}")