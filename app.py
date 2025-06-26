import streamlit as st
from PIL import Image

st.title("Mise en situation de produit - Démo")

st.markdown("""
### Guide utilisateur
1. Téléchargez une photo de votre pièce.
2. Téléchargez la photo du meuble ou fenêtre.
3. Cliquez sur “Placer l’objet” et patientez.
4. Découvrez le rendu final avec l'objet intégré.
""")

env_img = st.file_uploader("Uploader la photo d'environnement (ex: salon)", type=["png", "jpg", "jpeg"])
obj_img = st.file_uploader("Uploader la photo du meuble/fenêtre", type=["png", "jpg", "jpeg"])

if env_img and obj_img:
    env_img_pil = Image.open(env_img)
    obj_img_pil = Image.open(obj_img)

    st.image(env_img_pil, caption="Photo environnement")
    st.image(obj_img_pil, caption="Image meuble/fenêtre")

    if st.button("Placer l'objet"):
        # TODO: pipeline IA (SAM, MiDaS, OpenCV, SD)
        st.write("Traitement en cours...")

        st.image(env_img_pil, caption="Résultat final (placeholder)")
