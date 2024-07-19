import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Application de Détection de Visage")
st.write("""
### Instructions:
1. Téléchargez une image.
2. Ajustez les paramètres de détection de visage.
3. Choisissez la couleur des rectangles.
4. Cliquez sur 'Détecter les visages'.
5. Cliquez sur 'Enregistrer l'image' pour sauvegarder l'image avec les visages détectés.
""")

uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

# Paramètres de détection de visage
scaleFactor = st.slider("Ajustez le scaleFactor", 1.01, 2.0, 1.1, 0.01)
minNeighbors = st.slider("Ajustez le minNeighbors", 1, 20, 5)
rectangle_color = st.color_picker("Choisissez la couleur des rectangles", "#FF0000")

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file).convert('RGB'))
    st.image(image, caption='Image téléchargée', use_column_width=True)

    # Charger le classificateur en cascade pour la détection de visage
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if st.button("Détecter les visages"):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), tuple(int(rectangle_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)), 2)
        
        st.image(image, caption='Image avec visages détectés', use_column_width=True)

        if st.button("Enregistrer l'image"):
            cv2.imwrite("detected_faces.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            st.write("Image enregistrée sous le nom de detected_faces.png")