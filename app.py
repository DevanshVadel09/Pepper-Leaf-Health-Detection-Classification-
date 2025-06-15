import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model('PlantVillage.h5')

# Set page title
st.title("🌿 Pepper-Bell-Classification - Healthy vs Defective")
st.write("Upload a leaf image and let the CNN model classify it!")

# Upload an image
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (64, 64)) 
    img_normalized = img_resized / 255.0
    img_input = img_normalized.reshape(1, 64, 64, 3)

    # Make prediction
    prediction = model.predict(img_input)[0][0]

    # Show result
    st.write("### Prediction:")
    if prediction > 0.5:
        st.error("❌ Defective Leaf Detected")
    else:
        st.success("✅ Healthy Leaf Detected")
        
## open treminal type streamlit run app.py for webpage testing