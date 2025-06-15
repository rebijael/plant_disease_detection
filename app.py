import gdown
import os
from tensorflow.keras.models import load_model

# ✅ Download the model from Google Drive if not already downloaded
if not os.path.exists("plant_disease_model.h5"):
    url = "https://drive.google.com/uc?id=1_Zm_cuvHBiLlTYfCN1iLBskkBlRy-o8G"
    gdown.download(url, "plant_disease_model.h5", quiet=False)

# ✅ Load the model
model = load_model("plant_disease_model.h5")
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Load your model
model = load_model('plant_disease_model.h5')  # Make sure this is your model's filename

# Define class labels (Change as per your training)
class_labels = [f"Class {i}" for i in range(16)]

# App UI
st.title("🌿 Plant Disease Detection App")
st.write("Upload a plant leaf image to predict its condition using AI.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Resize and preprocess
    img = img.resize((180, 180))  # make sure this matches your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make prediction
    prediction = model.predict(img_array)

    # Debug info
    st.write("🔍 Prediction output:", prediction)
    st.write("📏 Prediction shape:", prediction.shape)
    st.write("🏷️ Number of class labels:", len(class_labels))

    # Safe prediction display
    if prediction.shape[1] == len(class_labels):
        predicted_label = class_labels[np.argmax(prediction)]
        st.success(f"🌿 **Predicted Disease:** {predicted_label}")
    else:
        st.error("⚠️ Prediction output doesn't match number of class labels. Check model and label list.")
