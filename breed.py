import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import json

# ===== Paths for model and classes =====
MODEL_PATH = "bovine_breed_with_invalid.h5"
CLASSES_PATH = "bovine_breed_with_invalid_classes.json"

# ===== Load model =====
try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error("⚠ Model not loaded, prediction disabled.")
    model = None

# ===== Load classes =====
try:
    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)
except Exception as e:
    st.error("⚠ Classes JSON not loaded.")
    classes = None

# ===== Helper function for prediction =====
def predict_breed(img):
    if model is None or classes is None:
        return "Prediction disabled"
    
    img = img.resize((224, 224))  # Model input size
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)
    idx = np.argmax(pred)
    return classes[str(idx)]

# ===== Streamlit UI =====
st.title("Bovine Breed Classifier")

uploaded_file = st.file_uploader("Upload a cow/bull image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Breed"):
        result = predict_breed(img)
        st.write(f"Predicted Breed: {result}")
