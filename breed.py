import streamlit as st
from tensorflow.keras.models import load_model
import os

MODEL_PATH = "bovine_breed_with_invalid.h5"

# Initialize model variable
model = None

# Try to load the model safely
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"⚠ Model failed to load: {e}")
else:
    st.warning(f"⚠ Model file not found at '{MODEL_PATH}'. Prediction disabled.")

# Example usage
if model:
    # Your prediction code here, e.g.:
    # img = preprocess_image(uploaded_file)
    # prediction = model.predict(img)
    pass
