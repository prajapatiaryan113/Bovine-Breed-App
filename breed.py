import streamlit as st
import sqlite3
import hashlib
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# ---------------------------
# Paths and Model Download
# ---------------------------
MODEL_PATH = "bovine_breed_with_invalid.h5"
CLASSES_PATH = "bovine_breed_with_invalid_classes.json"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    url_model = "https://drive.google.com/uc?id=YOUR_H5_FILE_ID"
    gdown.download(url_model, MODEL_PATH, quiet=False)

# Download classes JSON if not exists
if not os.path.exists(CLASSES_PATH):
    url_classes = "https://drive.google.com/uc?id=YOUR_JSON_FILE_ID"
    gdown.download(url_classes, CLASSES_PATH, quiet=False)

# Load model and class labels
try:
    model = load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r") as f:
        class_labels = json.load(f)
    st.success("✅ मॉडल लोड हुआ, आप prediction कर सकते हैं!")
except:
    st.error("⚠ मॉडल लोड नहीं हुआ, prediction disabled.")
    model = None
    class_labels = None

# ---------------------------
# Database Setup
# ---------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    filename TEXT,
    predicted_breed TEXT
)
""")
conn.commit()

# ---------------------------
# Authentication
# ---------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup(username, password):
    try:
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                    (username, hash_password(password)))
        conn.commit()
        st.success("Sign Up Successful! Please Login.")
    except sqlite3.IntegrityError:
        st.error("Username already exists!")

def login(username, password):
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", 
                (username, hash_password(password)))
    return cur.fetchone() is not None

# ---------------------------
# Prediction Function
# ---------------------------
def predict_breed(uploaded_file, username):
    if model is None:
        st.warning("Model not loaded.")
        return
    
    # Save uploaded file
    save_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Image preprocessing
    img = image.load_img(save_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Prediction
    preds = model.predict(x)
    class_idx = np.argmax(preds, axis=1)[0]
    predicted_class = class_labels[str(class_idx)]

    # Save prediction in DB
    cur.execute("INSERT INTO predictions (username, filename, predicted_breed) VALUES (?, ?, ?)",
                (username, uploaded_file.name, predicted_class))
    conn.commit()

    st.success(f"Prediction: {predicted_class}")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🐄 Bovine Breed Classifier")

menu = st.sidebar.radio("Menu", ["Login", "Sign Up"])

if menu == "Sign Up":
    st.subheader("Create a new account")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        signup(new_user, new_pass)

if menu == "Login":
    st.subheader("Login to your account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.success(f"Logged in as {username}")

            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
            if uploaded_file and model:
                if st.button("Predict Breed"):
                    predict_breed(uploaded_file, username)
        else:
            st.error("Invalid username or password")
