import streamlit as st
import sqlite3
import hashlib
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# -------------------- SESSION INIT --------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "auth"

# -------------------- DATABASE ------------------------
DB_PATH = "users.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE,
    password TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    breed TEXT,
    height REAL,
    weight REAL,
    age REAL,
    gender TEXT,
    image_path TEXT
)
""")

conn.commit()

# -------------------- USER AUTH -----------------------
def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

def add_user(email, password):
    cur.execute("INSERT INTO users (email, password) VALUES (?,?)", (email, hash_password(password)))
    conn.commit()

def login_user(email, password):
    cur.execute("SELECT * FROM users WHERE email=? AND password=?", (email, hash_password(password)))
    return cur.fetchone()

# -------------------- MODEL LOADING -------------------
MODEL_PATH = "bovine_breed_with_invalid.h5"
CLASS_PATH = "bovine_breed_with_invalid_classes.json"

if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_PATH):
    model = load_model(MODEL_PATH)
    with open(CLASS_PATH) as f:
        idx_to_class = {int(v): k for k, v in json.load(f).items()}
else:
    model = None
    idx_to_class = {}
    st.error("⚠ मॉडल लोड नहीं हुआ: Model files missing")

# -------------------- PREDICTION ---------------------
def predict_breed(img_path):
    if not model:
        st.error("Model not loaded")
        return None
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    class_idx = np.argmax(pred)
    return idx_to_class.get(class_idx, "Unknown")

# -------------------- SAVE PREDICTION ----------------
def save_prediction(user_id, breed, h, w, a, g, path):
    cur.execute(
        "INSERT INTO predictions (user_id, breed, height, weight, age, gender, image_path) VALUES (?,?,?,?,?,?,?)",
        (user_id, breed, h, w, a, g, path)
    )
    conn.commit()

# -------------------- PAGE: AUTH ---------------------
def page_auth():
    st.header("Login / Sign Up")
    option = st.radio("Menu", ["Login", "Sign Up", "Skip"])

    if option == "Sign Up":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Create Account"):
            try:
                add_user(email, password)
                st.success("✅ Account Created")
            except sqlite3.IntegrityError:
                st.error("⚠ Email already exists")

    elif option == "Login":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(email, password)
            if user:
                st.session_state.user = user
                st.session_state.page = "upload"
                st.experimental_rerun()
            else:
                st.error("⚠ Invalid credentials")

    else:  # Skip
        if st.button("Continue without login"):
            st.session_state.page = "upload"
            st.experimental_rerun()

# -------------------- PAGE: UPLOAD -------------------
def page_upload():
    st.header("Bovine Breed Prediction")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    h = st.number_input("Height (cm)", 0.0, 500.0, 100.0)
    w = st.number_input("Weight (kg)", 0.0, 2000.0, 200.0)
    a = st.number_input("Age (years)", 0.0, 50.0, 2.0)
    g = st.selectbox("Gender", ["Male", "Female"])

    if uploaded_file:
        save_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.g_
