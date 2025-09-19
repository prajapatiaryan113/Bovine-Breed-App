import streamlit as st
import sqlite3, hashlib, os, json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from datetime import datetime

# ========== Database Setup ==========
conn = sqlite3.connect("users.db", check_same_thread=False)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute("""CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE,
    password TEXT,
    name TEXT,
    phone TEXT,
    address TEXT
)""")
cur.execute("""CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    breed TEXT,
    height REAL,
    weight REAL,
    gender TEXT,
    image_path TEXT,
    age REAL,
    created_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
)""")
conn.commit()

# ========== Utilities ==========
def hash_pw(p): 
    return hashlib.sha256(p.encode()).hexdigest()

def add_user(email, pw):
    cur.execute("INSERT INTO users(email,password) VALUES(?,?)",
                (email, hash_pw(pw)))
    conn.commit()

def login_user(email, pw):
    cur.execute("SELECT * FROM users WHERE email=? AND password=?",
                (email, hash_pw(pw)))
    row = cur.fetchone()
    return dict(row) if row else None

def save_prediction(uid, breed, h, w, age, g, path):
    cur.execute("""INSERT INTO predictions
        (user_id,breed,height,weight,age,gender,image_path,created_at)
        VALUES(?,?,?,?,?,?,?,?)""",
        (uid, breed, h, w, age, g, path,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

def get_records(uid):
    cur.execute("""SELECT breed,height,weight,age,gender,image_path,created_at
                   FROM predictions WHERE user_id=? ORDER BY id DESC""",
                (uid,))
    return cur.fetchall()

def update_profile(uid, n, p, a):
    cur.execute("UPDATE users SET name=?,phone=?,address=? WHERE id=?",
                (n,p,a,uid))
    conn.commit()

# ========== Load Model ==========
MODEL_PATH = "bovine_breed_with_invalid.h5"
CLASS_PATH = "bovine_breed_with_invalid_classes.json"
try:
    model = load_model(MODEL_PATH)
    with open(CLASS_PATH) as f:
        idx_to_class = {v:k for k,v in json.load(f).items()}
except Exception as e:
    st.warning("⚠ मॉडल लोड नहीं हुआ, prediction disabled.")
    model, idx_to_class = None, {}

def predict(img_path):
    if model is None:
        return "Unknown", 0
    img = Image.open(img_path).resize((224,224))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr,0)
    p = model.predict(arr)
    idx = np.argmax(p[0])
    return idx_to_class.get(idx, "Unknown"), p[0][idx]*100

# ========== Streamlit Theme ==========
st.markdown("""
<style>
body {background:linear-gradient(135deg,#e6ffed,#ffffff,#d0f0fd);}
.stButton>button{
    background:#1e90ff;color:white;font-weight:bold;border-radius:8px;}
.stButton>button:hover{background:#28a745;color:white;}
</style>""", unsafe_allow_html=True)

# ========== Session ==========
if "user" not in st.session_state: st.session_state.user=None
if "page" not in st.session_state: st.session_state.page="auth"

# ========== Pages ==========
def page_auth():
    st.header("Login / Signup")
    option = st.radio("Menu", ["Login", "Sign Up", "Skip"])
    if option == "Sign Up":
        e = st.text_input("Email")
        p = st.text_input("Password", type="password")
        if st.button("Create"):
            try:
                add_user(e, p)
                st.success("✅ Account Created")
            except:
                st.error("⚠ Email already exists")
    elif option == "Login":
        e = st.text_input("Email")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            u = login_user(e, p)
            if u:
                st.session_state.user = u
                st.session_state.page = "upload"
                st.experimental_rerun()
            else:
                st.error("⚠ Invalid credentials")
    else:
        st.session_state.page = "upload"
        st.experimental_rerun()

def page_upload():
    st.header("Upload / Capture Image")
    uploaded_file = st.file_uploader("Choose Image", ["jpg","jpeg","png"])
    cam_file = st.camera_input("Or use Camera")
    img_path = None

    if uploaded_file or cam_file:
        os.makedirs("uploads", exist_ok=True)
        file_to_save = uploaded_file if uploaded_file else cam_file
        img_path = os.path.join("uploads", file_to_save.name if uploaded_file else "captured.jpg")
        with open(img_path, "wb") as f:
            f.write(file_to_save.getbuffer())
        st.image(img_path, width=220)

        if st.button("🔍 Predict Breed"):
            breed, conf = predict(img_path)
            st.success(f"Breed: {breed} ({conf:.1f}%)")

            with st.form("save_form"):
                breed_edit = st.text_input("Edit Breed", breed)
                c1, c2, c3, c4 = st.columns(4)
                with c1: h = st.number_input("Height(cm)", 50.0, 250.0, 120.0)
                with c2: w = st.number_input("Weight(kg)", 50.0, 1000.0, 300.0)
                with c3: a = st.number_input("Age(yrs)", 0.0, 25.0, 3.0)
                with c4: g = st.selectbox("Gender", ["Male","Female"])
                sub = st.form_submit_button("💾 Save Result")
                if sub:
                    if st.session_state.user:
                        save_prediction(st.session_state.user["id"], breed_edit, h, w, a, g, img_path)
                        st.success("✅ Saved")
                    else:
                        st.warning("⚠ Login first to save")

def page_profile():
    st.header("Profile")
    u = st.session_state.user
    if u:
        n = st.text_input("Name", u.get("name") or "")
        ph = st.text_input("Phone", u.get("phone") or "")
        ad = st.text_area("Address", u.get("address") or "")
        if st.button("💾 Save Profile"):
            update_profile(u["id"], n, ph, ad)
            st.success("Profile Updated!")
        if st.button("🚪 Logout"):
            st.session_state.user = None
            st.session_state.page = "auth"
            st.experimental_rerun()
    else:
        st.warning("⚠ Please login")

def page_records():
    st.header("📂 Prediction Records")
    if st.session_state.user:
        recs = get_records(st.session_state.user["id"])
        if recs:
            for i, r in enumerate(recs, 1):
                st.write(f"**{i}. {r['breed']}** | H:{r['height']} W:{r['weight']} A:{r['age']} G:{r['gender']} | {r['created_at']}")
                st.image(r["image_path"], width=150)
        else:
            st.info("No records yet.")
    else:
        st.warning("⚠ Login to view records")

# ========== Router ==========
pg = st.session_state.page
if pg == "auth": page_auth()
elif pg == "upload": page_upload()
elif pg == "profile": page_profile()
else: page_records()

st.markdown("---")
c1, c2, c3 = st.columns(3)
if c1.button("🏠 Home"): st.session_state.page="upload"; st.experimental_rerun()
if c2.button("👤 Profile"): st.session_state.page="profile"; st.experimental_rerun()
if c3.button("📂 Records"): st.session_state.page="records"; st.experimental_rerun()
