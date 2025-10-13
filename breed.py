import streamlit as st
import sqlite3, hashlib, os, json, traceback
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from datetime import datetime

# ====================== DB Setup ======================
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

# ====================== Utils ======================
def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

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

def update_profile(uid,n,p,a):
    cur.execute("UPDATE users SET name=?,phone=?,address=? WHERE id=?",
                (n,p,a,uid))
    conn.commit()

# ====================== Model Loading ======================
MODEL = "bovine_breed_with_invalid.h5"
CLASS = "bovine_breed_with_invalid_classes.json"

model = None
idx_to_class = {}

if not os.path.exists(MODEL):
    st.error(f"⚠ Model file not found at: {os.path.abspath(MODEL)}")
elif not os.path.exists(CLASS):
    st.error(f"⚠ Class index file not found at: {os.path.abspath(CLASS)}")
else:
    try:
        model = load_model(MODEL, compile=False)
        with open(CLASS) as f:
            idx_to_class = {v: k for k, v in json.load(f).items()}
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"⚠ Model load error: {e}")
        st.text(traceback.format_exc())
        model, idx_to_class = None, {}

# ====================== Prediction ======================
def predict(img):
    if model is None:
        return "Unknown", 0
    img = img.resize((224,224))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr,0)
    p = model.predict(arr)
    idx = int(np.argmax(p[0]))
    return idx_to_class.get(idx, "Unknown"), float(p[0][idx]*100)

# ====================== Custom CSS ======================
st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #f5f7fa, #c3cfe2);
    font-family: "Segoe UI", sans-serif;
}
h1, h2, h3 {
    text-align: center;
    color: #333333;
}
.stButton>button {
    background: linear-gradient(45deg, #1e90ff, #28a745);
    color: white;
    font-weight: 600;
    border-radius: 12px;
    padding: 0.5rem 1.2rem;
    border: none;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
.stButton>button:hover {
    background: linear-gradient(45deg, #28a745, #1e90ff);
    transform: scale(1.05);
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e0eafc, #cfdef3);
}
</style>
""", unsafe_allow_html=True)

# ====================== Session ======================
if "user" not in st.session_state: st.session_state.user=None
if "page" not in st.session_state: st.session_state.page="auth"

# ====================== Pages ======================
def page_auth():
    st.header("🔑 Login / Signup")
    opt = st.radio("Menu",["Login","Sign Up","Skip"])
    if opt=="Sign Up":
        e = st.text_input("Email")
        p = st.text_input("Password", type="password")
        if st.button("Create Account"):
            try:
                add_user(e,p)
                st.success("✅ User created")
            except Exception as ex:
                st.error(f"⚠ Email exists or error: {ex}")
    elif opt=="Login":
        e = st.text_input("Email")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            u = login_user(e,p)
            if u:
                st.session_state.user = u
                st.session_state.page = "upload"
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")
    else:
        st.session_state.page="upload"; st.experimental_rerun()

def page_upload():
    st.header("📸 Upload / Capture Image")
    up = st.file_uploader("Choose image", ["jpg","jpeg","png"])
    cam = st.camera_input("Take a photo")
    if up or cam:
        img = Image.open(up if up else cam)
        os.makedirs("temp",exist_ok=True)
        path = os.path.join("temp", up.name if up else "captured.jpg")
        img.save(path)
        st.image(img,width=250)
        if st.button("🔍 Predict Breed"):
            br, conf = predict(img)
            st.success(f"**Breed:** {br}  \n**Confidence:** {conf:.1f}%")
            with st.form("save_form"):
                breed = st.text_input("Edit Breed", br)
                c1,c2,c3,c4 = st.columns(4)
                with c1: h = st.number_input("Height (cm)",50.0,250.0,120.0)
                with c2: w = st.number_input("Weight (kg)",50.0,1000.0,300.0)
                with c3: a = st.number_input("Age (yrs)",0.0,25.0,3.0)
                with c4: g = st.selectbox("Gender",["Male","Female"])
                if st.form_submit_button("💾 Save Result"):
                    if st.session_state.user:
                        save_prediction(st.session_state.user["id"], breed, h, w, a, g, path)
                        st.success("Saved ✅")
                    else:
                        st.warning("Login first to save")

def page_profile():
    st.header("👤 Profile")
    u = st.session_state.user
    if u:
        n = st.text_input("Name", u.get("name") or "")
        ph = st.text_input("Phone", u.get("phone") or "")
        ad = st.text_area("Address", u.get("address") or "")
        if st.button("💾 Save Profile"):
            update_profile(u["id"], n, ph, ad)
            st.success("Profile Updated")
        if st.button("🚪 Logout"):
            st.session_state.user=None
            st.session_state.page="auth"
            st.experimental_rerun()
    else:
        st.warning("Please login")

def page_records():
    st.header("📂 Records")
    if st.session_state.user:
        rec = get_records(st.session_state.user["id"])
        if rec:
            for i, r in enumerate(rec, 1):
                st.markdown(f"**{i}. {r['breed']}**  "
                            f"H:{r['height']} W:{r['weight']}  "
                            f"A:{r['age']} G:{r['gender']}  "
                            f"| *{r['created_at']}*")
                if r["image_path"] and os.path.exists(r["image_path"]):
                    st.image(r["image_path"], width=150)
        else:
            st.info("No records yet")
    else:
        st.warning("Login to view records")

# ====================== Router ======================
pg = st.session_state.page
if pg=="auth": page_auth()
elif pg=="upload": page_upload()
elif pg=="profile": page_profile()
else: page_records()

st.markdown("---")
c1,c2,c3 = st.columns(3)
if c1.button("🏠 Home"): st.session_state.page="upload"; st.experimental_rerun()
if c2.button("👤 Profile"): st.session_state.page="profile"; st.experimental_rerun()
if c3.button("📂 Records"): st.session_state.page="records"; st.experimental_rerun()
