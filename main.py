import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Flower Classification App")

# ===============================
# CONSTANTS
# ===============================
MODEL_PATH = "best_model.h5"
CLASS_LABELS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
IMG_SIZE = (150, 150)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model_only():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ File model tidak ditemukan!")
        st.stop()

    return load_model(MODEL_PATH, compile=False)

model = load_model_only()

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "Upload gambar bunga",
    type=["jpg", "jpeg", "png"]
)

# ===============================
# PREDICTION
# ===============================
if uploaded_file:
    try:
        # Load image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar yang diupload", use_column_width=True)

        # Preprocessing
        img_resized = img.resize(IMG_SIZE)
        img_array = image.img_to_array(img_resized)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array, verbose=0)[0]
        idx = np.argmax(prediction)

        # Result
        st.success(f"🌼 Prediksi: **{CLASS_LABELS[idx]}**")
        st.write(f"🔍 Keyakinan: **{prediction[idx] * 100:.2f}%**")

        # Detail confidence
        with st.expander("Lihat detail confidence"):
            for label, prob in zip(CLASS_LABELS, prediction):
                st.write(f"{label}: {prob * 100:.2f}%")

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses gambar.")
        st.exception(e)
