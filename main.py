import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Flower Classification App")

# ===============================
# LOAD MODEL & LABEL
# ===============================
@st.cache_resource
def load_model_and_labels():
    model = load_model("best_model.h5")
    class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    return model, class_labels

model, class_labels = load_model_and_labels()

# ===============================
# UPLOAD IMAGE
# ===============================
uploaded_file = st.file_uploader(
    "Upload gambar bunga",
    type=["jpg", "jpeg", "png"]
)

# ===============================
# PREDICTION
# ===============================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Gambar yang diupload",
        use_column_width=True
    )

    # Preprocessing
    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    idx = np.argmax(prediction)

    # Result
    st.success(f"🌼 Prediksi: **{class_labels[idx]}**")
    st.write(f"🔍 Keyakinan: **{prediction[idx] * 100:.2f}%**")

    # Optional: tampilkan semua confidence
    with st.expander("Lihat detail confidence"):
        for label, prob in zip(class_labels, prediction):
            st.write(f"{label}: {prob * 100:.2f}%")
