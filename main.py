import streamlit as st
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Judul aplikasi
st.title("🌸 Flower Classification App")

# Ambil label kelas dari folder train
class_labels = sorted([
    d for d in os.listdir("flowers_split/train")
    if os.path.isdir(os.path.join("flowers_split/train", d))
])

# Load model (cache agar tidak reload terus)
@st.cache_resource
def load_my_model():
    return load_model("model.h5")

model = load_my_model()

# Upload gambar
uploaded_file = st.file_uploader(
    "Upload gambar bunga",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Buka gambar
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing
    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Tampilkan hasil
    st.success(f"🌼 **Prediksi:** {predicted_class}")
    st.write(f"🔍 **Keyakinan:** {confidence:.2f}%")
