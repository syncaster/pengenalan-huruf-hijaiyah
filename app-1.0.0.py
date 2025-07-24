# app-1.0.0.py

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
from streamlit_drawable_canvas import st_canvas
from gtts import gTTS
import io

# --- Konfigurasi Aplikasi ---
st.set_page_config(
    page_title="Pengenalan Huruf Hijaiyah",
    page_icon="✒️",
    layout="centered"
)

# --- Kamus untuk memetakan nama kelas ke karakter Arab ---
# Ini penting agar gTTS dapat melafalkan karakter Arab dengan benar.
HIJAIYAH_MAP = {
    'alif': 'ا', 'ba': 'ب', 'ta': 'ت', 'tsa': 'ث', 'jim': 'ج',
    'ha\'': 'ح', 'kho': 'خ', 'da': 'د', 'dzal': 'ذ', 'ra': 'ر',
    'zain': 'ز', 'sin': 'س', 'syin': 'ش', 'shod': 'ص', 'dhad': 'ض',
    'tho': 'ط', 'dzo': 'ظ', 'ain': 'ع', 'ghoin': 'غ', 'fa': 'ف',
    'qof': 'ق', 'kaf': 'ك', 'lam': 'ل', 'mim': 'م', 'nun': 'ن',
    'wawu': 'و', 'ha': 'ه', 'lamalif': 'لا', 'hamzah': 'ء', 'ya': 'ي'
}


# Catatan: Pastikan nama folder di dataset Anda (misalnya 'shod') cocok dengan kunci di kamus ini.
# Anda mungkin perlu menyesuaikan kunci ('shod', 'ha', dll.) agar sesuai persis dengan nama folder Anda.


# --- Fungsi untuk Memuat Model ---
@st.cache_resource
def load_model():
    """Memuat model H5 yang sudah dilatih."""
    model_path = 'model/hijaiyah_model.h5'
    if not os.path.exists(model_path):
        st.error(
            f"File model tidak ditemukan di {model_path}. Harap latih model terlebih dahulu dengan menjalankan "
            f"`train_model.py`.")
        return None
    model = tf.keras.models.load_model(model_path)
    return model


# --- Fungsi untuk Prediksi ---
def predict(image_data, model):
    """Melakukan prediksi pada gambar yang diunggah."""
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)

    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    img_array_normalized = img_array / 255.0
    img_batch = np.expand_dims(img_array_normalized, 0)
    prediction = model.predict(img_batch)
    return prediction


# --- Antarmuka Aplikasi ---
model = load_model()

if os.path.exists('dataset/training'):
    class_names = sorted(os.listdir('dataset/training'))
else:
    st.error("Direktori 'dataset/training' tidak ditemukan. Tidak dapat memuat nama kelas.")
    class_names = []

st.title("✒️ Pengenalan Tulisan Tangan Hijaiyah (Real-time)")
st.write(
    "Silakan gambar satu huruf hijaiyah pada kanvas di bawah ini, "
    "lalu klik tombol 'Prediksi' untuk melihat dan mendengar hasilnya."
)

# --- Kanvas untuk Menggambar ---
stroke_width = st.sidebar.slider("Tebal Kuas: ", 5, 25, 10)
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# Tombol untuk memulai prediksi
if st.button("Prediksi Tulisan Tangan") and model is not None and class_names:
    if canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('RGB')

        with st.spinner("Sedang menganalisis..."):
            prediction = predict(image, model)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class_index]
            confidence = np.max(prediction)

            st.success(f"Prediksi: **{predicted_class_name.capitalize()}**")
            st.info(f"Tingkat Keyakinan: **{confidence:.2%}**")

            # --- Modifikasi untuk Suara dengan Pelafalan Arab ---
            try:
                # Dapatkan karakter Arab dari kamus
                arabic_character = HIJAIYAH_MAP.get(predicted_class_name, None)

                if arabic_character:
                    # Buat objek audio di memori menggunakan karakter Arab
                    tts = gTTS(text=arabic_character, lang='ar', slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)

                    # Putar audio menggunakan st.audio
                    st.audio(audio_fp, format='audio/mp3')
                else:
                    st.warning(f"Tidak ditemukan pemetaan karakter Arab untuk '{predicted_class_name}'.")

            except Exception as e:
                st.error(f"Gagal memutar suara: {e}")
            # --- Akhir Modifikasi ---

            with st.expander("Lihat Detail Probabilitas"):
                prob_df = {
                    "Huruf": class_names,
                    "Probabilitas": [f"{p:.2%}" for p in prediction[0]]
                }
                st.dataframe(prob_df)
    else:
        st.warning("Mohon gambar sesuatu di kanvas terlebih dahulu.")

elif model is None:
    st.warning("Model tidak dapat dimuat. Aplikasi tidak dapat berjalan.")

st.sidebar.header("Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini menggunakan model Convolutional Neural Network (CNN) "
    "yang dilatih dengan TensorFlow/Keras untuk mengenali huruf hijaiyah. "
    "Antarmuka web dibuat menggunakan Streamlit dengan input kanvas real-time."
    "by: Saiful Nur Budiman dan Sri Lestanti"
)
