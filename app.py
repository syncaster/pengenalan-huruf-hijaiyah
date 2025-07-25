# versi 1.0.3 app.py

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
from streamlit_drawable_canvas import st_canvas
from gtts import gTTS
import io
import random

# --- Konfigurasi Aplikasi ---
st.set_page_config(
    page_title="Game Edukasi Hijaiyah",
    page_icon="✒️",
    layout="wide"
)

# --- Kamus untuk memetakan nama kelas ke karakter Arab ---
HIJAIYAH_MAP = {
    'alif': 'ا', 'ba': 'ب', 'ta': 'ت', 'tsa': 'ث', 'jim': 'ج',
    'ha\'': 'ح', 'kho': 'خ', 'da': 'د', 'dzal': 'ذ', 'ra': 'ر',
    'zain': 'ز', 'sin': 'س', 'syin': 'ش', 'shod': 'ص', 'dhad': 'ض',
    'tho': 'ط', 'dzo': 'ظ', 'ain': 'ع', 'ghoin': 'غ', 'fa': 'ف',
    'qof': 'ق', 'kaf': 'ك', 'lam': 'ل', 'mim': 'م', 'nun': 'ن',
    'wawu': 'و', 'ha': 'ه', 'lamalif': 'لا', 'hamzah': 'ء', 'ya': 'ي'
}

PASTEL_COLORS = [
    "#FFDFD3", "#D0E7FF", "#D4F0F0", "#F8E4F5", "#FFFACD",
    "#D1E2C4", "#FFDAB9", "#E6E6FA", "#BDECB6", "#FFC0CB",
    "#C8A2C8", "#F5DEB3", "#B0E0E6", "#FFE4E1", "#AFEEEE"
]


@st.cache_resource
def load_model():
    """Memuat model H5 yang sudah dilatih."""
    model_path = 'model/hijaiyah_model.h5'
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di {model_path}.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


def predict(image_data, model):
    """Melakukan prediksi pada gambar yang diunggah."""
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)

    # Konversi gambar RGBA dari kanvas ke RGB
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    img_array_normalized = img_array / 255.0
    img_batch = np.expand_dims(img_array_normalized, 0)
    prediction = model.predict(img_batch)
    return prediction


def play_sound(text):
    """Membuat dan memutar audio dari teks menggunakan gTTS."""
    try:
        tts = gTTS(text=text, lang='ar', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        st.audio(audio_fp, format='audio/mp3')
    except Exception as e:
        st.error(f"Gagal memutar suara: {e}")


model = load_model()
class_names = []
# Cek apakah direktori dataset ada
if os.path.exists('dataset/training'):
    class_names = sorted(os.listdir('dataset/training'))
else:
    # Jangan tampilkan error jika di halaman belajar, hanya jika model diperlukan
    pass

# =================================================================
# --- Sidebar untuk Navigasi ---
# =================================================================
sidebar_logo = "images/hijaiyah-1.0.3.png"
st.logo(sidebar_logo, size="large")
st.sidebar.markdown("**SiPuTiH** - Sistem Pengenalan Tulisan Tangan Hijaiyah")

st.sidebar.title("Pilih Menu")
menu_choice = st.sidebar.radio(
    "Navigasi:",
    ("📖 Belajar Huruf", "✍️ Kuis Tulisan Tangan", "🤔 Tebak Huruf Hijaiyah")
)

st.sidebar.header("Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini menggunakan model Convolutional Neural Network (CNN) "
    "yang dilatih dengan TensorFlow/Keras untuk mengenali huruf hijaiyah.\n"
    "Antarmuka web dibuat menggunakan Streamlit."
    "\n\n"
    "**Dibuat oleh:** Saiful Nur Budiman dan Sri Lestanti\n"
    "**version:** 1.0.3"
)

# =================================================================
# --- Halaman 1: Belajar Huruf
# =================================================================
if menu_choice == "📖 Belajar Huruf":
    st.title("📖 Mari Mengenal Huruf Hijaiyah")
    st.info("Klik tombol '🔊 Putar Suara' di bawah setiap huruf untuk mendengarkan pelafalannya.")
    st.markdown("---")

    hijaiyah_items = list(HIJAIYAH_MAP.items())

    # --- PERBAIKAN BUG: Simpan warna di session state ---
    # Buat daftar warna hanya jika belum ada di session state
    if 'belajar_card_colors' not in st.session_state:
        st.session_state.belajar_card_colors = [random.choice(PASTEL_COLORS) for _ in hijaiyah_items]

    # --- PERBAIKAN TAMPILAN MOBILE ---
    num_columns = 5
    for i in range(0, len(hijaiyah_items), num_columns):
        cols = st.columns(num_columns)
        row_items = hijaiyah_items[i:i + num_columns]

        for j, (name, char) in enumerate(row_items):
            with cols[j]:
                with st.container(border=True):
                    card_color = st.session_state.belajar_card_colors[i + j]

                    # Menampilkan arab
                    st.markdown(
                        f"""
                        <div style='background-color: {card_color}; border-radius: 8px; padding: 10px; margin-bottom: 
                        10px;'> <h1 style='text-align: center; font-size: 80px; font-family: "Noto Naskh Arabic", 
                        sans-serif; color: #333; margin: 0;'>{char}</h1>
                        </div>
                        """,
                        unsafe_allow_html=True)

                    # Menampilkan nama latin huruf
                    st.markdown(f"<p style='text-align: center; font-weight: bold;'>{name.capitalize()}</p>",
                                unsafe_allow_html=True)

                    # Tombol untuk memutar suara
                    if st.button(f"🔊 Putar Suara", key=f"play_{name}", use_container_width=True):
                        play_sound(char)

# =================================================================
# --- Halaman 2: Kuis Tulisan Tangan ---
# =================================================================
elif menu_choice == "✍️ Kuis Tulisan Tangan":
    # --- PERBAIKAN BUG: Hapus state warna agar di-reset saat kembali ---
    if 'belajar_card_colors' in st.session_state:
        del st.session_state.belajar_card_colors
    st.session_state.game_needs_reset = True

    st.title("✒️ Kuis Tulisan Tangan Hijaiyah")
    st.write(
        "Silakan gambar satu huruf hijaiyah pada kanvas di bawah ini, "
        "lalu klik tombol 'Prediksi' untuk melihat dan mendengar hasilnya."
    )

    if not class_names:
        st.error("Direktori 'dataset/training' tidak ditemukan. Kuis tidak dapat dijalankan.")

    if model is None:
        st.warning("Model tidak dapat dimuat. Fitur kuis tidak dapat berjalan.")

    stroke_width = st.slider("Tebal Kuas: ", 5, 25, 10)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Area Menggambar")
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
            image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')

            with st.spinner("Sedang menganalisis tulisan Anda..."):
                prediction = predict(image, model)
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = class_names[predicted_class_index]
                confidence = np.max(prediction)

                # Tampilkan hasil di kolom kedua
                with col2:
                    st.subheader("Hasil Prediksi")
                    st.success(f"Prediksi: **{predicted_class_name.capitalize()}**")
                    st.info(f"Tingkat Keyakinan: **{confidence:.2%}**")

                    # Dapatkan karakter Arab dari kamus untuk diputar suaranya
                    arabic_character = HIJAIYAH_MAP.get(predicted_class_name.lower(), None)
                    if arabic_character:
                        play_sound(arabic_character)
                    else:
                        st.warning(f"Tidak ditemukan pemetaan karakter Arab untuk '{predicted_class_name}'.")

                    # Tampilkan detail probabilitas dalam expander
                    with st.expander("Lihat Detail Probabilitas"):
                        prob_df = {
                            "Huruf": class_names,
                            "Probabilitas": [f"{p:.2%}" for p in prediction[0]]
                        }
                        st.dataframe(prob_df)
        else:
            with col2:
                st.warning("Mohon gambar sesuatu di kanvas terlebih dahulu.")

# =================================================================
# --- Halaman 3: Tebak Huruf Hijaiyah
# =================================================================
elif menu_choice == "🤔 Tebak Huruf Hijaiyah":
    if 'belajar_card_colors' in st.session_state:
        del st.session_state.belajar_card_colors

    st.title("🤔 Tebak Huruf Hijaiyah")
    st.info("Lihat huruf di layar, lalu pilih jawaban yang benar!")


    def setup_new_question():
        """Memilih huruf acak, 3 jawaban salah, dan warna latar belakang acak, lalu menyimpannya di session state."""
        correct_name, correct_char = random.choice(list(HIJAIYAH_MAP.items()))
        st.session_state.correct_char = correct_char
        st.session_state.correct_name = correct_name

        st.session_state.card_color = random.choice(PASTEL_COLORS)

        all_names = list(HIJAIYAH_MAP.keys())
        all_names.remove(correct_name)
        wrong_options = random.sample(all_names, 3)

        options = wrong_options + [correct_name]
        random.shuffle(options)
        st.session_state.options = options
        st.session_state.answered = False


    if st.session_state.get('game_needs_reset', True):
        setup_new_question()
        st.session_state.game_needs_reset = False

    card_color = st.session_state.get('card_color', '#F0F2F6')
    st.markdown(f"""
    <div style="background-color: {card_color}; border-radius: 15px; padding: 30px; text-align: center; margin-top: 
    20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"> <h1 style='font-size: 200px; font-family: "Noto Naskh Arabic", 
    "Arial", sans-serif; margin:0; color: #333;'>{st.session_state.correct_char}</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(4)
    for i, option in enumerate(st.session_state.options):
        if cols[i].button(option.capitalize(), key=f"option_{i}", use_container_width=True):
            st.session_state.selected_answer = option
            st.session_state.answered = True
            st.rerun()

    if st.session_state.get('answered', False):
        is_correct = (st.session_state.selected_answer == st.session_state.correct_name)

        if is_correct:
            st.success("🎉 Benar! Kamu Hebat!")
            st.balloons()
        else:
            st.error(f"Yah, salah. Jawaban yang benar adalah '{st.session_state.correct_name.capitalize()}'.")

        st.info(f"Ini adalah huruf **{st.session_state.correct_name.capitalize()}**.")
        play_sound(st.session_state.correct_char)

        if st.button("Lanjut ke Soal Berikutnya"):
            setup_new_question()
            st.rerun()
