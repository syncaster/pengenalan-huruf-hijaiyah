# game.py

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
from streamlit_drawable_canvas import st_canvas
from gtts import gTTS
import io
import base64
import random

# --- Konfigurasi Aplikasi ---
st.set_page_config(
    page_title="Game Edukasi Hijaiyah",
    page_icon="✒️",
    layout="wide"
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


# Catatan: Nama kunci (e.g., 'ha\'') harus cocok dengan nama folder di dataset Anda.


# --- Fungsi untuk Memuat Model ---
@st.cache_resource
def load_model():
    """Memuat model H5 yang sudah dilatih."""
    model_path = 'model/hijaiyah_model.h5'
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di {model_path}.")
        return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        # Mengompilasi ulang model setelah dimuat adalah praktik yang baik
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


# --- Fungsi untuk Prediksi ---
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


# --- Fungsi untuk memutar suara (untuk halaman kuis dan tebak huruf) ---
def play_sound(text):
    """Membuat dan memutar audio dari teks menggunakan gTTS dan st.audio."""
    try:
        tts = gTTS(text=text, lang='ar', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        st.audio(audio_fp, format='audio/mp3', start_time=0)
    except Exception as e:
        st.error(f"Gagal memutar suara: {e}")


# --- Memuat Model dan Nama Kelas ---
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
st.sidebar.title("Pilih Menu")
menu_choice = st.sidebar.radio(
    "Navigasi:",
    ("📖 Belajar Huruf", "✍️ Kuis Tulisan Tangan", "🤔 Tebak Huruf Hijaiyah")
)

st.sidebar.header("Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini menggunakan model Convolutional Neural Network (CNN) "
    "yang dilatih dengan TensorFlow/Keras untuk mengenali huruf hijaiyah. "
    "Antarmuka web dibuat menggunakan Streamlit."
    "\n\n"
    "**Dibuat oleh:**\nSaiful Nur Budiman & Sri Lestanti"
)

# =================================================================
# --- Halaman 1: Belajar Huruf ---
# =================================================================
if menu_choice == "📖 Belajar Huruf":
    st.title("📖 Mari Mengenal Huruf Hijaiyah")
    st.info("Klik pada sebuah huruf untuk mendengarkan pelafalannya.")
    st.markdown("---")

    num_columns = 5
    cols = st.columns(num_columns)
    hijaiyah_items = list(HIJAIYAH_MAP.items())
    colors = ["#FFADAD", "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF", "#A0C4FF", "#BDB2FF", "#FFC6FF"]


    @st.cache_data
    def get_audio_b64(char):
        try:
            tts = gTTS(text=char, lang='ar', slow=False)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            return base64.b64encode(audio_fp.read()).decode('utf-8')
        except Exception:
            return None


    for index, (name, char) in enumerate(hijaiyah_items):
        col = cols[index % num_columns]
        with col:
            audio_b64 = get_audio_b64(char)
            if audio_b64:
                bg_color = colors[index % len(colors)]
                safe_name_id = name.replace("'", "").replace(" ", "")
                html_card = f"""
                <div onclick="document.getElementById('audio-{safe_name_id}').play(); return false;"
                    style="background-color: {bg_color}; cursor: pointer; padding: 15px; border: 2px solid {bg_color}; border-radius: 15px; text-align: center; margin-bottom: 15px; transition: all 0.2s ease-in-out; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
                    onmouseenter="this.style.transform='scale(1.05)'; this.style.borderColor='#999999';"
                    onmouseleave="this.style.transform='scale(1)'; this.style.borderColor='{bg_color}';">
                    <h1 style='font-size: 80px; font-family: Arial, sans-serif; margin:0; color: #333;'>{char}</h1>
                    <p style='margin:0; text-align: center; color: #555; font-weight: bold;'>{name.capitalize()}</p>
                </div>
                <audio id="audio-{safe_name_id}" src="data:audio/mp3;base64,{audio_b64}"></audio>
                """
                st.markdown(html_card, unsafe_allow_html=True)

# =================================================================
# --- Halaman 2: Kuis Tulisan Tangan ---
# =================================================================
elif menu_choice == "✍️ Kuis Tulisan Tangan":
    st.title("✒️ Kuis Tulisan Tangan Hijaiyah")
    st.write(
        "Silakan gambar satu huruf hijaiyah pada kanvas di bawah ini, lalu klik tombol 'Prediksi' untuk melihat dan mendengar hasilnya.")

    if not class_names or model is None:
        st.warning("Fitur kuis tidak dapat dijalankan karena model atau dataset tidak ditemukan.")
    else:
        stroke_width = st.slider("Tebal Kuas: ", 5, 25, 10)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Area Menggambar")
            canvas_result = st_canvas(stroke_width=stroke_width, stroke_color="#000000", background_color="#FFFFFF",
                                      height=300, width=300, drawing_mode="freedraw", key="canvas")
        if st.button("Prediksi Tulisan Tangan"):
            if canvas_result.image_data is not None:
                image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                with st.spinner("Sedang menganalisis tulisan Anda..."):
                    prediction = predict(image, model)
                    predicted_class_index = np.argmax(prediction)
                    predicted_class_name = class_names[predicted_class_index]
                    confidence = np.max(prediction)
                    with col2:
                        st.subheader("Hasil Prediksi")
                        st.success(f"Prediksi: **{predicted_class_name.capitalize()}**")
                        st.info(f"Tingkat Keyakinan: **{confidence:.2%}**")
                        arabic_character = HIJAIYAH_MAP.get(predicted_class_name.lower())
                        if arabic_character:
                            play_sound(arabic_character)
                        with st.expander("Lihat Detail Probabilitas"):
                            st.dataframe({"Huruf": class_names, "Probabilitas": [f"{p:.2%}" for p in prediction[0]]})
            else:
                with col2:
                    st.warning("Mohon gambar sesuatu di kanvas terlebih dahulu.")

# =================================================================
# --- Halaman 3: Tebak Huruf Hijaiyah (BARU) ---
# =================================================================
elif menu_choice == "🤔 Tebak Huruf Hijaiyah":
    st.title("🤔 Tebak Huruf Hijaiyah")
    st.info("Lihat huruf di layar, lalu pilih jawaban yang benar!")


    def setup_new_question():
        """Memilih huruf acak dan 3 jawaban salah, lalu menyimpannya di session state."""
        correct_name, correct_char = random.choice(list(HIJAIYAH_MAP.items()))
        st.session_state.correct_char = correct_char
        st.session_state.correct_name = correct_name

        all_names = list(HIJAIYAH_MAP.keys())
        all_names.remove(correct_name)
        wrong_options = random.sample(all_names, 3)

        options = wrong_options + [correct_name]
        random.shuffle(options)
        st.session_state.options = options
        st.session_state.answered = False


    # Inisialisasi game jika belum dimulai
    if 'correct_name' not in st.session_state:
        setup_new_question()

    # Tampilkan huruf yang harus ditebak
    st.markdown(f"""
    <div style="background-color: #F0F2F6; border-radius: 15px; padding: 30px; text-align: center; margin-top: 20px;">
        <h1 style='font-size: 200px; font-family: "Noto Naskh Arabic", "Arial", sans-serif; margin:0;'>{st.session_state.correct_char}</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Tampilkan pilihan jawaban
    cols = st.columns(4)
    for i, option in enumerate(st.session_state.options):
        if cols[i].button(option.capitalize(), key=f"option_{i}", use_container_width=True):
            st.session_state.selected_answer = option
            st.session_state.answered = True
            st.rerun()  # Jalankan ulang skrip untuk menampilkan hasil

    # Tampilkan hasil setelah pengguna menjawab
    if st.session_state.get('answered', False):
        is_correct = (st.session_state.selected_answer == st.session_state.correct_name)

        if is_correct:
            st.success("🎉 Benar! Kamu Hebat!")
            # Buat dan putar suara "Hore"
            try:
                tts = gTTS(text="Hore, benar!", lang='id')
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                st.audio(audio_fp, format='audio/mp3', start_time=0, autoplay=True)
            except Exception as e:
                st.warning(f"Tidak bisa memutar suara apresiasi: {e}")
        else:
            st.error(f"Yah, salah. Jawaban yang benar adalah '{st.session_state.correct_name.capitalize()}'.")

        # Tampilkan kunci jawaban dan putar pelafalannya
        st.info(f"Ini adalah huruf **{st.session_state.correct_name.capitalize()}**.")
        play_sound(st.session_state.correct_char)

        # Tombol untuk soal berikutnya
        if st.button("Lanjut ke Soal Berikutnya"):
            setup_new_question()
            st.rerun()
