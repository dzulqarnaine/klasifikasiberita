import streamlit as st
import numpy as np
import pickle
from tensorflow import keras
from PIL import Image
from streamlit_option_menu import option_menu

# Muat model
model = keras.models.load_model('my_model.h5')

# Muat tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Muat encoder
with open('encoder.pickle', 'rb') as handle:
    encoder = pickle.load(handle)

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Gambar",
    page_icon=":📰:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Buat menu utama di sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Menu Utama",
        options=["Tentang", "Klasifikasi Berita"],
        icons=["house", "book"],
        menu_icon="cast",
        default_index=0,
    )

# Fungsi untuk Halaman 1 (Penjelasan Kelas Gambar)
def halaman_penjelasan():
    # Tampilkan judul di tengah halaman
    st.markdown("""
    <div style="text-align: center;">
        <h1>KLASIFIKASI BERITA</h1>
    </div>
    """, unsafe_allow_html=True)

    # Path ke gambar
    image_paths = [
       "bbc.jpeg"
    ]
    
    descriptions = [
        """BBC (British Broadcasting Corporation) adalah salah satu media penyiaran terbesar dan paling kredibel di dunia, yang menyediakan berita berkualitas tinggi dengan cakupan global. Berita-berita yang dihasilkan BBC mencakup berbagai topik seperti bisnis, teknologi, politik, olahraga, dan hiburan, yang ditulis dengan gaya profesional dan informatif. Berita-berita ini tidak hanya menjadi sumber informasi penting bagi masyarakat umum, tetapi juga menjadi rujukan dalam analisis media dan penelitian akademis. Dengan volume berita yang terus meningkat, pengelolaan dan pengklasifikasian konten berita secara otomatis menjadi semakin penting untuk mempermudah akses dan analisis."""
    ]

    # Gambar baru yang ingin ditambahkan di bawah deskripsi
    new_image_path = "nlp.jpg"  

    # Tampilkan gambar dan penjelasan menggunakan layout kolom
    for i in range(len(image_paths)):
        image = Image.open(image_paths[i])
        
        # Mengatur kolom dengan lebar yang sama
        col1 = st.columns(1)[0]  # Menggunakan satu kolom untuk gambar dan teks
        
        with col1:
            # Menampilkan gambar dengan ukuran yang sesuai
            st.image(image, use_container_width=True)  # Atur ukuran gambar agar sesuai dengan kolom
            
            # Menampilkan deskripsi di bawah gambar dengan justify
            st.markdown(f"<p style='text-align: justify;'>{descriptions[i]}</p>", unsafe_allow_html=True)

            # Menampilkan gambar baru dan teks tambahan secara bersebelahan
            col2, col3 = st.columns(2)  # Dua kolom untuk gambar dan teks tambahan
            
            with col2:
                # Menampilkan gambar baru
                new_image = Image.open(new_image_path)
                st.image(new_image, use_container_width=True)  # Atur ukuran gambar baru agar sesuai dengan kolom
            
            with col3:
                # Menampilkan teks tambahan
                additional_text = """
                Dataset BBC menawarkan solusi dengan menyediakan teks berita yang diklasifikasikan ke dalam lima kategori utama: business, tech, politic, sport, dan entertainment. Setiap kategori mencerminkan tema spesifik, seperti perkembangan pasar keuangan dalam kategori bisnis atau isu teknologi mutakhir dalam kategori teknologi. Klasifikasi ini memberikan dasar untuk pengembangan model pembelajaran mesin yang mampu mengelompokkan berita secara otomatis, sehingga memungkinkan aplikasi seperti portal berita pintar atau sistem rekomendasi konten berbasis minat. Dengan pendekatan ini, dataset BBC menjadi sumber yang relevan untuk eksplorasi NLP dan pengembangan solusi cerdas di era informasi digital.
                """
                st.markdown(f"<p style='text-align: justify;'>{additional_text}</p>", unsafe_allow_html=True)

# Fungsi untuk Halaman 2 (Klasifikasi Berita)
def halaman_klasifikasi():
    st.markdown("<h1 style='text-align: center;'>Klasifikasi Berita</h1>", unsafe_allow_html=True)
    input_text = st.text_area("Masukkan teks berita yang ingin diklasifikasikan (English):")

    if st.button("Klasifikasikan"):
        input_matrix = tokenizer.texts_to_matrix([input_text])
        prediction = model.predict(input_matrix)
        predicted_label_index = np.argmax(prediction)
        predicted_label = encoder.inverse_transform([predicted_label_index])
        st.markdown(f"<h3 style='text-align: center;'>Hasil Klasifikasi: {predicted_label[0]}</h3>", unsafe_allow_html=True)

# Menampilkan halaman berdasarkan pilihan
if selected == "Tentang":
    halaman_penjelasan()
elif selected == "Klasifikasi Berita":
    halaman_klasifikasi()