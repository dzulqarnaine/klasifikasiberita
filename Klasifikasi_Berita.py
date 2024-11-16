import streamlit as st
import numpy as np
import pickle
from tensorflow import keras
from PIL import Image
from streamlit_option_menu import option_menu

# Muat model
model = keras.models.load_model('model.h5')

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
        <h1>Penjelasan Dataset</h1>
    </div>
    """, unsafe_allow_html=True)

    # Path ke gambar
    image_paths = [
    "Train Data (224).jpg",
    "Train Data (83).jpg",
    "Train Data (87).jpg",
    "Train Data (99).jpg"
]
    
    descriptions = [
        """Pengamatan visual yang cermat terhadap keseluruhan gambar tidak
        memberikan indikasi adanya tanda-tanda kebakaran yang biasanya 
        tampak dalam situasi darurat, seperti jejak hangus pada permukaan,
        bekas-bekas material yang terbakar, atau adanya asap tipis yang
        membubung dari area tertentu. Selain itu, tidak ada tanda-tanda
        kehadiran sumber panas yang tidak biasa, seperti kilatan cahaya atau
        perubahan warna yang dapat menandakan suhu tinggi. Seluruh elemen yang
        terlihat dalam gambar tampak normal dan tidak menunjukkan aktivitas yang
        berpotensi berbahaya. Oleh karena itu, dapat disimpulkan dengan tingkat 
        keyakinan yang tinggi bahwa kondisi di lokasi tersebut saat ini sepenuhnya
        aman dan tidak menunjukkan adanya ancaman kebakaran yang perlu diwaspadai.""",

        """Gambar ini menunjukkan keberadaan api yang mendominasi area tersebut, 
        dengan nyala api yang jelas terlihat serta indikasi bahwa api tersebut dapat
        berkembang dengan cepat jika tidak segera dikendalikan. Warna-warna cerah 
        seperti merah dan oranye mencolok mengindikasikan bahwa api berada dalam fase 
        aktif, dengan kemungkinan bahaya besar terhadap lingkungan sekitarnya jika tidak
        diatasi dengan segera.""",

        """Gambar ini secara jelas menunjukkan dominasi adanya asap yang tebal dan membubung
        ke udara, menciptakan suasana yang misterius dan mendalam. Asap tersebut tampak 
        menyelimuti area sekitarnya, dengan warna abu-abu gelap yang menunjukkan potensi adanya 
        kebakaran atau sumber panas lainnya di dekatnya. Kehadiran asap ini bisa menjadi indikasi 
        bahwa suatu proses pembakaran sedang berlangsung, baik itu berupa kebakaran kecil, 
        pembakaran sampah, atau mungkin bahkan aktivitas industri. Dengan demikian, gambaran 
        ini mengundang perhatian dan menimbulkan pertanyaan tentang asal-usul asap tersebut 
        dan potensi bahayanya terhadap lingkungan sekitar.""",

        """Gambar ini secara mencolok menunjukkan keberadaan api dan asap yang muncul secara 
        bersamaan, menciptakan pemandangan yang dramatis dan penuh ketegangan. Nyala api yang 
        berkobar dengan warna merah dan oranye yang mencolok tampak menari-nari di antara kepulan 
        asap yang tebal dan gelap, yang membubung tinggi ke langit. Kombinasi antara api yang aktif 
        dan asap yang menyelimuti area tersebut memberikan indikasi bahwa suatu proses pembakaran 
        yang signifikan sedang berlangsung. Selain itu, asap yang terlihat dapat mengisyaratkan 
        potensi bahaya yang lebih besar, karena dapat menandakan adanya material yang terbakar atau 
        zat berbahaya di sekitarnya. Pemandangan ini tidak hanya menarik perhatian, tetapi juga 
        menimbulkan kekhawatiran mengenai keselamatan dan potensi ancaman yang ditimbulkan oleh 
        kebakaran yang mungkin terjadi di lokasi tersebut"""
    ]
    
    # Tampilkan gambar dan penjelasan menggunakan layout kolom
    for i in range(4):
        image = Image.open(image_paths[i])
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, use_container_width=True)
        
        with col2:
            st.write(descriptions[i])
    
    # st.write("SEMOGA MENANG")

# Fungsi untuk Halaman 2 (Klasifikasi Berita)
def halaman_klasifikasi():
    st.title("Klasifikasi Berita")
    input_text = st.text_area("Masukkan teks yang ingin diklasifikasikan:")

    if st.button("Klasifikasikan"):
        input_matrix = tokenizer.texts_to_matrix([input_text])
        prediction = model.predict(input_matrix)
        predicted_label_index = np.argmax(prediction)
        predicted_label = encoder.inverse_transform([predicted_label_index])
        st.write(f"Hasil Klasifikasi: {predicted_label[0]}")

# Menampilkan halaman berdasarkan pilihan
if selected == "Tentang":
    halaman_penjelasan()
elif selected == "Klasifikasi Berita":
    halaman_klasifikasi()