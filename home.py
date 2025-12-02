# Home.py
import streamlit as st

# Set page config with custom theme
st.set_page_config(
    page_title="AuraSense",
    layout="centered",
    page_icon="🏢"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #e6f2ff;
        }
        .stApp {
            background-color: #e6f2ff;
        }
        .header {
            background-color: #003366;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .content {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="header"><h1>🏢 AuraSense Presence</h1></div>', unsafe_allow_html=True)

# Main Content
with st.container():
    st.markdown('<div class="content">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3652/3652191.png", width=150)
    with col2:
        st.markdown("""
        ### Selamat Datang di AuraSense
        Sistem ini menggunakan teknologi **Face Recognition berbasis AI (ResNet50)** 
        dengan penyimpanan **Vector Database (Qdrant)**.
        """)
    
    st.markdown("---")
    st.subheader("Menu Utama")
    st.markdown("""
    Silakan pilih menu di sidebar (sebelah kiri):
    * **Registrasi Wajah:** (Khusus Admin) Untuk mendaftarkan karyawan baru dengan 8 pose.
    * **Absensi User:** (Akan dibuat selanjutnya) Untuk melakukan presensi harian.
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Info box at the bottom
st.info("💡 Pastikan Anda memiliki akses internet stabil untuk terhubung ke Cloud Database.")
