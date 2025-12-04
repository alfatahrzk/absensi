import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

from core.engines import FaceEngine 
from core.database import VectorDB
from core.config_manager import ConfigManager 
from core.logger import AttendanceLogger 
from core.admin_auth import AdminAuth # <--- IMPORT BARU

st.set_page_config(page_title="Dashboard Admin", layout="wide") 

# Custom CSS agar selaras dengan home.py dan Absensi.py
st.markdown("""
    <style>
        .main {
            background-color: #e6f2ff;
        }
        .stApp {
            background-color: #e6f2ff;
            color: #003366;
        }
        .header {
            background-color: #003366;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .content-box {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        /* Tab styling agar teks tidak menyatu dengan background */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #ffffff;
            color: #003366;
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #003366 !important;
            box-shadow: 0 -2px 4px rgba(0,0,0,0.1);
        }
        /* Make specific headers and labels dark blue */
        .stHeader {
            color: #003366 !important;
        }
        /* Custom class for dark blue text */
        .dark-blue-text {
            color: #003366 !important;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .stTextInput > div > div > input,
        .stTextInput > div > label {
            color: white !important;
        }
        .stButton>button, 
        .stDownloadButton>button,
        .stFormSubmitButton>button,
        button[data-testid="stBaseButton-secondaryFormSubmit"] {
            color: white !important;
            background-color: #003366 !important;
            border: 1px solid #002244 !important;
        }
        .stButton>button:hover,
        .stDownloadButton>button:hover,
        .stFormSubmitButton>button:hover,
        button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
            background-color: #002244 !important;
            border-color: #001122 !important;
        }
        .st-emotion-cache-zuyloh.emjbblw1[data-testid="stForm"] {
            padding: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- INISIALISASI ---
@st.cache_resource
def get_backends():
    return FaceEngine(), VectorDB(), ConfigManager(), AttendanceLogger(), AdminAuth()

engine, db, config_mgr, logger, auth = get_backends()

# --- LOGIN ADMIN (VERSI DATABASE) ---
if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False
if 'admin_name' not in st.session_state:
    st.session_state['admin_name'] = ""

if not st.session_state['is_admin']:
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.title("🔒 Admin Login System")
        
        st.markdown('<p class="dark-blue-text">Username</p>', unsafe_allow_html=True)
        form_user = st.text_input("", label_visibility="collapsed")
        st.markdown('<p class="dark-blue-text">Password</p>', unsafe_allow_html=True)
        form_pass = st.text_input("", type="password", label_visibility="collapsed")
        
        if st.button("Masuk", type="primary", use_container_width=True):
            if auth.login(form_user, form_pass):
                st.session_state['is_admin'] = True
                st.session_state['admin_name'] = form_user
                st.toast(f"Selamat datang, {form_user}!", icon="👋")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Username atau Password salah!")
    st.stop() 

# --- SIDEBAR LOGOUT ---
with st.sidebar:
    st.write(f"Login sebagai: **{st.session_state['admin_name']}**")
    if st.button("Logout"):
        st.session_state['is_admin'] = False
        st.rerun()

# Header utama selaras dengan halaman lain
st.markdown('<div class="header"><h1>⚙️ Dashboard Admin</h1></div>', unsafe_allow_html=True)

# BUAT 5 TAB MENU (TAMBAHAN SATU TAB BARU)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Registrasi Wajah", 
    "🎛️ Pengaturan Sistem", 
    "📊 Riwayat Absensi",
    "👥 Kelola Wajah",
    "🔑 Kelola Akun Admin" # <--- TAB BARU
])

# ====================================================
# TAB 1: REGISTRASI WAJAH (SAMA)
# ====================================================
with tab1:
    c_left, c_center, c_right = st.columns([1, 2, 1])
    with c_center:
        st.header("Pendaftaran Karyawan")
        st.markdown('<p class="dark-blue-text">Nama Karyawan Baru</p>', unsafe_allow_html=True)
        username = st.text_input("", label_visibility="collapsed")

        if 'reg_data' not in st.session_state: st.session_state['reg_data'] = [] 
        if 'step' not in st.session_state: st.session_state['step'] = 0

        instructions = [
            "😐 1. Wajah Datar (Netral)", "😁 2. Tersenyum Lebar",
            "↗️ 3. Hadap Serong Kanan", "↖️ 4. Hadap Serong Kiri",
            "⬆️ 5. Menghadap Atas", "⬇️ 6. Menghadap Bawah",
            "🤪 7. Miring Kanan", "🤪 8. Miring Kiri"
        ]
        total_steps = 8
        current = st.session_state['step']

        if current < total_steps:
            st.info(f"**Langkah {current + 1}/{total_steps}:** {instructions[current]}")
            st.markdown('<p class="dark-blue-text">Ambil Foto</p>', unsafe_allow_html=True)
            img_file = st.camera_input("", key=f"cam_{current}", label_visibility="collapsed")
            
            if img_file:
                bytes_data = img_file.getvalue()
                raw_cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
                cv_img = cv2.flip(raw_cv_img, 1)
                coords = engine.extract_face_coords(cv_img)
                
                if coords is None:
                    st.warning("⚠️ Wajah tidak terdeteksi.")
                else:
                    x, y, w, h = coords
                    img_box = cv_img.copy()
                    cv2.rectangle(img_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    
                    face_crop = cv_img[y:y+h, x:x+w]
                    emb = engine.get_embedding(face_crop)
                    
                    st.session_state['reg_data'].append(emb)
                    st.session_state['step'] += 1
                    st.toast(f"Pose {current+1} Tersimpan", icon="✅")
                    time.sleep(0.5)
                    st.rerun()
            st.progress(current / total_steps)
        else:
            st.success("✅ Data Lengkap!")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("💾 Simpan ke Database", type="primary", use_container_width=True):
                    if not username:
                        st.error("Nama wajib diisi!")
                    else:
                        with st.spinner("Upload ke Qdrant..."):
                            master_emb = engine.calculate_average_embedding(st.session_state['reg_data'])
                            success = db.save_user(username, master_emb)
                            if success:
                                st.balloons()
                                st.success(f"Sukses! {username} terdaftar.")
                                st.session_state['reg_data'] = []
                                st.session_state['step'] = 0
                                st.rerun()
                            else:
                                st.error("Gagal simpan.")
            with btn_col2:
                if st.button("🔄 Ulangi", use_container_width=True):
                    st.session_state['reg_data'] = []
                    st.session_state['step'] = 0
                    st.rerun()

# ====================================================
# TAB 2: PENGATURAN SISTEM (SAMA)
# ====================================================
with tab2:
    st.header("Konfigurasi Global")
    @st.cache_data(ttl=10)
    def load_config(): return config_mgr.get_config()
    current_conf = load_config()
    
    with st.form("edit_config"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📍 Lokasi Kantor")
            st.markdown('<p class="dark-blue-text">Latitude</p>', unsafe_allow_html=True)
            lat = st.number_input("", value=float(current_conf.get('office_lat', -7.25)), format="%.6f", label_visibility="collapsed")
            st.markdown('<p class="dark-blue-text">Longitude</p>', unsafe_allow_html=True)
            lon = st.number_input("", value=float(current_conf.get('office_lon', 112.75)), format="%.6f", label_visibility="collapsed")
            st.markdown('<p class="dark-blue-text">Radius (km)</p>', unsafe_allow_html=True)
            rad = st.number_input("", value=float(current_conf.get('radius_km', 0.5)), step=0.1, label_visibility="collapsed")
        with col2:
            st.subheader("🧠 Sensitivitas AI")
            st.markdown('<p class="dark-blue-text">Threshold Wajah</p>', unsafe_allow_html=True)
            face_thresh = st.slider("", 0.0, 1.0, float(current_conf.get('face_threshold', 0.70)), 0.01, label_visibility="collapsed")

        if st.form_submit_button("Simpan Konfigurasi", use_container_width=True):
            if config_mgr.save_config(lat, lon, rad, face_thresh, 0.0): 
                load_config.clear()
                st.success("✅ Tersimpan!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Gagal update.")

# ====================================================
# TAB 3: RIWAYAT ABSENSI
# ====================================================
with tab3:
    st.header("📊 Data Log Absensi")
    if st.button("🔄 Refresh Log"):
        st.cache_data.clear()
        st.rerun()
    
    df_logs = logger.get_logs(limit=100)
    if not df_logs.empty:
        st.dataframe(df_logs, use_container_width=True, hide_index=True)
        csv = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name="logs.csv", mime='text/csv')
    else:
        st.info("Belum ada data.")

# ====================================================
# TAB 4: KELOLA WAJAH (UPGRADED)
# ====================================================
with tab4:
    st.header("👥 Manajemen Database Wajah")
    
    # 1. Pilih User
    with st.spinner("Mengambil daftar karyawan..."):
        users_list = db.get_all_users()
    
    if not users_list:
        st.warning("Database kosong.")
    else:
        col_sel, col_info = st.columns([1, 2])
        
        with col_sel:
            selected_user = st.selectbox("Pilih Karyawan:", users_list)
        
        with col_info:
            st.info("Pilih nama karyawan untuk melihat detail data wajah yang tersimpan.")

        st.divider()
        
        if selected_user:
            # 2. Ambil Variasi Wajah
            variations = db.get_user_variations(selected_user)
            
            st.subheader(f"Data Wajah: {selected_user}")
            st.write(f"Total Variasi Tersimpan: **{len(variations)} titik**")
            
            if len(variations) > 0:
                # Tampilkan Tabel Variasi
                df_var = pd.DataFrame(variations)
                df_var.columns = ["ID Database (UUID)", "Waktu Direkam"]
                
                # Tampilkan tabel biar keren
                st.dataframe(df_var, use_container_width=True)
                
                # 3. Hapus Variasi Spesifik (Misal yang terbaru error)
                st.markdown("##### 🗑️ Hapus Variasi Tertentu")
                
                # Bikin dictionary biar user milih berdasarkan tanggal, bukan ID yang ribet
                options_map = {f"{v['created_at']} (ID: {v['id'][:8]}...)": v['id'] for v in variations}
                
                selected_var_label = st.selectbox("Pilih data yang ingin dihapus:", list(options_map.keys()))
                selected_var_id = options_map[selected_var_label]
                
                if st.button(f"Hapus Data Tanggal {selected_var_label.split('(')[0]}"):
                    with st.spinner("Menghapus..."):
                        if db.delete_point(selected_var_id):
                            st.success("✅ Data berhasil dihapus!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Gagal menghapus.")
            
            st.divider()
            
            # 4. Hapus User Total (Tombol Bahaya)
            with st.expander("🚨 ZONA BAHAYA: Hapus Karyawan Permanen"):
                st.warning(f"Tindakan ini akan menghapus SEMUA data wajah milik **{selected_user}**.")
                if st.button(f"Hapus User {selected_user} Selamanya", type="primary"):
                    if db.delete_user(selected_user):
                        st.success(f"User {selected_user} telah dihapus total.")
                        time.sleep(1)
                        st.rerun()

# ====================================================
# TAB 5: KELOLA ADMIN (FITUR BARU)
# ====================================================
with tab5:
    st.header("🔑 Manajemen Akun Admin")
    
    # Bagian 1: List Admin
    st.subheader("Daftar Admin Terdaftar")
    admins = auth.get_all_admins()
    
    if admins:
        df_admin = pd.DataFrame(admins)
        st.dataframe(df_admin, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Bagian 2: Tambah Admin Baru
    col_add, col_rem = st.columns(2)
    
    with col_add:
        st.subheader("➕ Tambah Admin Baru")
        with st.form("add_admin_form"):
            st.markdown('<p class="dark-blue-text">Username Baru</p>', unsafe_allow_html=True)
            new_user = st.text_input("", label_visibility="collapsed")
            st.markdown('<p class="dark-blue-text">Password Baru</p>', unsafe_allow_html=True)
            new_pass = st.text_input("", type="password", label_visibility="collapsed")
            submitted = st.form_submit_button("Tambah Admin")
            
            if submitted:
                if new_user and new_pass:
                    success, msg = auth.add_admin(new_user, new_pass)
                    if success:
                        st.success(f"Admin {new_user} berhasil ditambahkan!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Isi username dan password.")
    
    # Bagian 3: Hapus Admin
    with col_rem:
        st.subheader("⛔ Hapus Admin")
        
        # Ambil list username saja
        admin_usernames = [a['username'] for a in admins] if admins else []
        
        # Jangan izinkan hapus diri sendiri (cegah bunuh diri akun)
        current_user = st.session_state.get('admin_name', '')
        valid_to_delete = [u for u in admin_usernames if u != current_user]
        
        if valid_to_delete:
            st.markdown('<p class="dark-blue-text">Pilih Admin untuk dihapus:</p>', unsafe_allow_html=True)
            del_target = st.selectbox("", valid_to_delete, label_visibility="collapsed")
            
            if st.button("Hapus Admin Terpilih", type="primary"):
                if auth.delete_admin(del_target):
                    st.success(f"Admin {del_target} dihapus.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Gagal menghapus.")
        else:
            st.info("Tidak ada admin lain yang bisa dihapus (Anda tidak bisa menghapus diri sendiri).")
