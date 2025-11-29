import streamlit as st
import cv2
import numpy as np
from core.engines import FaceEngine
from core.database import VectorDB
from core.config_manager import ConfigManager # <--- Import Baru

st.set_page_config(page_title="Halaman Admin", layout="centered")

# --- LOGIN ADMIN ---
if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False

if not st.session_state['is_admin']:
    st.title("🔒 Admin Login")
    pwd = st.text_input("Masukkan Password Admin", type="password")
    if st.button("Login"):
        if pwd == "admin123":
            st.session_state['is_admin'] = True
            st.rerun()
        else:
            st.error("Password Salah!")
    st.stop()

# --- INISIALISASI ---
@st.cache_resource
def get_resources():
    return FaceEngine(), VectorDB(), ConfigManager()

engine, db, config_mgr = get_resources()

# --- FUNGSI WRAPPER DENGAN CACHE (INI SOLUSINYA) ---
@st.cache_data(ttl=600) # Simpan di memori selama 600 detik (10 menit)
def load_config_data():
    return config_mgr.get_config()

def clear_config_cache():
    load_config_data.clear()

st.title("⚙️ Dashboard Admin")

# BUAT TAB MENU
tab1, = st.tabs(["📍 Lokasi Kantor"])



# =========================================
# TAB 2: PENGATURAN LOKASI (FITUR BARU)
# =========================================
with tab1:
    st.header("Pengaturan Titik Absensi")
    
    # Gunakan spinner agar user tau sedang loading (hanya muncul saat cache habis)
    with st.spinner("Memuat konfigurasi dari Cloud..."):
        # PANGGIL FUNGSI YANG SUDAH DI-CACHE
        current_conf = load_config_data()
    
    st.info(f"Lokasi saat ini: {current_conf.get('office_lat')}, {current_conf.get('office_lon')} (Radius: {current_conf.get('radius_km')} km)")
    
    # Form Edit
    with st.form("edit_lokasi"):
        col1, col2 = st.columns(2)
        with col1:
            # Gunakan .get() dengan default value untuk mencegah error jika sheet kosong
            val_lat = float(current_conf.get('office_lat', -7.2575))
            new_lat = st.number_input("Latitude Kantor", value=val_lat, format="%.6f")
        with col2:
            val_lon = float(current_conf.get('office_lon', 112.7521))
            new_lon = st.number_input("Longitude Kantor", value=val_lon, format="%.6f")
        
        val_rad = float(current_conf.get('radius_km', 0.5))
        new_radius = st.number_input("Radius Toleransi (km)", value=val_rad, step=0.1, format="%.3f")
        
        submitted = st.form_submit_button("Simpan Perubahan")
        
        if submitted:
            # Simpan ke Google Sheets
            success = config_mgr.save_config(new_lat, new_lon, new_radius)
            
            if success:
                # PENTING: Hapus Cache lama agar data baru ter-load
                clear_config_cache()
                st.success("✅ Lokasi kantor berhasil diperbarui!")
                import time
                time.sleep(1)
                st.rerun()
            else:
                st.error("Gagal menyimpan ke Google Sheets.")

    # Preview Peta (Sekarang akan cepat karena data koordinatnya dari Cache)
    try:
        import pandas as pd
        map_data = pd.DataFrame({'lat': [new_lat], 'lon': [new_lon]})
        st.map(map_data, zoom=15)
    except Exception as e:
        st.warning(f"Gagal memuat peta: {e}")