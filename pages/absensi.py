import streamlit as st
import cv2
import numpy as np
from haversine import haversine, Unit
from datetime import datetime
import pandas as pd
import time

# Import Backend
from core.engines import FaceEngine
from core.database import VectorDB
from core.logger import AttendanceLogger
from core.locator import LocationService # <--- IMPORT BARU

st.set_page_config(page_title="Absensi Karyawan", layout="centered")

# --- INISIALISASI BACKEND ---
@st.cache_resource
def get_backends():
    return FaceEngine(), VectorDB(), AttendanceLogger(), LocationService()

engine, db, logger, locator = get_backends()

# --- FUNGSI LOGIKA LOKASI ---
def check_location(user_lat, user_lon, office_lat, office_lon, radius_km):
    distance = haversine((user_lat, user_lon), (office_lat, office_lon), unit=Unit.KILOMETERS)
    return distance, distance <= radius_km

# --- HALAMAN UTAMA ---
st.title("📸 Absensi Harian")

# 1. DETEKSI LOKASI OTOMATIS
OFFICE_COORD = (-7.3244,112.7974) # Koordinat Kantor (Surabaya)
MAX_RADIUS_KM = 0.5

st.subheader("📍 Deteksi Lokasi")

with st.spinner("Mencari koordinat Anda..."):
    user_lat, user_lon, source = locator.get_coordinates()

# TAMPILKAN STATUS LOKASI
if user_lat is None:
    st.warning("⚠️ Sedang meminta izin lokasi browser... (Klik 'Allow' jika muncul pop-up)")
    st.info("Jika tidak muncul, pastikan GPS aktif. Sistem akan mencoba menggunakan IP Address.")
    st.stop() # Stop sampai lokasi ditemukan

else:
    # Hitung Jarak
    distance, is_in_radius = check_location(user_lat, user_lon, *OFFICE_COORD, MAX_RADIUS_KM)
    
    with st.spinner("Mendeteksi nama jalan..."):
        current_address = locator.get_address(user_lat, user_lon)
    
    # Tampilkan di UI
    st.info(f"📍 Posisi anda saat ini di **{current_address}**") # Biar user tau dia terdeteksi di mana

    col1, col2, col3 = st.columns(3)
    # ... metric latitude longitude sama ...
    
    if is_in_radius:
        st.success(f"✅ Lokasi Valid! Jarak: {distance:.3f} km")
    else:
        st.error(f"❌ Di Luar Kantor! Jarak: {distance:.3f} km")
        st.stop()

st.divider()

# 2. PILIH TIPE ABSENSI
absen_type = st.radio("Jenis Absensi:", ["Masuk", "Keluar"], horizontal=True)

# --- STATE MANAGEMENT (Agar tidak looping) ---
if 'berhasil_absen' not in st.session_state:
    st.session_state['berhasil_absen'] = None

# --- LOGIKA UI: TAMPILKAN HASIL ATAU KAMERA ---

if st.session_state['berhasil_absen'] is not None:
    # ... (Bagian Struk Sukses SAMA PERSIS dengan sebelumnya) ...
    user_data = st.session_state['berhasil_absen']
    
    st.balloons() 
    st.success(f"✅ Absensi Berhasil!")
    
    st.info(f"""
    **STRUK BUKTI KEHADIRAN**
    -------------------------
    👤 Nama   : {user_data['nama']}
    📅 Waktu  : {user_data['waktu']}
    📍 Jarak  : {user_data['jarak']} km
    🏠 Alamat : {user_data.get('alamat', '-')}
    -------------------------
    Data telah tersimpan di Cloud Database.
    """)
    
    if st.button("🔄 Kembali ke Kamera", type="primary"):
        st.session_state['berhasil_absen'] = None 
        st.rerun()

else:
    # 3. KAMERA INPUT
    # ... (Bagian Kamera Input SAMA PERSIS dengan sebelumnya) ...
    # (Copy dari kode sebelumnya, hanya pastikan variabel 'distance' diambil dari atas)
    
    img_file = st.camera_input("Scan Wajah Anda", key="absen_cam")

    if img_file is not None:
        bytes_data = img_file.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
        
        # Deteksi Koordinat Wajah (Fitur Kotak Hijau)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = engine.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            st.warning("⚠️ Wajah tidak terdeteksi.")
        else:
            x, y, w, h = faces[0]
            # Gambar Kotak
            img_with_box = cv_img.copy()
            cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
            st.image(img_with_box, channels="BGR", caption="Wajah Terdeteksi")
            
            face_crop = cv_img[y:y+h, x:x+w]
            
            with st.spinner("Mencocokkan biometrik..."):
                input_emb = engine.get_embedding(face_crop)
                found_user, score = db.search_user(input_emb, threshold=0.5)
                
                if found_user:
                    status_absen = absen_type
                    if logger.log_attendance(found_user, status_absen, distance, current_address):
                        
                        st.session_state['berhasil_absen'] = {
                            'nama': found_user,
                            'waktu': datetime.now().strftime('%H:%M:%S'),
                            'jarak': f"{distance:.3f}",
                            'alamat': current_address # Simpan juga di session buat struk
                        }
                        st.rerun()
                    else:
                        st.error("Gagal terhubung ke Database Log.")
                else:
                    st.error("❌ Wajah tidak dikenali!")