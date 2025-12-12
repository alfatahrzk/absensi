import streamlit as st
import cv2
import numpy as np
from haversine import haversine, Unit
from datetime import datetime, timedelta
import pandas as pd
import time
import pytz

# --- IMPORT LIBRARY ---
from core.engines import FaceEngine 
from core.database import VectorDB
from core.logger import AttendanceLogger
from core.config_manager import ConfigManager
from core.locator import LocationService 

st.set_page_config(
    page_title="Absensi Karyawan",
    layout="centered",
    page_icon="📸"
)

# Custom CSS
st.markdown("""
    <style>
        .main { background-color: #e6f2ff; }
        .stApp { background-color: #e6f2ff; }
        .header { background-color: #003366; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; }
        .stRadio > div { background-color: #003366; padding: 10px; border-radius: 10px; }
        .stRadio > div > label { color: #ffffff !important; font-weight: 600; font-size: 16px; }
        .stRadio > div[data-baseweb='radio'] > div:first-child > div:first-child > div { background-color: #003366 !important; border-color: #003366 !important; }
        .stRadio > div[data-baseweb='radio'] > div > div > label[data-testid="stMarkdownContainer"] > span { color: #ffffff !important; }
        .stRadio > div[data-baseweb='radio'] > div > div > label > span { color: #ffffff !important; }
        .stRadio > div[data-baseweb='radio'] > div > div > span { color: #ffffff !important; }
        .stRadio [data-testid="stMarkdownContainer"] span { color: #ffffff !important; }
        .stRadio div { color: #ffffff !important; }
        [data-testid="stHorizontalBlock"] { background-color: #004080; padding: 10px 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        [data-testid="stHorizontalBlock"] a { color: #ffffff !important; font-weight: 600; }
        .navbar {
            background-color: #004080;
            padding: 10px 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0 20px 0;
        }
        .navbar a {
            color: #ffffff !important;
            font-weight: 600;
            text-decoration: none;
            margin-right: 20px;
        }
        .navbar a:last-child {
            margin-right: 0;
        }
        .jam-kerja-info {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
        .jam-kerja-warning {
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
        .jam-kerja-success {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI WAKTU JAKARTA ---
def get_jakarta_time():
    """Mendapatkan waktu saat ini di zona waktu Jakarta (Asia/Jakarta)"""
    jakarta_tz = pytz.timezone('Asia/Jakarta')
    jakarta_time = datetime.now(jakarta_tz)
    return jakarta_time

# --- FUNGSI CEK JAM KERJA ---
def cek_jam_kerja_absensi(nama_karyawan, tipe_absensi, waktu_sekarang):
    """
    Fungsi untuk memeriksa apakah absensi keluar memenuhi minimal 8 jam kerja.
    
    Parameter:
    - nama_karyawan: nama karyawan yang absen
    - tipe_absensi: "Masuk" atau "Keluar"
    - waktu_sekarang: waktu sekarang dalam zona Jakarta
    
    Return:
    - status: True jika boleh absen, False jika ditolak
    - pesan: pesan informasi/error
    - jam_kerja: total jam kerja (jika absen keluar)
    - kurang_jam: jam yang masih kurang (jika ditolak)
    """
    
    # Batas minimal jam kerja
    MINIMAL_JAM_KERJA = 8
    
    if tipe_absensi == "Masuk":
        # Untuk absen masuk, selalu izinkan
        return {
            "status": True,
            "pesan": "✅ Silakan absen masuk",
            "jam_kerja": 0,
            "kurang_jam": 0
        }
    
    # Untuk absen keluar, cek data absen masuk hari ini
    # Dalam implementasi nyata, ini akan query database
    # Untuk demo, kita gunakan session_state
    
    if 'data_absensi_masuk' not in st.session_state:
        st.session_state['data_absensi_masuk'] = {}
    
    # Cek apakah karyawan sudah absen masuk hari ini
    if nama_karyawan not in st.session_state['data_absensi_masuk']:
        return {
            "status": False,
            "pesan": f"❌ ABSENSI KELUAR DITOLAK! Tidak ada data absen masuk hari ini untuk {nama_karyawan}",
            "jam_kerja": 0,
            "kurang_jam": MINIMAL_JAM_KERJA
        }
    
    # Ambil waktu masuk
    waktu_masuk = st.session_state['data_absensi_masuk'][nama_karyawan]
    
    # Hitung selisih waktu
    selisih = waktu_sekarang - waktu_masuk
    jam_kerja = selisih.total_seconds() / 3600  # Konversi ke jam
    
    # Bulatkan ke 1 desimal
    jam_kerja = round(jam_kerja, 1)
    
    # Cek apakah sudah memenuhi minimal 8 jam
    if jam_kerja < MINIMAL_JAM_KERJA:
        kurang_jam = MINIMAL_JAM_KERJA - jam_kerja
        kurang_jam = round(kurang_jam, 1)
        
        # Konversi ke jam dan menit untuk pesan yang lebih detail
        jam_kurang = int(kurang_jam)
        menit_kurang = int((kurang_jam - jam_kurang) * 60)
        
        return {
            "status": False,
            "pesan": f"❌ ABSENSI KELUAR DITOLAK! Jam kerja masih kurang {kurang_jam:.1f} jam ({jam_kurang} jam {menit_kurang} menit)",
            "jam_kerja": jam_kerja,
            "kurang_jam": kurang_jam
        }
    else:
        return {
            "status": True,
            "pesan": f"✅ Jam kerja sudah mencukupi: {jam_kerja:.1f} jam",
            "jam_kerja": jam_kerja,
            "kurang_jam": 0
        }

# --- FUNGSI SIMPAN DATA ABSEN MASUK ---
def simpan_absen_masuk(nama_karyawan, waktu_masuk):
    """Menyimpan data absen masuk untuk cek jam kerja nanti"""
    if 'data_absensi_masuk' not in st.session_state:
        st.session_state['data_absensi_masuk'] = {}
    
    st.session_state['data_absensi_masuk'][nama_karyawan] = waktu_masuk
    return True

# Header and Navigation Section
st.markdown("""
<div class="header">
    <h1>🏢 AuraSense Presence</h1>
    <nav class="navbar">
        <a href="home.py">🏠 Home</a>
        <a href="pages/Absensi.py">📸 Absen</a>
    </nav>
</div>
""", unsafe_allow_html=True)

# --- INISIALISASI BACKEND ---
@st.cache_resource
def get_backends():
    return FaceEngine(), VectorDB(), AttendanceLogger(), LocationService(), ConfigManager()

engine, db, logger, locator, config_mgr = get_backends()

# --- LOAD CONFIG GLOBAL ---
office_conf = config_mgr.get_config()
OFFICE_COORD = (float(office_conf.get('office_lat', -7.25)), float(office_conf.get('office_lon', 112.75)))
MAX_RADIUS_KM = float(office_conf.get('radius_km', 0.5))
THRESHOLD_VAL = float(office_conf.get('face_threshold', 0.70))

# --- FUNGSI LOKASI ---
def check_location(user_lat, user_lon, office_lat, office_lon, radius_km):
    distance = haversine((user_lat, user_lon), (office_lat, office_lon), unit=Unit.KILOMETERS)
    return distance, distance <= radius_km

# --- PROSES LOKASI ---
with st.spinner("Mencari lokasi Anda..."):
    user_lat, user_lon, source = locator.get_coordinates()

if user_lat is None:
    st.warning("⚠️ Sedang meminta izin lokasi browser...")
    st.stop()
else:
    distance, is_in_radius = check_location(user_lat, user_lon, *OFFICE_COORD, MAX_RADIUS_KM)
    
    with st.spinner("Mendeteksi nama jalan..."):
        current_address = locator.get_address(user_lat, user_lon)
    
    if is_in_radius:
        st.success(f"✅ Lokasi Valid! ({distance:.3f} km)")
    else:
        st.error(f"❌ Di Luar Kantor! Jarak: {distance:.3f} km")
        st.stop()

# 2. PILIH TIPE ABSENSI
st.markdown("""
<div style='text-align: left; margin-bottom: 20px;'>
    <h3 style='color: #003366; margin-bottom: 10px;'>Pilih Jenis Absensi</h3>
</div>
""", unsafe_allow_html=True)

absen_type = st.radio("", ["Masuk", "Keluar"], horizontal=True, label_visibility="collapsed")

# Tampilkan informasi jam kerja
st.markdown(f"""
<div class="jam-kerja-info">
    <h4 style="color: #856404; margin-top: 0;">⏰ INFORMASI JAM KERJA</h4>
    <p style="color: #856404; margin-bottom: 5px;"><strong>Minimal Jam Kerja:</strong> 8 jam</p>
    <p style="color: #856404; margin-bottom: 0;"><strong>Keterangan:</strong> Absen keluar hanya diizinkan setelah bekerja minimal 8 jam</p>
</div>
""", unsafe_allow_html=True)

if 'berhasil_absen' not in st.session_state:
    st.session_state['berhasil_absen'] = None

# --- UI LOGIC ---

if st.session_state['berhasil_absen'] is not None:
    user_data = st.session_state['berhasil_absen']
    
    with st.container():
        st.markdown(f"""
        <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; margin-bottom: 20px;'>
            <h3 style='color: #28a745; text-align: center; margin-top: 0;'>✅ Absensi Berhasil!</h3>
        </div>
        """, unsafe_allow_html=True)

        if 'foto_bukti' in user_data:
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2:
                st.image(user_data['foto_bukti'], channels="BGR", caption="Visualisasi AI", use_container_width=True)

        nama = user_data.get('nama', '-')
        waktu = user_data.get('waktu', '-')
        alamat = user_data.get('alamat', '-')
        
        # Tambahkan informasi jam kerja jika absen keluar
        if user_data.get('tipe') == "Keluar":
            jam_kerja = user_data.get('jam_kerja', '0.0')
            st.markdown(f"""
            <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4 style='color: #003366; text-align: center; margin-top: 0;'>STRUK BUKTI KEHADIRAN</h4>
                <hr style='border: 1px solid #003366; opacity: 0.3;'>
                <p style='color: #003366;'><strong>Nama</strong>   : {nama}</p>
                <p style='color: #003366;'><strong>Waktu Keluar</strong>  : {waktu} WIB</p>
                <p style='color: #003366;'><strong>Total Jam Kerja</strong> : {jam_kerja} jam</p>
                <p style='color: #003366;'><strong>Minimal Jam Kerja</strong> : 8 jam</p>
                <p style='color: #003366;'><strong>Lokasi</strong> : {alamat}</p>
                <hr style='border: 1px solid #003366; opacity: 0.3;'>
                <p style='color: #003366; text-align: center; margin-bottom: 0;'>Data tersimpan di Cloud.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h4 style='color: #003366; text-align: center; margin-top: 0;'>STRUK BUKTI KEHADIRAN</h4>
                <hr style='border: 1px solid #003366; opacity: 0.3;'>
                <p style='color: #003366;'><strong>Nama</strong>   : {nama}</p>
                <p style='color: #003366;'><strong>Waktu Masuk</strong>  : {waktu} WIB</p>
                <p style='color: #003366;'><strong>Minimal Jam Kerja</strong> : 8 jam</p>
                <p style='color: #003366;'><strong>Lokasi</strong> : {alamat}</p>
                <hr style='border: 1px solid #003366; opacity: 0.3;'>
                <p style='color: #003366; text-align: center; margin-bottom: 0;'>Data tersimpan di Cloud.</p>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("🔄 Kembali ke Kamera", type="primary"):
        st.session_state['berhasil_absen'] = None 
        st.rerun()

else:
    with st.container():
        st.markdown("<h3 style='color: #003366; text-align: center;'>Scan Wajah Anda</h3>", unsafe_allow_html=True)
        img_file = st.camera_input("", key="absen_cam")

    if img_file is not None:
        bytes_data = img_file.getvalue()
        
        raw_cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
        cv_img = cv2.flip(raw_cv_img, 1)
        
        coords = engine.extract_face_coords(cv_img)
        
        if coords is None:
            st.warning("⚠️ Wajah tidak terdeteksi.")
            st.image(cv_img, channels="BGR", caption="Gagal Deteksi")
        else:
            x, y, w, h = coords 
            face_crop = cv_img[y:y+h, x:x+w]

            with st.spinner("Mencocokkan biometrik..."):
                # Ambil waktu Jakarta sekarang
                waktu_jakarta = get_jakarta_time()
                waktu_str = waktu_jakarta.strftime('%H:%M:%S WIB')
                tanggal_str = waktu_jakarta.strftime('%d/%m/%Y')
                
                input_emb = engine.get_embedding(face_crop)
                found_user, score = db.search_user(input_emb, threshold=0.0)
                
                # --- CEK THRESHOLD WAJAH ---
                if found_user and score >= THRESHOLD_VAL:
                    
                    # --- CEK JAM KERJA UNTUK ABSEN KELUAR ---
                    if absen_type == "Keluar":
                        cek_jam_kerja = cek_jam_kerja_absensi(found_user, absen_type, waktu_jakarta)
                        
                        if not cek_jam_kerja["status"]:
                            # Tampilkan pesan error detail
                            st.markdown(f"""
                            <div class="jam-kerja-warning">
                                <h4 style="color: #721c24; margin-top: 0;">⚠️ ABSENSI DITOLAK</h4>
                                <p style="color: #721c24; margin-bottom: 5px;"><strong>Nama:</strong> {found_user}</p>
                                <p style="color: #721c24; margin-bottom: 5px;"><strong>Waktu Sekarang:</strong> {waktu_str}</p>
                                <p style="color: #721c24; margin-bottom: 5px;"><strong>Total Jam Kerja:</strong> {cek_jam_kerja['jam_kerja']:.1f} jam</p>
                                <p style="color: #721c24; margin-bottom: 0;"><strong>Alasan:</strong> {cek_jam_kerja['pesan']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Log kehadiran gagal karena jam kerja kurang
                            logger.log_attendance(
                                name=found_user,
                                status="Gagal",
                                location_dist=distance,
                                address=current_address,
                                lat=user_lat,
                                lon=user_lon,
                                similarity=score,
                                liveness=100.0,
                                validation_status=f"Gagal: Jam kerja kurang {cek_jam_kerja['kurang_jam']:.1f} jam"
                            )
                            st.stop()
                        else:
                            # Tampilkan informasi jam kerja cukup
                            st.markdown(f"""
                            <div class="jam-kerja-success">
                                <h4 style="color: #155724; margin-top: 0;">✅ JAM KERJA CUKUP</h4>
                                <p style="color: #155724; margin-bottom: 5px;"><strong>Nama:</strong> {found_user}</p>
                                <p style="color: #155724; margin-bottom: 5px;"><strong>Total Jam Kerja:</strong> {cek_jam_kerja['jam_kerja']:.1f} jam</p>
                                <p style="color: #155724; margin-bottom: 0;"><strong>Status:</strong> {cek_jam_kerja['pesan']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Gambar Kotak & Nama
                    img_result = cv_img.copy()
                    cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    label_text = f"{found_user} ({score:.2f})"
                    (w_text, h_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(img_result, (x, y - 35), (x + w_text, y), (0, 51, 102), -1)
                    cv2.putText(img_result, label_text, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # --- FITUR ADAPTIF ---
                    ADAPTIVE_THRESHOLD = THRESHOLD_VAL + 0.08 
                    
                    if score >= ADAPTIVE_THRESHOLD:
                        db.add_variation(found_user, input_emb)
                        st.toast(f"Data wajah {found_user} diperbarui otomatis! 🧠", icon="✨")

                    # Simpan data absen masuk untuk pengecekan jam kerja nanti
                    if absen_type == "Masuk":
                        simpan_absen_masuk(found_user, waktu_jakarta)

                    # Simpan Log
                    sukses = logger.log_attendance(
    name=found_user, 
    status=absen_type, 
    location_dist=distance, 
    address=current_address,
    lat=user_lat,
    lon=user_lon,
    similarity=score,
    liveness=100.0,
    validation_status="Berhasil"
)
                    
                    if sukses:
                        user_data = {
                            'nama': found_user,
                            'skor': f"{score:.4f}",
                            'waktu': waktu_str, 
                            'jarak': f"{distance:.3f}",
                            'alamat': current_address,
                            'foto_bukti': img_result,
                            'tipe': absen_type
                        }
                         
                        # Tambahkan informasi jam kerja jika absen keluar
                        if absen_type == "Keluar":
                            user_data['jam_kerja'] = f"{cek_jam_kerja['jam_kerja']:.1f}"
                        
                        st.session_state['berhasil_absen'] = user_data
                        st.rerun()
                    else:
                        st.error("Gagal terhubung ke Database Log.")
                else:
                    # Gagal wajah tidak dikenali
                    st.error(f"❌ Ditolak! Wajah tidak dikenali.\nHarap hubungi admin!")
                    logger.log_attendance(
                        name=f"{found_user} (Ditolak)" if found_user else "Unknown (Ditolak)",
                        status="Gagal",
                        location_dist=distance,
                        address=current_address,
                        lat=user_lat,
                        lon=user_lon,
                        similarity=score,
                        liveness=0.0,
                        validation_status="Gagal: Skor Rendah"
                    )