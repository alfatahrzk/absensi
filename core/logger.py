# core/logger.py
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
from datetime import datetime

class AttendanceLogger:
    def __init__(self):
        # Membuat koneksi ke Google Sheets
        try:
            self.conn = st.connection("gsheets", type=GSheetsConnection)
        except Exception as e:
            st.error(f"Error init koneksi: {e}")

    def log_attendance(self, name, status, location_dist, address):
        try:
            sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
            
            # Update usecols jadi 0-5 (karena ada 6 kolom sekarang)
            existing_data = self.conn.read(spreadsheet=sheet_url, ttl=0, usecols=list(range(6)))
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_row = pd.DataFrame([{
                "Waktu": now,
                "Nama": name,
                "Status": status,
                "Lokasi (km)": f"{location_dist:.4f}",
                "Verifikasi": "Wajah (Qdrant)",
                "Alamat": address  # <--- DATA BARU
            }])
            
            if existing_data.empty:
                updated_data = new_row
            else:
                updated_data = pd.concat([existing_data, new_row], ignore_index=True)
            
            self.conn.update(spreadsheet=sheet_url, data=updated_data)
            return True
            
        except Exception as e:
            st.error(f"Gagal menyimpan log: {e}")
            return False