# core/logger.py
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

class AttendanceLogger:
    def __init__(self):
        # Setup Scope
        self.scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        # Load Credentials dari secrets.toml
        try:
            # Kita ambil dictionary creds dari secrets
            creds_dict = dict(st.secrets["connections"]["gsheets"])
            
            # Bersihkan key yang tidak perlu (karena format streamlit connections agak beda)
            # Library gspread butuh format dict json murni
            # Hapus key 'spreadsheet' karena itu bukan bagian dari creds JSON
            if "spreadsheet" in creds_dict:
                self.sheet_url = creds_dict.pop("spreadsheet")
            
            self.creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, self.scope)
            self.client = gspread.authorize(self.creds)
            
        except Exception as e:
            st.error(f"Gagal auth gspread: {e}")

    def log_attendance(self, name, status, location_dist, address):
        """
        Mode Turbo: Langsung append row tanpa read all data
        """
        try:
            # Buka Spreadsheet
            sheet = self.client.open_by_url(self.sheet_url).sheet1
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Data yang mau dimasukkan (List biasa, bukan DataFrame)
            row_data = [
                now,            # Kolom A: Waktu
                name,           # Kolom B: Nama
                status,         # Kolom C: Status
                f"{location_dist:.4f}", # Kolom D: Jarak
                "Wajah (Qdrant)",       # Kolom E: Verifikasi
                address         # Kolom F: Alamat
            ]
            
            # Perintah Sakti: Append Row (Hanya kirim data kecil ini ke server)
            sheet.append_row(row_data)
            return True
            
        except Exception as e:
            st.error(f"Gagal menyimpan log (Turbo): {e}")
            return False