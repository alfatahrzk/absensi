# core/config_manager.py
import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd

class ConfigManager:
    def __init__(self):
        # Inisialisasi koneksi (sama seperti logger)
        try:
            self.conn = st.connection("gsheets", type=GSheetsConnection)
            self.sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        except Exception as e:
            st.error(f"Error Config Init: {e}")

    def get_config(self):
        """
        Membaca Config dari Tab 'Config' di Google Sheets
        """
        try:
            # Baca Worksheet bernama 'Config'
            # ttl=0 agar selalu ambil data terbaru (real-time)
            df = self.conn.read(spreadsheet=self.sheet_url, worksheet="Config", ttl=0)
            
            # Konversi DataFrame ke Dictionary biar mudah dipakai
            # Contoh hasil: {'office_lat': -7.25, 'radius_km': 0.5}
            config_dict = dict(zip(df['Key'], df['Value']))
            
            return config_dict
            
        except Exception as e:
            # Fallback jika gagal baca sheet
            st.warning(f"Gagal baca config online, pakai default. Error: {e}")
            return {
                "office_lat": -7.2575,
                "office_lon": 112.7521,
                "radius_km": 0.5
            }

    def save_config(self, lat, lon, radius):
        """
        Menyimpan Config baru ke Tab 'Config'
        """
        try:
            # Siapkan DataFrame baru
            new_data = pd.DataFrame([
                {"Key": "office_lat", "Value": float(lat)},
                {"Key": "office_lon", "Value": float(lon)},
                {"Key": "radius_km", "Value": float(radius)}
            ])
            
            # Update Worksheet 'Config'
            self.conn.update(spreadsheet=self.sheet_url, worksheet="Config", data=new_data)
            return True
            
        except Exception as e:
            st.error(f"Gagal menyimpan config: {e}")
            return False