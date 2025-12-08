import streamlit as st
from supabase import create_client, Client

class ConfigManager:
    def __init__(self):
        try:
            url = st.secrets["supabase"]["URL"]
            key = st.secrets["supabase"]["KEY"]
            self.supabase: Client = create_client(url, key)
        except Exception as e:
            st.error(f"Error Config Init: {e}")

    def get_config(self):
        """Ambil semua config dari tabel 'config' dan berikan nilai default baru."""
        try:
            # Select * from config
            response = self.supabase.table("config").select("*").execute()
            data = response.data
            
            # Convert List of Dicts ke Single Dict
            config_dict = {item['key']: item['value'] for item in data}
            
            # --- TAMBAHAN NILAI DEFAULT BARU ---
            return {
                # 5 Parameter Lama
                "office_lat": float(config_dict.get("office_lat", -7.2575)),
                "office_lon": float(config_dict.get("office_lon", 112.7521)),
                "radius_km": float(config_dict.get("radius_km", 0.5)),
                "face_threshold": float(config_dict.get("face_threshold", 0.70)),
                "liveness_threshold": float(config_dict.get("liveness_threshold", 0.0)), # Dibuat 0.0 karena fitur dihapus
                
                # 3 Parameter Jam Kerja Baru
                "start_time": config_dict.get("start_time", "08:00"),               # Jam Masuk Standar
                "late_tolerance_time": config_dict.get("late_tolerance_time", "09:00"), # Batas Terlambat
                "cutoff_time": config_dict.get("cutoff_time", "10:00"),             # Batas Akhir Absensi
            }
            
        except Exception:
            # Jika Supabase gagal, berikan semua default
            return {
                "office_lat": -7.2575,
                "office_lon": 112.7521,
                "radius_km": 0.5,
                "face_threshold": 0.70,
                "liveness_threshold": 0.0,
                "start_time": "08:00",
                "late_tolerance_time": "09:00",
                "cutoff_time": "10:00",
            }

    def save_config(self, lat, lon, radius, face_thresh, start_time, late_tolerance_time, cutoff_time):
        """Update 7 parameter config utama (Liveness dihapus dari argumen)"""
        try:
            # Kita set liveness_threshold manual ke 0.0 agar fungsi lama tetap aman
            liveness_thresh = 0.0 
            
            updates = [
                {"key": "office_lat", "value": str(lat)},
                {"key": "office_lon", "value": str(lon)},
                {"key": "radius_km", "value": str(radius)},
                {"key": "face_threshold", "value": str(face_thresh)}, 
                {"key": "liveness_threshold", "value": str(liveness_thresh)},
                
                # --- TAMBAHAN PARAMETER JAM KERJA ---
                {"key": "start_time", "value": str(start_time)}, 
                {"key": "late_tolerance_time", "value": str(late_tolerance_time)},
                {"key": "cutoff_time", "value": str(cutoff_time)}
            ]
            
            # Supabase upsert (Insert or Update)
            self.supabase.table("config").upsert(updates).execute()
            return True
        except Exception as e:
            st.error(f"Gagal update config: {e}")
            return False