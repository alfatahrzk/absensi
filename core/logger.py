import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import pandas as pd

class AttendanceLogger:
    def __init__(self):
        try:
            url = st.secrets["supabase"]["URL"]
            key = st.secrets["supabase"]["KEY"]
            self.supabase: Client = create_client(url, key)
        except Exception as e:
            st.error(f"Gagal konek Supabase: {e}")

    def log_attendance(self, name, status, location_dist, address, lat, lon, similarity, liveness, validation_status="Berhasil"):
        """Mencatat log ke Supabase"""
        try:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = {
                "nama": name,
                "status": status,
                "waktu_absen": now_str,
                "jarak": f"{location_dist:.4f}",
                "alamat": address,
                "verifikasi": "Wajah (Qdrant)",
                "koordinat": f"{lat}, {lon}",
                "skor_kemiripan": float(similarity),
                "skor_liveness": float(liveness),
                "status_validasi": validation_status 
            }
            self.supabase.table("logs").insert(data).execute()
            return True
        except Exception as e:
            st.error(f"Gagal simpan log: {e}")
            return False

    # --- FITUR BARU: AMBIL LOG UNTUK ADMIN ---
    def get_logs(self, limit=100):
        """
        Mengambil 100 log terakhir, diurutkan dari yang terbaru.
        """
        try:
            # Select * from logs order by id desc limit 100
            response = self.supabase.table("logs")\
                .select("*")\
                .order("id", desc=True)\
                .limit(limit)\
                .execute()
            
            data = response.data
            
            if data:
                return pd.DataFrame(data)
            else:
                return pd.DataFrame() # Return DF kosong jika tidak ada data
                
        except Exception as e:
            st.error(f"Gagal mengambil log: {e}")
            return pd.DataFrame()