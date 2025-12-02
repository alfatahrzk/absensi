import streamlit as st
from supabase import create_client, Client
import hashlib

class AdminAuth:
    def __init__(self):
        try:
            url = st.secrets["supabase"]["URL"]
            key = st.secrets["supabase"]["KEY"]
            self.supabase: Client = create_client(url, key)
        except Exception as e:
            st.error(f"Error Auth Init: {e}")

    def _hash_password(self, password):
        """Mengubah password menjadi kode acak (SHA256) agar aman"""
        return hashlib.sha256(password.encode()).hexdigest()

    def login(self, username, password):
        """Cek apakah username & password cocok"""
        try:
            hashed_pw = self._hash_password(password)
            
            response = self.supabase.table("admins")\
                .select("*")\
                .eq("username", username)\
                .eq("password", hashed_pw)\
                .execute()
            
            # Jika ada data yang cocok, return True
            if response.data:
                return True
            return False
        except Exception as e:
            st.error(f"Login Error: {e}")
            return False

    def add_admin(self, username, password):
        """Menambah admin baru"""
        try:
            hashed_pw = self._hash_password(password)
            data = {"username": username, "password": hashed_pw}
            self.supabase.table("admins").insert(data).execute()
            return True, "Berhasil"
        except Exception as e:
            # Error biasanya karena username kembar (Primary Key violation)
            return False, f"Gagal: Username '{username}' mungkin sudah ada."

    def delete_admin(self, username):
        """Menghapus admin"""
        try:
            self.supabase.table("admins").delete().eq("username", username).execute()
            return True
        except Exception as e:
            return False

    def get_all_admins(self):
        """Mengambil daftar admin"""
        try:
            response = self.supabase.table("admins").select("username, created_at").execute()
            return response.data
        except:
            return []