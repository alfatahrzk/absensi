# core/locator.py
import streamlit as st
from streamlit_js_eval import get_geolocation
import requests
from geopy.geocoders import Nominatim # <--- Import Baru

class LocationService:
    def __init__(self):
        # Inisialisasi Geocoder (Wajib pakai user_agent unik)
        self.geolocator = Nominatim(user_agent="skripsi_absensi_app_v1")

    def get_coordinates(self):
        """
        Mencoba mendapatkan lokasi: GPS Browser -> IP Address
        """
        lat, lon, source = None, None, None

        # 1. GPS BROWSER
        loc_data = get_geolocation(component_key='get_gps_loc')

        if loc_data and 'coords' in loc_data:
            lat = loc_data['coords']['latitude']
            lon = loc_data['coords']['longitude']
            source = "GPS (Akurasi Tinggi)"
            return lat, lon, source

        # 2. IP ADDRESS (FALLBACK)
        try:
            response = requests.get('http://ip-api.com/json/', timeout=3)
            if response.status_code == 200:
                data = response.json()
                lat = data['lat']
                lon = data['lon']
                source = "IP Address (Estimasi)"
                return lat, lon, source
        except Exception as e:
            print(f"Gagal IP Location: {e}")

        return None, None, None

    def get_address(self, lat, lon):
        """
        Mengubah Lat/Lon menjadi Alamat Lengkap (Reverse Geocoding)
        """
        try:
            # language='id' agar outputnya Bahasa Indonesia
            location = self.geolocator.reverse((lat, lon), language='id', timeout=5)
            if location:
                return location.address
            else:
                return "Alamat tidak ditemukan"
        except Exception as e:
            return f"Gagal memuat alamat: {e}"