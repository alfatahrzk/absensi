import streamlit as st
from qdrant_client import QdrantClient
# Tambahkan PayloadSchemaType di import
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
import uuid
import requests 
import json
import numpy as np

class VectorDB:
    def __init__(self):
        self.client = None
        self.api_key = None
        self.url = None
        self.collection_name = "absensi" 
        
        try:
            if "QDRANT_URL" in st.secrets:
                self.url = st.secrets["QDRANT_URL"].strip().rstrip("/")
                self.api_key = st.secrets["QDRANT_API_KEY"]
            else:
                st.error("Secrets QDRANT belum disetting!")
                return

            self.client = QdrantClient(url=self.url, api_key=self.api_key)
            self._init_collection()
            
        except Exception as e:
            st.error(f"Gagal init Qdrant: {e}")

    def _init_collection(self):
        if self.client:
            # 1. Buat Collection (Jika belum ada)
            try:
                self.client.get_collection(self.collection_name)
            except:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                )
            
            # --- PERBAIKAN: BUAT INDEX UNTUK USERNAME ---
            # Ini solusi agar tidak error "Index required" saat delete
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="username",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                # print("Index username berhasil dibuat/sudah ada.")
            except Exception as e:
                # Biasanya error kalau index sudah ada, jadi kita pass saja
                pass
            # --------------------------------------------

    def save_user(self, username, embedding):
        if not self.client: return False
        point_id = str(uuid.uuid4())
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={"username": username, "role": "user"}
                    )
                ]
            )
            return True
        except Exception as e:
            st.error(f"Gagal simpan user: {e}")
            return False

    def search_user(self, embedding, threshold=0.5):
        if not self.url: return None, 0.0

        api_endpoint = f"{self.url}/collections/{self.collection_name}/points/search"
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        
        payload = {
            "vector": embedding.tolist(),
            "limit": 1,
            "score_threshold": threshold,
            "with_payload": True
        }

        try:
            response = requests.post(api_endpoint, headers=headers, data=json.dumps(payload), timeout=5)
            if response.status_code == 200:
                result_json = response.json()
                points = result_json.get('result', [])
                if points:
                    best_match = points[0]
                    return best_match['payload']['username'], best_match['score']
            return None, 0.0
        except Exception as e:
            st.error(f"Error Search API: {e}")
            return None, 0.0

    def get_all_users(self):
        if not self.url: return []
        
        api_endpoint = f"{self.url}/collections/{self.collection_name}/points/scroll"
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        
        payload = {
            "limit": 1000,
            "with_payload": True,
            "with_vector": False
        }
        
        try:
            response = requests.post(api_endpoint, headers=headers, data=json.dumps(payload), timeout=10)
            if response.status_code == 200:
                points = response.json().get('result', {}).get('points', [])
                users = [p['payload']['username'] for p in points if 'payload' in p and 'username' in p['payload']]
                return sorted(list(set(users)))
            else:
                return []
        except Exception as e:
            st.error(f"Error Get Users: {e}")
            return []

    def delete_user(self, username):
        if not self.url: return False
        
        api_endpoint = f"{self.url}/collections/{self.collection_name}/points/delete"
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        
        payload = {
            "filter": {
                "must": [
                    {
                        "key": "username",
                        "match": {"value": username}
                    }
                ]
            }
        }
        
        try:
            response = requests.post(api_endpoint, headers=headers, data=json.dumps(payload), timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                # Jika masih error, kita tampilkan detailnya
                st.error(f"🔥 Gagal Delete (Status {response.status_code}): {response.text}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error Koneksi Delete: {e}")
            return False

    def update_user_embedding(self, username, new_vector):
        """
        Mengupdate vektor wajah user dengan metode Moving Average.
        Rumus: (VektorLama * 0.9) + (VektorBaru * 0.1)
        """
        if not self.url: return False
        
        # 1. Cari Point ID dan Vektor Lama berdasarkan Username
        api_scroll = f"{self.url}/collections/{self.collection_name}/points/scroll"
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        
        payload_search = {
            "filter": {
                "must": [{"key": "username", "match": {"value": username}}]
            },
            "limit": 1,
            "with_vector": True # Kita butuh vektor lamanya
        }
        
        try:
            # Ambil data lama
            resp = requests.post(api_scroll, headers=headers, data=json.dumps(payload_search), timeout=5)
            if resp.status_code != 200: return False
            
            points = resp.json().get('result', {}).get('points', [])
            if not points: return False
            
            # Data lama ditemukan
            old_point = points[0]
            point_id = old_point['id']
            old_vector = np.array(old_point['vector'])
            new_input_vector = np.array(new_vector)
            
            # 2. Hitung Rata-rata Berbobot (Weighted Average)
            # Kita beri bobot 90% ke data lama (agar stabil) dan 10% ke data baru (agar adaptif)
            # Anda bisa ubah rasionya, misal 0.8 dan 0.2
            updated_vector = (old_vector * 0.90) + (new_input_vector * 0.10)
            
            # Normalisasi ulang (Wajib untuk Cosine Similarity)
            updated_vector = updated_vector / np.linalg.norm(updated_vector)
            
            # 3. Update ke Qdrant (Upsert dengan ID yang sama = Overwrite)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id, # Pakai ID lama biar menimpa
                        vector=updated_vector.tolist(),
                        payload=old_point['payload'] # Pakai payload lama
                    )
                ]
            )
            return True
            
        except Exception as e:
            # Silent error (jangan ganggu user kalau update gagal, karena ini cuma fitur tambahan)
            print(f"Gagal update adaptif: {e}")
            return False