import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType
import uuid
import requests 
import json

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
            try:
                self.client.get_collection(self.collection_name)
            except:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                )
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="username",
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except:
                pass

    def save_user(self, username, embedding):
        """Mendaftarkan user baru (Menambah titik baru)"""
        if not self.client: return False
        
        # Generate ID Unik setiap kali save
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

    # --- FITUR BARU: TAMBAH VARIASI WAJAH (BUKAN REPLACE) ---
    def add_variation(self, username, new_vector):
        """
        Menambahkan variasi wajah baru untuk user yang sama.
        Tidak menghapus data lama, tidak merata-rata.
        Hanya menambah 'perwakilan' baru di database.
        """
        return self.save_user(username, new_vector)
    # -------------------------------------------------------

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
        
        payload = {"limit": 1000, "with_payload": True, "with_vector": False}
        
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
                "must": [{"key": "username", "match": {"value": username}}]
            }
        }
        
        try:
            response = requests.post(api_endpoint, headers=headers, data=json.dumps(payload), timeout=10)
            return response.status_code == 200
        except Exception as e:
            st.error(f"❌ Error Delete: {e}")
            return False