import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
import requests # <--- Kita butuh ini untuk menembak API langsung
import json

class VectorDB:
    def __init__(self):
        self.client = None
        self.api_key = None
        self.url = None
        
        try:
            # Ambil Credentials
            if "QDRANT_URL" in st.secrets:
                self.url = st.secrets["QDRANT_URL"]
                self.api_key = st.secrets["QDRANT_API_KEY"]
            else:
                st.error("Secrets QDRANT belum disetting!")
                return

            # Tetap init client untuk fungsi save/create collection (karena biasanya yang error cuma search)
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
            self.collection_name = "wajah_karyawan"
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
        """
        Mencari user menggunakan REST API Langsung (Bypass Library Error)
        """
        if not self.url: return None, 0.0

        # --- JURUS PAMUNGKAS: PAKE HTTP REQUEST ---
        # Kita tidak pakai self.client.search() karena error di cloud
        # Kita tembak langsung URL API Qdrant
        
        # Bersihkan URL (hapus port :6333 jika ada, karena requests biasanya auto handle atau butuh format bersih)
        # Tapi biasanya format cloud qdrant: https://xyz...:6333
        
        # Endpoint Search Qdrant
        api_endpoint = f"{self.url}/collections/{self.collection_name}/points/search"
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "vector": embedding.tolist(),
            "limit": 1,
            "score_threshold": threshold,
            "with_payload": True
        }

        try:
            # Kirim Request HTTP POST
            response = requests.post(api_endpoint, headers=headers, data=json.dumps(payload), timeout=5)
            
            if response.status_code == 200:
                result_json = response.json()
                # Parsing hasil JSON dari Qdrant
                # Struktur: {'result': [{'id': '...', 'score': 0.8, 'payload': {'username': '...'}}], ...}
                points = result_json.get('result', [])
                
                if points:
                    best_match = points[0]
                    username = best_match['payload']['username']
                    score = best_match['score']
                    return username, score
            else:
                # Jika gagal, coba print errornya di log (bukan di layar user biar ga panik)
                print(f"API Error: {response.text}")
                
            return None, 0.0

        except Exception as e:
            st.error(f"Error Search via API: {e}")
            return None, 0.0