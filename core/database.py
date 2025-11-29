# core/database.py
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

class VectorDB:
    def __init__(self):
        # Ambil credentials dari st.secrets
        try:
            self.client = QdrantClient(
                url=st.secrets["QDRANT_URL"],
                api_key=st.secrets["QDRANT_API_KEY"]
            )
            self.collection_name = "absensi"
            self._init_collection()
        except Exception as e:
            st.error(f"Gagal konek ke Qdrant: {e}")

    def _init_collection(self):
        """Buat koleksi jika belum ada"""
        try:
            self.client.get_collection(self.collection_name)
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )

    def save_user(self, username, embedding):
        """Simpan user baru dengan vector rata-rata"""
        point_id = str(uuid.uuid4())
        
        operation_info = self.client.upsert(
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

    def search_user(self, embedding, threshold=0.5):
        """Cari user berdasarkan kemiripan"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=1,
            score_threshold=threshold
        )
        if results:
            return results[0].payload['username'], results[0].score
        return None, 0.0