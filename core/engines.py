# core/engines.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import cv2
import streamlit as st
from facenet_pytorch import MTCNN

class _IndonesianFaceModel(nn.Module):
    def __init__(self, num_classes=68):
        super(_IndonesianFaceModel, self).__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        input_dim = 2048
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(0.4)
        self.fc_embedding = nn.Linear(input_dim, 512)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bn_input(x)
        x = self.dropout(x)
        return self.fc_embedding(x)

class FaceEngine:
    def __init__(self, model_path='models/model-absensi.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = _IndonesianFaceModel(num_classes=68)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Model Load Error: {e}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.detector = MTCNN(keep_all=True, device=self.device, min_face_size=40, thresholds=[0.6, 0.7, 0.7])

    def extract_face_coords(self, image_cv2):
        if image_cv2 is None: return None
        height, width, _ = image_cv2.shape
        img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        try:
            boxes, probs = self.detector.detect(img_pil)
            if boxes is None or len(boxes) == 0: return None
            
            best_idx = np.argmax(probs)
            box = boxes[best_idx]
            x1, y1, x2, y2 = [int(b) for b in box]
            
            x = max(0, x1)
            y = max(0, y1)
            w = min(width - x, x2 - x1)
            h = min(height - y, y2 - y1)
            
            if w < 20 or h < 20: return None
            return (x, y, w, h)
        except Exception:
            return None

    def get_embedding(self, face_crop):
        if face_crop is None or face_crop.size == 0: return np.zeros(512)
        img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            embedding = self.model(img_tensor)
        return embedding.cpu().numpy()[0]
    
    def calculate_average_embedding(self, embeddings_list):
        if not embeddings_list: return None
        stack = np.stack(embeddings_list)
        mean_emb = np.mean(stack, axis=0)
        return mean_emb / np.linalg.norm(mean_emb)

    # --- FITUR BARU: ANTI SPOOFING (TEXTURE ANALYSIS) ---
    def check_liveness(self, face_crop):
        """
        Menganalisis 'kekayaan tekstur' wajah menggunakan Laplacian Variance.
        Wajah asli punya tekstur (pori-pori/kulit) -> Variance Tinggi.
        Wajah di layar HP/Kertas cenderung blur/flat -> Variance Rendah.
        
        Returns: (Is_Real: bool, Score: float)
        """
        if face_crop is None or face_crop.size == 0:
            return False, 0.0
            
        # 1. Ubah ke Grayscale
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # 2. Hitung Laplacian Variance (Kekayaan Tepi/Tekstur)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 3. Tentukan Threshold Liveness
        # Score < 50 biasanya blur/layar HP
        # Score > 80 biasanya wajah asli tajam
        # Kita set threshold di angka 60 (bisa disesuaikan)
        is_real = score > 60.0
        
        return is_real, score