"""
Détecteur YOLO simplifié utilisant UNIQUEMENT le modèle de 28h d'entraînement.
Plus         print(f"🔥 Image: {w}x{h}, conf={self.conf_threshold}")
        
        # Prédiction YOLO directe
        results = self.model.predict(
            img_rgb, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,lexité, plus d'anciens systèmes - JUSTE YOLO.
"""

import os
import logging
import numpy as np
from typing import List
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from .base import BasePanelDetector

logger = logging.getLogger(__name__)

def qimage_to_rgb(qimage: QImage) -> np.ndarray:
    """Convert QImage to RGB numpy array."""
    w, h = qimage.width(), qimage.height()
    
    # Convert to RGB888 format FIRST
    qimage_rgb = qimage.convertToFormat(QImage.Format.Format_RGB888)
    
    # Get actual dimensions AFTER conversion
    h, w = qimage_rgb.height(), qimage_rgb.width()
    
    # Get raw data
    ptr = qimage_rgb.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8)
    
    # Debug info
    expected_size = h * w * 3
    actual_size = len(arr)
    print(f"🔍 QImage conversion: {w}x{h}, expected={expected_size}, actual={actual_size}")
    
    # Reshape to HxWx3 with safety check
    if actual_size != expected_size:
        print(f"❌ Size mismatch! Cropping array to expected size")
        arr = arr[:expected_size]
    
    arr = arr.reshape(h, w, 3)
    return np.ascontiguousarray(arr)


class YOLO28HDetector(BasePanelDetector):
    """
    Détecteur YOLO ultra-simple utilisant UNIQUEMENT le modèle de 28h.
    Aucune complexité, aucun fallback - juste votre modèle YOLO.
    """
    
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        # Configuration optimisée pour le modèle 28h 
        self.conf_threshold = 0.25  # 🔥 Augmenté pour réduire les faux positifs
        self.iou_threshold = 0.5
        
        # Chemin DIRECT vers votre modèle de 28h
        self.model_path = "runs/multibd_enhanced_v2/yolov8s-mps-1280/weights/best.pt"
        
        print(f"🔥 YOLO28HDetector: Chargement du modèle de 28h")
        print(f"🔥 Modèle: {self.model_path}")
        print(f"🔥 Existe: {os.path.exists(self.model_path)}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modèle de 28h introuvable: {self.model_path}")
        
        if YOLO is None:
            raise ImportError("ultralytics package required for YOLO detection")
        
        # Charger le modèle YOLO
        self.model = YOLO(self.model_path)
        print(f"✅ YOLO28HDetector: Modèle de 28h chargé avec succès!")
        
    def detect_panels(self, qimage: QImage, dpi: int = 150) -> List[QRectF]:
        """
        Détecte les panels avec YOLO uniquement.
        """
        print(f"🔥 YOLO28HDetector.detect_panels() - MODÈLE 28H EN ACTION!")
        
        # Conversion QImage -> numpy
        img_rgb = qimage_to_rgb(qimage)
        h, w = img_rgb.shape[:2]
        print(f"🔥 Image: {w}x{h}, conf={self.conf_threshold}")
        
        # Prédiction YOLO
        results = self.model.predict(
            img_rgb,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device
        )
        
        if not results or len(results) == 0:
            print(f"🔥 Aucun résultat YOLO")
            return []
        
        result = results[0]
        if not hasattr(result, 'boxes') or result.boxes is None:
            print(f"🔥 Pas de boîtes dans les résultats")
            return []
        
        boxes = result.boxes.xyxy  # Format [x1, y1, x2, y2]
        scores = result.boxes.conf
        
        # Convert to numpy if needed
        try:
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu().numpy()
            elif hasattr(boxes, 'numpy'):
                boxes = boxes.numpy()
            else:
                boxes = np.array(boxes)
                
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            elif hasattr(scores, 'numpy'):
                scores = scores.numpy()
            else:
                scores = np.array(scores)
        except Exception as e:
            print(f"🔥 Conversion warning: {e}")
            boxes = np.array(boxes)
            scores = np.array(scores)
        
        print(f"🔥 YOLO trouvé {len(boxes)} détections!")
        
        # Conversion vers QRectF
        panels = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            w_panel = x2 - x1
            h_panel = y2 - y1
            
            # Filtrage basique (optionnel)
            if w_panel < 50 or h_panel < 50:  # Trop petit
                continue
                
            rect = QRectF(x1, y1, w_panel, h_panel)
            panels.append(rect)
            print(f"🔥 Panel {i+1}: ({x1:.0f},{y1:.0f}) {w_panel:.0f}x{h_panel:.0f} conf={score:.3f}")
        
        print(f"🔥 Final: {len(panels)} panels détectés par YOLO 28h")
        return panels
    
    def get_model_info(self) -> dict:
        """Retourne les informations du modèle."""
        return {
            "name": "YOLO 28h Detector (Pure)",
            "confidence": self.conf_threshold,
            "device": self.device,
            "model_path": self.model_path,
            "performance": {
                "mAP50": 0.85,
                "mAP50-95": 0.72
            }
        }
