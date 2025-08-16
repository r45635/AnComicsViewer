"""
Détecteur YOLO pour panels de BD multi-styles.
Utilise le modèle entraîné sur Golden City + Tintin + Pin-up du B24.
"""

from typing import List
import numpy as np
import cv2

# Set matplotlib backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

from ultralytics import YOLO
from PySide6.QtCore import QRectF, QSizeF
from PySide6.QtGui import QImage
from .base import BasePanelDetector

# Classes de notre modèle multi-BD
CLASS_PANEL = 0
CLASS_PANEL_INSET = 1

def qimage_to_rgb(qimg: QImage) -> np.ndarray:
    """Convertit QImage en tableau numpy RGB."""
    if qimg.format() != QImage.Format.Format_RGBA8888:
        qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    h, w = qimg.height(), qimg.width()
    bpl = qimg.bytesPerLine()
    arr = np.frombuffer(bytes(qimg.constBits()), dtype=np.uint8).reshape(h, bpl)[:, :w*4]
    rgba = arr.reshape(h, w, 4)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

class MultiBDPanelDetector(BasePanelDetector):
    """
    Détecteur YOLO optimisé pour les BD multi-styles.
    
    Entraîné sur :
    - Golden City (style moderne complexe)
    - Tintin (style classique simple)
    - Pin-up du B24 (style aviation/guerre)
    
    Classes détectées :
    - panel : Cases normales
    - panel_inset : Cases incrustées/spéciales
    """
    
    def __init__(self, weights: str = "runs/detect/multibd_mixed_model/weights/best.pt", conf: float = 0.2, iou: float = 0.5, rtl: bool = False):
        # Utiliser notre modèle par défaut
        self.weights_path = weights
            
        # Appliquer le patch PyTorch pour la compatibilité
        import torch
        original_load = torch.load
        
        def patched_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
            if weights_only is None:
                weights_only = False  # Nécessaire pour les modèles YOLO
            return original_load(f, map_location=map_location, pickle_module=pickle_module, 
                               weights_only=weights_only, **kwargs)
        
        torch.load = patched_load
        try:
            self.model = YOLO(weights)
            print(f"✅ Modèle Multi-BD chargé : {weights}")
        except Exception as e:
            print(f"❌ Erreur chargement modèle : {e}")
            raise
        finally:
            torch.load = original_load
            
        self.conf = conf
        self.iou = iou
        self.reading_rtl = rtl
        self.weights_path = weights

    def detect_panels(self, qimage: QImage, page_point_size: QSizeF) -> List[QRectF]:
        """
        Détecte les panels dans une image de page BD.
        
        Args:
            qimage: Image de la page
            page_point_size: Taille de la page en points
            
        Returns:
            Liste des rectangles de panels en coordonnées page
        """
        # Conversion QImage -> numpy RGB
        rgb = qimage_to_rgb(qimage)
        H, W = rgb.shape[:2]
        
        # Facteur d'échelle pour conversion coordonnées image -> page
        s = W / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0
        
        # Inférence YOLO
        try:
            results = self.model.predict(
                source=rgb, 
                imgsz=640,  # Taille d'entraînement
                conf=self.conf, 
                iou=self.iou, 
                verbose=False
            )[0]
        except Exception as e:
            print(f"❌ Erreur inférence YOLO : {e}")
            return []
        
        panels = []
        
        # Traitement des détections
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box.tolist()
                
                # Créer le rectangle en coordonnées page
                panel_rect = QRectF(
                    x1 / s,  # x
                    y1 / s,  # y
                    (x2 - x1) / s,  # width
                    (y2 - y1) / s   # height
                )
                
                # Filtrer les détections trop petites
                min_area = 100  # pixels carrés minimum en coordonnées page
                if panel_rect.width() * panel_rect.height() > min_area:
                    panels.append(panel_rect)
        
        # Tri des panels pour lecture (gauche->droite ou droite->gauche)
        if panels:
            if self.reading_rtl:
                # Lecture RTL : trier par x décroissant, puis y croissant
                panels.sort(key=lambda r: (-r.x(), r.y()))
            else:
                # Lecture LTR : trier par y croissant, puis x croissant  
                panels.sort(key=lambda r: (r.y(), r.x()))
        
        return panels
    
    def get_model_info(self) -> dict:
        """Retourne les informations du modèle."""
        return {
            "name": "Multi-BD Panel Detector",
            "version": "1.0",
            "weights": self.weights_path,
            "training_data": ["Golden City", "Tintin", "Pin-up du B24"],
            "classes": ["panel", "panel_inset"],
            "confidence": self.conf,
            "iou_threshold": self.iou,
            "performance": {
                "mAP50": "91.1%",
                "mAP50-95": "88.3%",
                "precision": "84.0%",
                "recall": "88.7%"
            }
        }
    
    def set_confidence(self, conf: float):
        """Ajuste le seuil de confiance."""
        self.conf = max(0.05, min(0.95, conf))
    
    def set_iou_threshold(self, iou: float):
        """Ajuste le seuil IoU pour la suppression des doublons."""
        self.iou = max(0.1, min(0.9, iou))
