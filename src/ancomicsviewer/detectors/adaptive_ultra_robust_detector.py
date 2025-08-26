#!/usr/bin/env python3
"""
Détecteur Adaptive Ultra Robuste - Architecture Requirements Compliant
=====================================================================

Implémente les AR-01 à AR-08 pour un système de détection parfaitement aligné :
- Même QImage pour affichage et détection
- Mapping correct des boîtes Ultralytics
- Filtrage sécurisé des classes
- Support du letterboxing avec coordonnées exactes
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PySide6 import QtGui

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ Ultralytics non disponible - mode fallback")

from .base import BasePanelDetector
from ..utils.qimage_utils import qimage_to_numpy
from ..utils.box_mapping import ultra_yolobox_to_display


class AdaptiveUltraRobustDetector(BasePanelDetector):
    """
    Détecteur AR-compliant avec mapping exact des coordonnées.
    """
    
    def __init__(self, dpi: float = 150):
        super().__init__()
        self.dpi = dpi
        self.model_path = Path(__file__).parent.parent.parent.parent / "data" / "models" / "multibd_enhanced_v2.pt"
        self.model = None
        self.model_size = 1280  # Taille Ultralytics standard
        
        # AR-06: Classes sécurisées
        self.KEEP_CLASSES = {"panel", "panel_inset", "balloon"}
        
        # Charger le modèle
        self._load_model()
    
    def _load_model(self):
        """Chargement sécurisé du modèle avec logs de diagnostic."""
        if not ULTRALYTICS_AVAILABLE:
            print("❌ Ultralytics non disponible")
            return
            
        if not self.model_path.exists():
            print(f"❌ Modèle non trouvé: {self.model_path}")
            return
            
        try:
            self.model = YOLO(str(self.model_path))
            
            # AR-06: Log des noms de classes pour diagnostic
            if hasattr(self.model, 'names') and self.model.names:
                names = list(self.model.names.values())[:5]  # Premier 5
                print(f"🔤 Model classes (first 5): {names}")
            
            print(f"✅ Modèle chargé: {self.model_path.name}")
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            self.model = None
    
    def _name_of_class(self, cls_id) -> str:
        """Conversion sécurisée ID classe -> nom."""
        if self.model and hasattr(self.model, 'names'):
            if isinstance(cls_id, (int, float)):
                return self.model.names.get(int(cls_id), f"class_{int(cls_id)}")
            else:
                return str(cls_id)
        return "unknown"
    
    def detect_on_qimage(self, qimg: QtGui.QImage) -> List[Dict[str, Any]]:
        """
        AR-02: Détection directe sur QImage (même que l'affichage).
        
        Args:
            qimg: QImage exactement tel qu'affiché dans PageView
            
        Returns:
            Liste de détections avec coordonnées dans l'espace QImage
        """
        if not self.model:
            print("⚠️ Pas de modèle chargé")
            return []
        
        # AR-02: Conversion QImage -> numpy (même image que l'affichage)
        img_array = qimage_to_numpy(qimg)
        W, H = qimg.width(), qimg.height()
        
        # AR-07: Logs de diagnostic
        print(f"[Debug] QImage {W}x{H} -> numpy {img_array.shape}")
        
        try:
            # Inférence Ultralytics
            results = self.model(img_array, imgsz=self.model_size, verbose=False)
            
            detections = []
            raw_count = 0
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    raw_count += len(boxes)
                    
                    for i in range(len(boxes)):
                        # Récupérer les coordonnées brutes (espace letterbox S×S)
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        
                        # AR-03: Mapping letterbox -> QImage
                        ox1, oy1, ox2, oy2 = ultra_yolobox_to_display(
                            x1, y1, x2, y2, W, H, self.model_size
                        )
                        
                        # AR-06: Filtrage sécurisé des classes
                        cls_name = self._name_of_class(cls_id)
                        if cls_name not in self.KEEP_CLASSES:
                            continue
                        
                        detection = {
                            'x1': ox1, 'y1': oy1, 'x2': ox2, 'y2': oy2,
                            'cls': cls_name,
                            'conf': conf
                        }
                        detections.append(detection)
            
            # AR-07: Logs de diagnostic
            print(f"[Debug] Raw dets: {raw_count}, Filtered: {len(detections)}")
            if detections:
                print(f"[Debug] First detection: {detections[0]}")
            
            return detections
            
        except Exception as e:
            print(f"❌ Erreur détection: {e}")
            return []
    
    def detect_panels_and_balloons(self, qimg: QtGui.QImage, page_size: Tuple[float, float], 
                                 img_size: Tuple[int, int]) -> Tuple[List, List]:
        """
        Méthode compatible avec l'interface existante.
        
        Args:
            qimg: QImage à analyser
            page_size: Taille de la page en points PDF (ignoré - on utilise QImage directement)  
            img_size: Taille de l'image (ignoré - on utilise QImage directement)
            
        Returns:
            Tuple (panels, balloons) avec coordonnées dans l'espace QImage
        """
        detections = self.detect_on_qimage(qimg)
        
        panels = []
        balloons = []
        
        for det in detections:
            cls_name = det['cls']
            # Convertir en QRectF pour compatibilité
            from PySide6.QtCore import QRectF
            rect = QRectF(det['x1'], det['y1'], det['x2'] - det['x1'], det['y2'] - det['y1'])
            
            if cls_name in ['panel', 'panel_inset']:
                panels.append(rect)
            elif cls_name == 'balloon':
                balloons.append(rect)
        
        return panels, balloons
    
    def detect_panels(self, qimg: QtGui.QImage, page_size: Tuple[float, float], 
                     img_size: Tuple[int, int]) -> List:
        """Méthode compatible pour détection panels uniquement."""
        panels, _ = self.detect_panels_and_balloons(qimg, page_size, img_size)
        return panels
