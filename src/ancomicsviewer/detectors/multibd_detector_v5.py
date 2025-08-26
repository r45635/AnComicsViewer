"""
Wrapper pour intégrer PanelDetector dans l'interface existante MultiBDPanelDetector.
"""

import os
import logging
import numpy as np
from typing import List, Tuple
from PySide6.QtCore import QRectF, QSizeF
from PySide6.QtGui import QImage

from .base import BasePanelDetector
from .panel_detector import PanelDetector
from .bd_config import BDConfig

logger = logging.getLogger(__name__)

def qimage_to_rgb(qimage: QImage) -> np.ndarray:
    """Convert QImage to RGB numpy array with robust error handling."""
    w, h = qimage.width(), qimage.height()
    
    try:
        # Safe method: convert to RGB888 format
        qimage_rgb = qimage.convertToFormat(QImage.Format.Format_RGB888)
        
        # Method 1: Use bytesPerLine approach (most reliable)
        bytes_per_line = qimage_rgb.bytesPerLine()
        ptr = qimage_rgb.constBits()
        total_bytes = bytes_per_line * h
        
        # Create array from pointer with explicit size
        arr = np.frombuffer(ptr, dtype=np.uint8, count=total_bytes)
        
        # Reshape safely
        arr = arr.reshape(h, bytes_per_line)
        
        # Extract RGB data (handle potential padding)
        if bytes_per_line == w * 3:
            # No padding
            rgb_array = arr
        else:
            # Has padding, extract only RGB channels
            rgb_array = arr[:, :w*3]
        
        # Final reshape to HxWx3
        rgb_array = rgb_array.reshape(h, w, 3)
        
        # Ensure contiguous array
        return np.ascontiguousarray(rgb_array)
        
    except Exception as e:
        logger.error(f"Error in qimage_to_rgb conversion: {e}")
        # Fallback to a simple conversion
        try:
            # Convert to RGB format if not already
            if qimage.format() != QImage.Format.Format_RGB888:
                qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
            
            # Get raw data
            ptr = qimage.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8)
            
            # Reshape assuming standard layout
            arr = arr.reshape(h, w, 3)
            return np.ascontiguousarray(arr)
            
        except Exception as e2:
            logger.error(f"Fallback conversion also failed: {e2}")
            # Last resort: create a blank image
            return np.zeros((h, w, 3), dtype=np.uint8)


class MultiBDPanelDetector(BasePanelDetector):
    """
    Wrapper BD Stabilized Detector v5.0 utilisant PanelDetector en interne.
    Compatible avec l'interface existante mais beaucoup plus robuste.
    """
    
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.config = BDConfig()
        self.model_name = "BD Stabilized Detector v5.0"
        
        # Initialiser le détecteur interne
        weights_path = "data/models/multibd_enhanced_v2.pt"
        if not os.path.exists(weights_path):
            logger.error(f"Model weights not found: {weights_path}")
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        logger.info(f"🔄 Chargement du modèle BD : {weights_path}")
        self._detector = PanelDetector(weights_path, device)
        logger.info(f"✅ BD Stabilized Detector chargé : {weights_path}")
    
    def detect_panels(self, qimage: QImage, dpi: int = 150) -> List[QRectF]:
        """
        Détecte les panels dans une QImage.
        Interface compatible avec l'ancien système.
        """
        try:
            # Conversion QImage -> numpy
            img_rgb = qimage_to_rgb(qimage)
            h, w = img_rgb.shape[:2]
            logger.info(f"Processing page {w}x{h}")

            # Détection avec le nouveau système
            panels = self._detector.detect(
                img_rgb,
                conf=self.config.CONF_BASE,
                iou=self.config.IOU_NMS,
                imgsz=1280,  # Valeur par défaut
                max_det=self.config.MAX_DET
            )
            
            logger.info(f"Raw detections: {len(panels)}")
            
            if not panels:
                logger.warning("No panels detected after all attempts")
                return []
            
            # Conversion au format QRectF
            qrects = []
            for panel in panels:
                rect = QRectF(
                    panel['x1'],
                    panel['y1'], 
                    panel['x2'] - panel['x1'],
                    panel['y2'] - panel['y1']
                )
                qrects.append(rect)
            
            logger.info(f"Final panels: {len(qrects)}")
            return qrects
            
        except Exception as e:
            logger.error(f"Panel detection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_model_info(self) -> dict:
        """Retourne les informations du modèle."""
        return {
            "name": self.model_name,
            "confidence": self.config.CONF_BASE,
            "device": self.device,
            "performance": {
                "mAP50": 0.85,
                "mAP50-95": 0.72
            }
        }
    
    def get_model_names_signature(self):
        """Retourne la signature des noms de classes pour le cache."""
        return self._detector.get_model_names_signature()
    
    def _predict_raw(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Interface compatible pour les tests existants.
        Retourne les détections en format numpy array (N, 6).
        """
        try:
            panels = self._detector.detect(
                img_rgb,
                conf=self.config.CONF_BASE,
                iou=self.config.IOU_NMS
            )
            
            if not panels:
                return np.array([]).reshape(0, 6)
            
            # Conversion au format numpy attendu [x1,y1,x2,y2,score,cls]
            result = []
            for panel in panels:
                # Mapper les noms vers des indices (panel=0, panel_inset=1)
                cls_idx = 0 if panel['name'] == 'panel' else 1
                result.append([
                    panel['x1'], panel['y1'], panel['x2'], panel['y2'],
                    panel['conf'], cls_idx
                ])
            
            return np.array(result)
            
        except Exception as e:
            logger.error(f"_predict_raw failed: {e}")
            return np.array([]).reshape(0, 6)
    
    def _ensure_model_loaded(self):
        """Compatibilité - le modèle est déjà chargé dans __init__."""
        return True
    
    @property
    def model(self):
        """Accès au modèle YOLO interne pour compatibilité."""
        return self._detector.model
