"""
Détecteur YOLO Ultra-Robuste - Version Drop-in
==============================================
Remplace le YOLO28HDetector avec une architecture ultra-robuste sans filtrage par classes.
"""

import os
import logging
import numpy as np
from typing import List
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage
from src.ancomicsviewer.detectors.ultra_panel_detector import UltraPanelDetector, ensure_rgb_uint8

log = logging.getLogger("Panels")

def qimage_to_rgb_array(qimage: QImage) -> np.ndarray:
    """
    Conversion robuste QImage -> RGB array.
    Version ultra-sécurisée sans setsize.
    """
    if qimage.isNull():
        raise ValueError("QImage is null")
    
    # Force conversion vers RGBA8888 pour standardiser
    if qimage.format() != QImage.Format.Format_RGBA8888:
        qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
    
    w, h = qimage.width(), qimage.height()
    bpl = qimage.bytesPerLine()
    ptr = qimage.constBits()
    
    # Conversion buffer sécurisée
    buffer_size = bpl * h
    buffer = bytes(ptr)[:buffer_size]
    
    log.debug(f"🔍 QImage conversion: {w}x{h}, bpl={bpl}, buffer_size={buffer_size}")
    
    # Reshape en array RGBA
    arr = np.frombuffer(buffer, dtype=np.uint8).reshape(h, bpl)[:, :w*4]
    rgba = arr.reshape(h, w, 4)
    
    # Extraire RGB (drop alpha)
    rgb = rgba[:, :, :3]
    
    # Assurer la contiguïté mémoire (CRITICAL pour YOLO)
    return ensure_rgb_uint8(rgb)

class UltraRobustDetector:
    """
    Détecteur YOLO ultra-robuste utilisant le modèle 28h avec architecture drop-in.
    AUCUN filtrage par classes en entrée, filtrage par noms normalisés en sortie.
    """
    
    def __init__(self):
        """Initialise le détecteur ultra-robuste."""
        self.model_path = "runs/multibd_enhanced_v2/yolov8s-mps-1280/weights/best.pt"
        self.detector = None
        self.conf_threshold = 0.25
        self.iou_threshold = 0.60
        
        log.info("🔥 UltraRobustDetector: Initialisation...")
        self._ensure_detector_loaded()
        log.info("✅ UltraRobustDetector: Modèle ultra-robuste chargé avec succès!")
        log.info("🔥 MÉNAGE FAIT: Utilisation EXCLUSIVE du modèle YOLO 28h ULTRA-ROBUSTE !")
        
    def _ensure_detector_loaded(self):
        """Charge le détecteur si nécessaire."""
        if self.detector is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
            
            log.info(f"🔥 Chargement modèle ultra-robuste: {self.model_path}")
            # Essaie MPS, fallback CPU
            try:
                self.detector = UltraPanelDetector(self.model_path, device="mps")
                log.info("✅ Modèle chargé sur MPS")
            except Exception:
                self.detector = UltraPanelDetector(self.model_path, device="cpu") 
                log.info("✅ Modèle chargé sur CPU (fallback)")
    
    def get_model_info(self):
        """Retourne les informations du modèle pour compatibilité."""
        return {
            "name": "YOLO 28h Ultra-Robuste Drop-in",
            "confidence": self.conf_threshold,
            "device": "mps/cpu",
            "architecture": "ultra-robust",
            "filtering": "by_names_normalized"
        }
    
    def detect_panels(self, qimage: QImage, page_size_or_dpi = None) -> List[QRectF]:
        """
        Détecte les panels dans une QImage avec l'architecture ultra-robuste.
        
        Args:
            qimage: Image Qt à analyser
            page_size_or_dpi: Taille de page (QSizeF) ou DPI (float) - ignoré dans cette version
            
        Returns:
            Liste des rectangles de panels détectés
        """
        log.info("🔥 UltraRobustDetector.detect_panels() - MODÈLE 28H ULTRA-ROBUSTE EN ACTION!")
        
        try:
            # 1) Conversion QImage -> RGB ultra-sécurisée
            img_rgb = qimage_to_rgb_array(qimage)
            h, w = img_rgb.shape[:2]
            
            log.info(f"🔥 Image: {w}x{h}, conf={self.conf_threshold}")
            
            # 2) Détection ultra-robuste (SANS filtrage classes en entrée)
            self._ensure_detector_loaded()
            detections = self.detector.detect(
                img_rgb, 
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=None  # Auto-calculé
            )
            
            log.info(f"🔥 YOLO trouvé {len(detections)} détections ultra-robustes!")
            
            # 3) Conversion vers QRectF
            panels = []
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                w_panel = x2 - x1
                h_panel = y2 - y1
                
                # QRectF(x, y, width, height)
                rect = QRectF(x1, y1, w_panel, h_panel)
                panels.append(rect)
                
                log.info(f"🔥 Panel {i+1}: ({x1:.0f},{y1:.0f}) {w_panel:.0f}x{h_panel:.0f} conf={det['conf']:.3f} name={det['name']}")
            
            log.info(f"🔥 Final: {len(panels)} panels détectés par YOLO 28h ultra-robuste")
            return panels
            
        except Exception as e:
            log.error(f"❌ Erreur détection ultra-robuste: {e}")
            import traceback
            traceback.print_exc()
            return []
