# panels_service.py 
import logging
from typing import List, Tuple, Optional
from PySide6.QtCore import QSizeF
from src.ancomicsviewer.detectors.ultra_panel_detector import UltraPanelDetector
from src.ancomicsviewer.ui.qimage_utils import qimage_to_rgb_np

log = logging.getLogger("Panels")

# Instance globale du détecteur (1 seule fois)
_DETECTOR: Optional[UltraPanelDetector] = None

def get_detector() -> UltraPanelDetector:
    """Obtient l'instance unique du détecteur ultra-robuste."""
    global _DETECTOR
    if _DETECTOR is None:
        # Utilise le modèle 28h existant
        model_path = "runs/multibd_enhanced_v2/yolov8s-mps-1280/weights/best.pt"
        try:
            _DETECTOR = UltraPanelDetector(model_path, device="mps")  # ou "cpu"
            log.info(f"🔥 Détecteur ultra-robuste initialisé avec {model_path}")
        except Exception as e:
            log.error(f"❌ Erreur initialisation détecteur: {e}")
            # Fallback CPU
            _DETECTOR = UltraPanelDetector(model_path, device="cpu")
    return _DETECTOR

def detect_panels_for_qimage(qimg, page_point_size: QSizeF) -> List[Tuple]:
    """
    Détection de panels ultra-robuste pour QImage.
    
    Returns:
        List[Tuple]: [(rect, name, conf), ...] où rect = (x, y, w, h) en coordonnées page
    """
    try:
        # 1) Conversion saine QImage -> RGB numpy
        rgb = qimage_to_rgb_np(qimg)  # RGB uint8 contigu
        H, W = rgb.shape[:2]
        
        # 2) Calcul facteur d'échelle
        scale = W / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0
        
        # 3) Détection sans filtrage par classes (ultra-robuste)
        detector = get_detector()
        dets = detector.detect(rgb, conf=0.25, iou=0.60, imgsz=None)
        
        log.info(f"[Panels] Détections trouvées: {len(dets)}")
        
        # 4) Projection vers coordonnées page
        rects = []
        for d in dets:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            # Conversion vers coordonnées page
            page_x = x1 / scale
            page_y = y1 / scale
            page_w = (x2 - x1) / scale
            page_h = (y2 - y1) / scale
            
            rect = (page_x, page_y, page_w, page_h)
            rects.append((rect, d["name"], d["conf"]))
            
            log.debug(f"Panel {d['name']}: conf={d['conf']:.3f} rect=({page_x:.1f},{page_y:.1f},{page_w:.1f},{page_h:.1f})")
        
        # 5) Tri par lecture (haut-bas, gauche-droite)
        rects.sort(key=lambda r: (r[0][1], r[0][0]))  # y puis x
        
        log.info(f"[Panels] Panels finaux après tri: {len(rects)}")
        return rects
        
    except Exception as e:
        log.error(f"❌ Erreur détection panels: {e}")
        import traceback
        traceback.print_exc()
        return []
