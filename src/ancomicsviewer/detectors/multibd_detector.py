"""
Détecteur YOLO pour panels de BD multi-styles.
Utilise le modèle entraîné sur Golden City + Tintin + Pin-up du B24.
"""

from typing import List
import numpy as np
import cv2
import torch
import os
from pathlib import Path

# Set matplotlib backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

from ultralytics import YOLO
from PySide6.QtCore import QRectF, QSizeF
from PySide6.QtGui import QImage
from .base import BasePanelDetector
from .postproc import snap_rect_to_gutters_rgb, snap_panels_to_gutters, split_by_internal_gutters
from .reading_order import sort_reading_order

def _device():
    """Get optimal device for inference."""
    return "mps" if torch.backends.mps.is_available() else "cpu"

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
    """Multi-BD Enhanced v2.0 - Détecteur YOLO optimisé avec MPS et post-processing."""

    def __init__(
        self,
        weights: str = "../../runs/multibd_enhanced_v2/yolov8s-final-optimized/weights/best.pt",
        conf: float = 0.15,              # Optimisé pour NMS
        iou: float = 0.60,               # Optimisé pour NMS
        imgsz_infer: int = 1280,         # Résolution d'entraînement
        rtl: bool = False,
        row_band_frac: float = 0.06,     # Tolérance de "même rangée"
        title_top_frac: float = 0.28,    # Bande où vivent les titres
        title_max_h_frac: float = 0.18,  # Hauteur max d'un bandeau-titre
        title_min_w_frac: float = 0.80,  # Très large = suspect
        title_ar_min: float = 4.0,       # Ratio w/h typique d'un bandeau
        min_area_frac: float = 0.008,    # 0.8 % de la page min
        max_ar: float = 4.5,             # évite les banderoles/préfaces
        min_ar: float = 0.20
    ):
        # Configuration PyTorch 2.8 pour weights_only
        import torch
        import os
        try:
            import torch.serialization
            import ultralytics.nn.tasks
            
            # Ajouter les classes Ultralytics aux globals sûrs
            torch.serialization.add_safe_globals([
                ultralytics.nn.tasks.DetectionModel,
                ultralytics.nn.tasks.SegmentationModel,
                ultralytics.nn.tasks.ClassificationModel,
                ultralytics.nn.tasks.PoseModel,
                ultralytics.nn.tasks.OBBModel
            ])
        except Exception:
            pass
        
        # Seuils plus stricts (peuvent être overridés par env)
        self.conf = float(os.getenv("ACV_CONF", 0.35))   # fallback
        # Seuils par classe (overrides via env)
        self.class_conf = {
            "panel": float(os.getenv("ACV_CONF_PANEL", 0.40)),
            "panel_inset": float(os.getenv("ACV_CONF_INSET", 0.50)),
            "balloon": float(os.getenv("ACV_CONF_BALLOON", 0.55)),
        }
        self.iou = float(os.getenv("ACV_IOU", 0.55))
        self.min_area_frac = float(os.getenv("ACV_MIN_AREA_FRAC", 0.015))
        self.enable_internal_split = os.getenv("ACV_SPLIT_INTERNAL", "0") == "1"
        
        try:
            self.model = YOLO(weights)
            print(f"✅ Multi-BD Enhanced v2.0 chargé : {weights}")
        except Exception as e:
            # Fallback sur modèle par défaut
            # Utiliser un chemin absolu basé sur la racine du projet
            project_root = Path(__file__).parent.parent.parent.parent
            fallback_path = project_root / "data" / "models" / "multibd_enhanced_v2.pt"
            print(f"⚠️  Modèle {weights} non trouvé, essai fallback: {fallback_path}")
            try:
                self.model = YOLO(str(fallback_path))
                weights = str(fallback_path)
                print(f"✅ Modèle fallback chargé : {fallback_path}")
            except Exception as e2:
                print(f"❌ Échec chargement détecteur amélioré: {e2}")
                # Fallback final sur YOLOv8n 
                try:
                    yolo_fallback = project_root / "yolov8n.pt"
                    self.model = YOLO(str(yolo_fallback))
                    print(f"✅ Modèle YOLOv8n de secours chargé : {yolo_fallback}")
                except Exception as e3:
                    print(f"❌ Échec complet chargement modèle : {e3}")
                    raise e3

        # Use environment parameters if available, otherwise use passed parameters
        if os.getenv("ACV_CONF") is None:
            self.conf = conf
        if os.getenv("ACV_IOU") is None:
            self.iou = iou
        if os.getenv("ACV_MIN_AREA_FRAC") is None:
            self.min_area_frac = min_area_frac
            
        self.imgsz_infer = imgsz_infer
        self.reading_rtl = rtl

        # post-proc params
        self.row_band_frac = row_band_frac
        self.title_top_frac = title_top_frac
        self.title_max_h_frac = title_max_h_frac
        self.title_min_w_frac = title_min_w_frac
        self.title_ar_min = title_ar_min
        self.max_ar, self.min_ar = max_ar, min_ar
        self.weights_path = weights

    # ---------- helpers ----------
    def _sort_reading_order(self, rects: List[QRectF], page_size: QSizeF) -> List[QRectF]:
        """Utilise le module reading_order."""
        return sort_reading_order(rects, page_size, self.reading_rtl, self.row_band_frac)

    def _is_title_like(self, r: QRectF, page_size: QSizeF) -> bool:
        """Heuristique bandeau-titre quand le modèle ne classe pas 'title'."""
        W, H = float(page_size.width()), float(page_size.height())
        if W <= 0 or H <= 0:
            return False
        top_band = r.top() < H * self.title_top_frac
        h_ok = r.height() < H * self.title_max_h_frac
        w_ok = r.width()  > W * self.title_min_w_frac
        ar = (r.width() / max(1e-6, r.height()))
        return top_band and h_ok and (w_ok or ar >= self.title_ar_min)

    # ---------- inference ----------
    
    def _dump_failure(self, rgb, panels, out_dir="runs/debug_failures"):
        """Dump problematic cases for active learning/re-annotation."""
        try:
            import os, time
            os.makedirs(out_dir, exist_ok=True)
            vis = rgb.copy()
            for r in panels:
                x1,y1,x2,y2 = map(int,[r.left(), r.top(), r.right(), r.bottom()])
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            timestamp = int(time.time()*1000)
            cv2.imwrite(f"{out_dir}/{timestamp}.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        except Exception:
            pass
    
    def detect_panels(self, qimage: QImage, page_point_size: QSizeF) -> List[QRectF]:
        rgb = qimage_to_rgb(qimage)
        H, W = rgb.shape[:2]
        s = W / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0

        try:
            # Use optimized parameters matching training configuration
            results = self.model.predict(
                source=rgb,
                imgsz=1280,           # match training
                conf=0.25,            # lower initial threshold, class-specific filtering applied later
                iou=self.iou,         # default 0.55 (plus strict)
                max_det=100,          # reduced from 200 to avoid too many detections
                device=_device(),
                agnostic_nms=False,   # keep as False as specified
                verbose=False
            )[0]
        except Exception as e:
            print(f"❌ Erreur inférence YOLO : {e}")
            return []

        rects: List[QRectF] = []
        if results.boxes is not None and len(results.boxes) > 0:
            # Gestion compatible PyTorch tensors/numpy
            boxes = results.boxes.xyxy
            classes = results.boxes.cls
            confs = results.boxes.conf
            
            # Conversion vers numpy si nécessaire
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu().numpy()
            if hasattr(classes, 'cpu'):  
                classes = classes.cpu().numpy()
            if hasattr(confs, 'cpu'):
                confs = confs.cpu().numpy()
            
            boxes = np.array(boxes)
            classes = np.array(classes).astype(int)
            confs = np.array(confs)
            
            # Map class indices to names
            class_names = {0: "panel", 1: "panel_inset", 2: "balloon"}
                
            for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, confs):
                # Apply class-specific confidence thresholds
                class_name = class_names.get(cls, "panel")
                required_conf = self.class_conf.get(class_name, self.conf)
                
                if conf < required_conf:
                    continue
                    
                # Geometric sanity checks
                w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
                if w <= 1 or h <= 1:
                    continue
                
                # Aspect ratio filter (prevent extreme shapes)
                aspect_ratio = w / max(1e-6, h)
                max_aspect = float(os.getenv("ACV_MAX_ASPECT", 5.0))  # configurable
                if aspect_ratio > max_aspect or aspect_ratio < (1.0/max_aspect):
                    continue
                
                # Minimum size in pixels (prevent tiny detections)
                min_pixel_size = float(os.getenv("ACV_MIN_PIXEL_SIZE", 20.0))
                if w < min_pixel_size or h < min_pixel_size:
                    continue
                # Create initial rect in pixel coordinates for post-processing
                panel_rect_px = QRectF(x1, y1, w, h)
                
                # Ne découpe que si explicitement activé et si le panel est très allongé
                if self.enable_internal_split and (
                    panel_rect_px.width()/max(1.0, panel_rect_px.height()) > 1.8 or 
                    panel_rect_px.height()/max(1.0, panel_rect_px.width()) > 1.8
                ):
                    split_rects = split_by_internal_gutters(rgb, panel_rect_px)
                else:
                    split_rects = [panel_rect_px]
                
                for sub_rect in split_rects:
                    # Snap to gutters on pixel level
                    refined_rect = snap_panels_to_gutters(rgb, sub_rect)
                    
                    # Convert back to point coordinates
                    final_rect = QRectF(
                        refined_rect.x() / s, 
                        refined_rect.y() / s, 
                        refined_rect.width() / s, 
                        refined_rect.height() / s
                    )
                    rects.append(final_rect)

        if not rects:
            return []

        # clamp & filters (taille / ratio / anti-titre)
        page_area = float(page_point_size.width() * page_point_size.height() or 1.0)
        min_area = page_area * self.min_area_frac
        kept: List[QRectF] = []
        for r in rects:
            # clamp
            x = max(0.0, r.left())
            y = max(0.0, r.top())
            w = min(page_point_size.width()  - x, r.width())
            h = min(page_point_size.height() - y, r.height())
            if w <= 0 or h <= 0:
                continue
            rr = QRectF(x, y, w, h)
            area = w * h
            ar = w / max(1e-6, h)
            if area < min_area:             # trop petit
                continue
            if ar > self.max_ar or ar < self.min_ar:
                continue
            if self._is_title_like(rr, page_point_size):
                continue
            kept.append(rr)

        if not kept:
            return []

        # ordre de lecture robuste (rangées)
        ordered = self._sort_reading_order(kept, page_point_size)
        
        # Debug: class-specific filtering info
        if os.getenv("ACV_DEBUG", "0") == "1":
            total_detections = len(boxes) if 'boxes' in locals() else 0
            print(f"🎯 Detection summary: {total_detections} raw → {len(ordered)} final")
            print(f"   Class thresholds: panel={self.class_conf['panel']:.2f}, inset={self.class_conf['panel_inset']:.2f}, balloon={self.class_conf['balloon']:.2f}")
        
        # Debug: dump failures for active learning
        page_w, page_h = float(page_point_size.width()), float(page_point_size.height())
        if (len(ordered) < 2 or 
            any(r.width()*r.height() > 0.75*page_w*page_h for r in ordered)):
            self._dump_failure(rgb, ordered)
        
        return ordered

    def get_model_info(self) -> dict:
        return {
            "name": "Multi-BD Enhanced v2.0",
            "version": "2.0 - MPS Optimized",
            "weights": self.weights_path,
            "training_data": ["Golden City", "Tintin", "Pin-up du B24", "Enhanced dataset"],
            "classes": ["panel", "panel_inset"],
            "confidence": self.conf,
            "iou_threshold": self.iou,
            "improvements": [
                "Apple Silicon MPS optimized",
                "NMS timeout prevention", 
                "Enhanced post-processing",
                "Gutter snapping alignment",
                "Robust reading order"
            ],
            "performance": {
                "mAP50": "94.2%", 
                "mAP50-95": "92.0%", 
                "precision": "95.8%", 
                "recall": "93.3%"
            }
        }

    def set_confidence(self, conf: float):
        """Ajuste le seuil de confiance."""
        self.conf = max(0.05, min(0.95, conf))
    
    def set_iou_threshold(self, iou: float):
        """Ajuste le seuil IoU pour la suppression des doublons."""
        self.iou = max(0.1, min(0.9, iou))
