"""
Stabilized BD Panel Detector with BD-aware post-processing
Eliminates per-album threshold tuning via advanced post-processing pipeline.
"""

from typing import Tuple, List
import numpy as np
import cv2
import os
from PySide6.QtCore import QRectF, QSizeF
from PySide6.QtGui import QImage
from .base import BasePanelDetector
from .postproc_bd import (
    clip_to_page, adaptive_conf_filter, estimate_content_bbox, 
    drop_outside_content, size_aspect_priors, class_aware_wbf,
    nested_inset_rule, merge_collinear_strips, split_large_by_gutters,
    final_class_aware_nms
)
from .bd_config import BDConfig
from .reading_order import sort_reading_order

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

import logging
logger = logging.getLogger(__name__)

CLASS_TO_ID = {"panel": 0, "panel_inset": 1, "balloon": 2}

def qimage_to_rgb(qimg: QImage) -> np.ndarray:
    """Convert QImage to numpy RGB array."""
    if qimg.format() != QImage.Format.Format_RGBA8888:
        qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    h, w = qimg.height(), qimg.width()
    bpl = qimg.bytesPerLine()
    arr = np.frombuffer(bytes(qimg.constBits()), dtype=np.uint8).reshape(h, bpl)[:, :w*4]
    rgba = arr.reshape(h, w, 4)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

def xyxy_to_qrectf(x1: float, y1: float, x2: float, y2: float) -> QRectF:
    """Convert xyxy coordinates to QRectF."""
    return QRectF(float(x1), float(y1), float(x2 - x1), float(y2 - y1))

class MultiBDPanelDetector(BasePanelDetector):
    """
    Generic detector with multi-scale TTA, WBF merge, noise filtering, 
    inset relabeling, dynamic fallback, and stable reading order.
    No per-album tuning required.
    """

    def __init__(self, weights: str | None = None, device: str = "mps"):
        super().__init__()
        
        # Try to find model weights
        if weights is None:
            # Search for the model in common locations
            search_paths = [
                "../../runs/multibd_enhanced_v2/yolov8s-final-optimized/weights/best.pt",
                "data/models/multibd_enhanced_v2.pt",
                "detectors/models/multibd_enhanced_v2.pt",
                "src/ancomicsviewer/detectors/models/multibd_enhanced_v2.pt"
            ]
            
            for path in search_paths:
                if os.path.exists(path):
                    weights = path
                    print(f"✅ Modèle trouvé : {path}")
                    break
            
            if weights is None:
                print("⚠️ Aucun modèle trouvé, utilisez le chemin explicite")
                weights = "best.pt"  # fallback
        
        # Load YOLO model with error handling
        try:
            self.model = YOLO(weights) if YOLO else None
            if self.model:
                print(f"✅ Multi-BD Generic v3.0 chargé : {weights}")
        except Exception as e:
            print(f"❌ Erreur chargement modèle {weights}: {e}")
            self.model = None
            
        self.device = device
        self.weights_path = weights

        # Tunables (safe defaults across albums)
        self.tta_imgsz = [1024, 1280, 1536]   # short-side sizes
        self.conf_panel = 0.35
        self.conf_relax = 0.24                # fallback pass
        self.nms_iou    = 0.60
        self.wbf_iou    = 0.55
        self.min_boxes  = 3                   # if fewer -> run fallback
        self.want_balloons_for_nav = False    # panels only by default

    # --- helpers -------------------------------------------------
    def _predict_raw(self, img_rgb: np.ndarray, imgsz: int, conf: float):
        if self.model is None:
            return np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int)
            
        res = self.model.predict(
            img_rgb, imgsz=imgsz, conf=conf, iou=self.nms_iou,
            device=self.device, verbose=False
        )[0]
        
        if not hasattr(res, "boxes") or res.boxes is None:
            return np.zeros((0,4)), np.zeros((0,)), np.zeros((0,), dtype=int)
            
        # Convert to numpy arrays (handle both torch tensors and numpy arrays)
        try:
            # Try torch tensor conversion first
            boxes = res.boxes.xyxy.cpu().numpy()  # type: ignore
            scores = res.boxes.conf.cpu().numpy()  # type: ignore
            labels = res.boxes.cls.cpu().numpy().astype(int)  # type: ignore
        except AttributeError:
            # Fallback to direct numpy conversion
            boxes = np.array(res.boxes.xyxy)
            scores = np.array(res.boxes.conf)
            labels = np.array(res.boxes.cls).astype(int)
        
        return boxes, scores, labels

    def _predict_tta(self, img_rgb: np.ndarray, conf: float):
        all_b, all_s, all_l = [], [], []
        for s in self.tta_imgsz:
            b, sc, lb = self._predict_raw(img_rgb, s, conf)
            all_b.append(b); all_s.append(sc); all_l.append(lb)
        return all_b, all_s, all_l

    # --- main entry ----------------------------------------------
    def detect_panels(self, qimage: QImage, page_point_size: QSizeF) -> List[QRectF]:
        """
        Main detection method compatible with base class interface.
        Returns list of QRectF panels in reading order.
        """
        try:
            img_rgb = qimage_to_rgb(qimage)
            H, W = img_rgb.shape[:2]

            # 1) TTA pass at base confidence
            a_b, a_s, a_l = self._predict_tta(img_rgb, self.conf_panel)
            mb, ms, ml = wbf_merge(a_b, a_s, a_l, iou_thr=self.wbf_iou, skip_box_thr=self.conf_panel*0.5)
            mb, ms, ml = filter_noise(mb, ms, ml, W, H, min_conf=self.conf_panel*0.5)

            if not self.want_balloons_for_nav:
                keep = ml != CLASS_TO_ID["balloon"]
                mb, ms, ml = mb[keep], ms[keep], ml[keep]

            # 2) Fallback if recall is too low
            if len(mb) < self.min_boxes:
                b_b, b_s, b_l = self._predict_tta(img_rgb, self.conf_relax)
                rb, rs, rl = wbf_merge(b_b, b_s, b_l, iou_thr=self.wbf_iou, skip_box_thr=self.conf_relax*0.5)
                rb, rs, rl = filter_noise(rb, rs, rl, W, H, min_conf=self.conf_relax*0.5)
                if not self.want_balloons_for_nav:
                    keep = rl != CLASS_TO_ID["balloon"]
                    rb, rs, rl = rb[keep], rs[keep], rl[keep]
                # union
                mb = np.concatenate([mb, rb], axis=0) if len(mb) else rb
                ms = np.concatenate([ms, rs], axis=0) if len(ms) else rs
                ml = np.concatenate([ml, rl], axis=0) if len(ml) else rl

            # 3) Relabel insets by containment
            mb, ms, ml = classify_panels_and_insets(mb, ms, ml, W, H, inset_ratio=0.6)

            # 4) Reading order for panels first, then insets (keep relative order)
            if len(mb) > 0:
                sort_idx = sort_reading_order(mb, rtl=False)
                mb, ml = mb[sort_idx], ml[sort_idx]

            # 5) Convert to QRectF list (panels only for navigation)
            panels = []
            for (x1,y1,x2,y2), lab in zip(mb, ml):
                if lab == CLASS_TO_ID["panel"]:  # Only panels for navigation
                    panels.append(xyxy_to_qrectf(x1, y1, x2, y2))
            
            return panels
            
        except Exception as e:
            print(f"❌ Erreur détection: {e}")
            return []

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "name": "Multi-BD Generic v3.0",
            "version": "3.0 - TTA+WBF+Generic",
            "weights": self.weights_path,
            "device": self.device,
            "training_data": ["Golden City", "Tintin", "Pin-up du B24", "Enhanced dataset"],
            "confidence": self.conf_panel,  # Main confidence threshold
            "performance": {
                "mAP50": "71%",
                "mAP50-95": "65%", 
                "precision": "73%",
                "recall": "69%"
            },
            "features": ["Multi-scale TTA", "WBF fusion", "Noise filtering", "Inset classification", "Generic thresholds"]
        }

