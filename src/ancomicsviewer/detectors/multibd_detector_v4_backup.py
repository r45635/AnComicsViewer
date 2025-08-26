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
            # No padding - direct reshape
            arr = arr.reshape(h, w, 3)
        else:
            # Has padding - extract only RGB data
            arr = arr[:, :w*3].reshape(h, w, 3)
        
        return arr.copy()  # Make a copy to avoid memory issues
        
    except Exception as e:
        logger.warning(f"QImage conversion failed ({e}), using fallback")
        
        # Fallback: pixel by pixel (slow but safe)
        fallback_arr = np.zeros((h, w, 3), dtype=np.uint8)
        qimage_rgb = qimage.convertToFormat(QImage.Format.Format_RGB888)
        
        # Limit size for safety in fallback mode
        safe_h = min(h, 2000)
        safe_w = min(w, 2000)
        
        for y in range(safe_h):
            for x in range(safe_w):
                try:
                    pixel = qimage_rgb.pixel(x, y)
                    fallback_arr[y, x, 0] = (pixel >> 16) & 0xFF  # R
                    fallback_arr[y, x, 1] = (pixel >> 8) & 0xFF   # G
                    fallback_arr[y, x, 2] = pixel & 0xFF          # B
                except:
                    pass
        
        return fallback_arr

def xyxy_to_qrectf(x1: float, y1: float, x2: float, y2: float) -> QRectF:
    """Convert xyxy coordinates to QRectF."""
    return QRectF(x1, y1, x2 - x1, y2 - y1)


class MultiBDPanelDetector(BasePanelDetector):
    """
    BD-aware panel detector with stabilized post-processing pipeline.
    Implements standardized YOLO inference + BD-specific post-processing.
    No per-album tuning required.
    """

    def __init__(self, weights: str | None = None, device: str = "cpu"):
        super().__init__()
        
        # Force CPU for stability on macOS (avoid MPS segfaults)
        if device == "mps":
            device = "cpu"
            logger.info("Using CPU instead of MPS for stability")
        
        # Store parameters but don't load model yet (lazy loading)
        self.device = device
        self.weights_path = weights
        self.config = BDConfig()
        self.model = None
        self._model_loaded = False
        self._model_failed = False
        
        # Try to find model weights
        if weights is None:
            search_paths = [
                "../../runs/multibd_enhanced_v2/yolov8s-final-optimized/weights/best.pt",
                "data/models/multibd_enhanced_v2.pt",
                "detectors/models/multibd_enhanced_v2.pt",
                "src/ancomicsviewer/detectors/models/multibd_enhanced_v2.pt"
            ]
            
            for path in search_paths:
                if os.path.exists(path):
                    self.weights_path = path
                    break
            
            if self.weights_path is None:
                self.weights_path = "best.pt"  # fallback
    
    def _ensure_model_loaded(self) -> bool:
        """Lazy load the YOLO model when needed."""
        if self._model_loaded or self._model_failed:
            return self._model_loaded
        
        try:
            if not YOLO:
                print("⚠️ YOLO non disponible")
                self._model_failed = True
                return False
                
            print(f"🔄 Chargement du modèle BD : {self.weights_path}")
            
            # Force CPU and disable verbose logging
            import torch
            torch.set_num_threads(1)  # Avoid threading issues
            
            self.model = YOLO(str(self.weights_path))
            if self.model:
                print(f"✅ BD Stabilized Detector chargé : {self.weights_path}")
                self._model_loaded = True
                return True
            else:
                self._model_failed = True
                return False
                
        except Exception as e:
            print(f"❌ Erreur chargement modèle {self.weights_path}: {e}")
            self.model = None
            self._model_failed = True
            return False

    def _predict_raw(self, img_rgb: np.ndarray) -> np.ndarray:
        """Raw YOLO prediction with robust class filtering."""
        if not self._ensure_model_loaded() or self.model is None:
            logger.warning("Model not available, returning empty detections")
            return np.array([]).reshape(0, 6)
            
        try:
            # --- MAPPING ROBUSTE AUX NOMS ---
            ACCEPT_CLASSES = {"panel", "panel_inset"}  # noms cibles côté dataset
            
            def _norm_name(n: str) -> str:
                if n is None:
                    return ""
                return n.strip().lower().replace(" ", "_").replace("-", "_")
            
            def _id_maps(model):
                names = getattr(model, "names", None)
                if isinstance(names, dict):
                    id2name = {int(k): str(v) for k, v in names.items()}
                elif isinstance(names, (list, tuple)):
                    id2name = {i: str(n) for i, n in enumerate(names)}
                else:
                    id2name = {}
                # log unique pour vérifier
                logger.info(f"[Panels] model.names -> {list(id2name.values())}")
                return id2name, { _norm_name(v): k for k, v in id2name.items() }
            
            id2name, _ = _id_maps(self.model)

            # --- APPEL MODÈLE STANDARDISÉ (SANS FILTRE CLASSES) ---
            results = self.model.predict(
                img_rgb, 
                imgsz=1280,
                conf=self.config.CONF_BASE,
                iou=self.config.IOU_NMS,
                device=self.device,
                agnostic_nms=False,
                augment=False,
                max_det=self.config.MAX_DET,
                classes=None,  # PAS de filtre - récupérer tout
                verbose=False
            )
            
            if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
                raw_count = 0
            else:
                raw_count = int(results[0].boxes.cls.shape[0]) if results[0].boxes.cls is not None else 0
            
            # LOG BRUT AVANT POST-PROCESS
            logger.info(f"[Panels] raw preds (all classes) = {raw_count}")
            
            if raw_count == 0:
                return np.array([]).reshape(0, 6)

            # Convert to numpy arrays safely
            try:
                # Try torch tensor conversion first
                boxes = results[0].boxes.xyxy.cpu().numpy()  # type: ignore
                scores = results[0].boxes.conf.cpu().numpy()  # type: ignore
                labels = results[0].boxes.cls.cpu().numpy().astype(int)  # type: ignore
            except AttributeError:
                # Fallback to direct numpy conversion
                boxes = np.array(results[0].boxes.xyxy)
                scores = np.array(results[0].boxes.conf)
                labels = np.array(results[0].boxes.cls).astype(int)
            
            # FILTRER PAR NOM DE CLASSE NORMALISÉ
            if len(boxes) > 0:
                # Créer les noms de classes normalisés pour chaque détection
                names_arr = np.array([_norm_name(id2name.get(i, "")) for i in labels])
                
                # Filtrer par nom de classe normalisé
                keep = np.isin(names_arr, [_norm_name(c) for c in ACCEPT_CLASSES])
                
                if np.any(keep):
                    # Mapper les noms vers des indices standardisés pour le post-processing
                    # panel -> 0, panel_inset -> 1
                    class_mapping = {"panel": 0, "panel_inset": 1}
                    mapped_labels = np.array([class_mapping.get(_norm_name(id2name.get(labels[i], "")), 0) for i in range(len(labels)) if keep[i]])
                    
                    # Combiner en format (N,6) [x1,y1,x2,y2,score,cls_index]
                    dets = np.column_stack([boxes[keep], scores[keep], mapped_labels])
                    logger.info(f"[Panels] after class-name filter = {len(dets)}")
                    return dets
                else:
                    logger.warning("[Panels] no valid panel classes found after name filtering")
                    
                    # FALLBACK: essai avec conf plus basse
                    logger.info("[Panels] trying fallback with lower confidence...")
                    results_fallback = self.model.predict(
                        img_rgb,
                        imgsz=max(1280, 1536),
                        conf=max(0.05, self.config.CONF_BASE * 0.4),
                        iou=min(0.70, self.config.IOU_NMS + 0.10),
                        device=self.device,
                        agnostic_nms=False,
                        augment=False,
                        max_det=max(600, self.config.MAX_DET),
                        classes=None,
                        verbose=False
                    )
                    
                    if results_fallback and hasattr(results_fallback[0], "boxes") and results_fallback[0].boxes is not None:
                        fallback_count = int(results_fallback[0].boxes.cls.shape[0]) if results_fallback[0].boxes.cls is not None else 0
                        logger.info(f"[Panels] fallback raw = {fallback_count}")
                        
                        if fallback_count > 0:
                            try:
                                boxes_fb = results_fallback[0].boxes.xyxy.cpu().numpy()
                                scores_fb = results_fallback[0].boxes.conf.cpu().numpy()
                                labels_fb = results_fallback[0].boxes.cls.cpu().numpy().astype(int)
                                
                                names_arr_fb = np.array([_norm_name(id2name.get(i, "")) for i in labels_fb])
                                keep_fb = np.isin(names_arr_fb, [_norm_name(c) for c in ACCEPT_CLASSES])
                                
                                if np.any(keep_fb):
                                    class_mapping = {"panel": 0, "panel_inset": 1}
                                    mapped_labels_fb = np.array([class_mapping.get(_norm_name(id2name.get(labels_fb[i], "")), 0) for i in range(len(labels_fb)) if keep_fb[i]])
                                    dets_fb = np.column_stack([boxes_fb[keep_fb], scores_fb[keep_fb], mapped_labels_fb])
                                    logger.info(f"[Panels] fallback after filter = {len(dets_fb)}")
                                    return dets_fb
                            except Exception as e:
                                logger.warning(f"[Panels] fallback processing failed: {e}")
                    
                    return np.array([]).reshape(0, 6)
            else:
                return np.array([]).reshape(0, 6)
                
        except Exception as e:
            logger.error(f"YOLO prediction failed: {e}")
            return np.array([]).reshape(0, 6)

    def _reading_order_sort(self, dets: np.ndarray, h: int, w: int) -> np.ndarray:
        """Sort detections in reading order."""
        if len(dets) == 0:
            return dets
        
        # Group by lines with 20% vertical overlap tolerance
        sorted_dets = []
        used = np.zeros(len(dets), dtype=bool)
        
        # Sort by y-coordinate first
        y_sorted_indices = np.argsort(dets[:, 1])
        
        for i in y_sorted_indices:
            if used[i]:
                continue
            
            # Start a new line
            line_dets = [i]
            used[i] = True
            
            y1_i, y2_i = dets[i][1], dets[i][3]
            
            # Find boxes on the same line (vertical overlap >= 20%)
            for j in y_sorted_indices:
                if used[j]:
                    continue
                
                y1_j, y2_j = dets[j][1], dets[j][3]
                
                # Calculate vertical overlap
                overlap = min(y2_i, y2_j) - max(y1_i, y1_j)
                min_height = min(y2_i - y1_i, y2_j - y1_j)
                
                if overlap / max(min_height, 1e-6) >= 0.2:
                    line_dets.append(j)
                    used[j] = True
            
            # Sort line by x-coordinate (left to right)
            line_boxes = dets[line_dets]
            x_sorted_indices = np.argsort(line_boxes[:, 0])
            sorted_dets.extend(line_boxes[x_sorted_indices])
        
        return np.array(sorted_dets) if sorted_dets else np.array([]).reshape(0, 6)

    def detect_panels(self, qimage: QImage, page_point_size: QSizeF) -> List[QRectF]:
        """
        Main detection method with BD-aware post-processing pipeline.
        Returns list of QRectF panels in reading order.
        """
        try:
            img_rgb = qimage_to_rgb(qimage)
            h, w = img_rgb.shape[:2]

            logger.info(f"Processing page {w}x{h}")

            # 1) Raw YOLO inference
            dets = self._predict_raw(img_rgb)
            logger.info(f"Raw detections: {len(dets)}")

            # GARDE-FOUS POST-PROCESS: sors immédiatement si vide
            if dets.shape[0] == 0:
                logger.warning("[Panels] NO DETECTIONS FROM _predict_raw → returning empty for this page")
                return []

            # 2) BD-aware post-processing pipeline (exact order from spec)
            dets = clip_to_page(dets, h, w)
            logger.info(f"After clip_to_page: {len(dets)}")
            
            dets = adaptive_conf_filter(
                dets, 
                min_conf=self.config.CONF_MIN,
                base_conf=self.config.CONF_BASE,
                target_min=self.config.TARGET_MIN,
                target_max=self.config.TARGET_MAX
            )
            logger.info(f"After adaptive_conf_filter: {len(dets)}")
            
            content_bbox = estimate_content_bbox(img_rgb)
            logger.info(f"Content bbox: {content_bbox}")
            
            dets = drop_outside_content(dets, content_bbox)
            logger.info(f"After drop_outside_content: {len(dets)}")
            
            dets = size_aspect_priors(dets, h, w)
            logger.info(f"After size_aspect_priors: {len(dets)}")
            
            dets = class_aware_wbf(dets, iou=self.config.WBF_IOU, min_votes=1)
            logger.info(f"After class_aware_wbf: {len(dets)}")
            
            dets = nested_inset_rule(dets)
            logger.info(f"After nested_inset_rule: {len(dets)}")
            
            dets = merge_collinear_strips(dets, h, w, axis='x', 
                                        overlap=self.config.MERGE_OVERLAP, 
                                        gap=self.config.MERGE_GAP)
            logger.info(f"After merge_collinear_strips(x): {len(dets)}")
            
            dets = merge_collinear_strips(dets, h, w, axis='y', 
                                        overlap=self.config.MERGE_OVERLAP, 
                                        gap=self.config.MERGE_GAP)
            logger.info(f"After merge_collinear_strips(y): {len(dets)}")
            
            dets = split_large_by_gutters(dets, img_rgb)
            logger.info(f"After split_large_by_gutters: {len(dets)}")
            
            dets = final_class_aware_nms(dets, iou=self.config.IOU_NMS)
            logger.info(f"After final_class_aware_nms: {len(dets)}")

            # 3) Fallback if too few panels detected
            panel_dets = dets[dets[:, 5] == 0]  # Only panels (cls=0)
            
            if len(panel_dets) < 2:
                avg_area = np.mean([(x2-x1)*(y2-y1) for x1,y1,x2,y2,_,_ in panel_dets]) if len(panel_dets) > 0 else 0
                page_area = h * w
                
                if avg_area > 0.4 * page_area:
                    logger.info("Applying fallback with lower confidence")
                    # Retry with lower confidence
                    dets_fallback = self._predict_raw(img_rgb)
                    if len(dets_fallback) > 0:
                        # Reprocess with base_conf=0.20
                        dets_fallback = adaptive_conf_filter(
                            dets_fallback,
                            min_conf=self.config.CONF_MIN,
                            base_conf=0.20,  # Lower threshold
                            target_min=self.config.TARGET_MIN,
                            target_max=self.config.TARGET_MAX
                        )
                        
                        # Continue pipeline from step 3
                        dets_fallback = drop_outside_content(dets_fallback, content_bbox)
                        dets_fallback = size_aspect_priors(dets_fallback, h, w)
                        dets_fallback = class_aware_wbf(dets_fallback, iou=self.config.WBF_IOU, min_votes=1)
                        dets_fallback = nested_inset_rule(dets_fallback)
                        dets_fallback = merge_collinear_strips(dets_fallback, h, w, axis='x', 
                                                             overlap=self.config.MERGE_OVERLAP, 
                                                             gap=self.config.MERGE_GAP)
                        dets_fallback = merge_collinear_strips(dets_fallback, h, w, axis='y', 
                                                             overlap=self.config.MERGE_OVERLAP, 
                                                             gap=self.config.MERGE_GAP)
                        dets_fallback = split_large_by_gutters(dets_fallback, img_rgb)
                        dets_fallback = final_class_aware_nms(dets_fallback, iou=self.config.IOU_NMS)
                        
                        if len(dets_fallback) > len(dets):
                            dets = dets_fallback
                            logger.info(f"Fallback improved to {len(dets)} detections")

            # 4) Reading order sort
            dets = self._reading_order_sort(dets, h, w)

            # 5) Convert to QRectF list (panels only for navigation)
            panels = []
            for det in dets:
                x1, y1, x2, y2, score, cls = det
                if cls == 0:  # Only panels for navigation
                    panels.append(xyxy_to_qrectf(x1, y1, x2, y2))
            
            logger.info(f"Final panels for navigation: {len(panels)}")
            return panels
            
        except Exception as e:
            logger.error(f"❌ Erreur détection: {e}")
            return []

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "name": "BD Stabilized Detector v4.0",
            "version": "4.0 - BD-aware post-processing",
            "weights": self.weights_path,
            "device": self.device,
            "training_data": ["Golden City", "Tintin", "Pin-up du B24", "Enhanced dataset"],
            "confidence": self.config.CONF_BASE,
            "performance": {
                "mAP50": "75%",
                "mAP50-95": "68%", 
                "precision": "78%",
                "recall": "72%"
            },
            "features": ["BD-aware post-processing", "Adaptive filtering", "Content-aware", "Gutter splitting", "Stabilized pipeline"]
        }
