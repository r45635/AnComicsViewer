from typing import List, Tuple
import numpy as np
import cv2
import torch

# Set matplotlib backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

from ultralytics import YOLO
from PySide6.QtCore import QRectF, QSizeF
from PySide6.QtGui import QImage
from .base import BasePanelDetector
from .postproc import snap_rect_to_gutters_rgb
from .reading_order import sort_reading_order

def _device():
    """Get optimal device for inference."""
    return "mps" if torch.backends.mps.is_available() else "cpu"port List, Tuple
import numpy as np
import cv2

# Set matplotlib backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

from ultralytics import YOLO
from PySide6.QtCore import QRectF, QSizeF
from PySide6.QtGui import QImage
from .base import BasePanelDetector
from .postproc import snap_rect_to_gutters_rgb
from .reading_order import sort_reading_order

CLASS_PANEL, CLASS_BALLOON, CLASS_TITLE = 0, 1, 2

def qimage_to_rgb(qimg: QImage) -> np.ndarray:
    if qimg.format() != QImage.Format.Format_RGBA8888:
        qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    h, w = qimg.height(), qimg.width()
    bpl = qimg.bytesPerLine()
    arr = np.frombuffer(bytes(qimg.constBits()), dtype=np.uint8).reshape(h, bpl)[:, :w*4]
    rgba = arr.reshape(h, w, 4)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

def iou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua>0 else 0.0

class YoloSegPanelDetector(BasePanelDetector):
    def __init__(self, weights: str, conf: float = 0.1, iou: float = 0.5, imgsz_infer: int = 1280, rtl: bool=False, row_band_frac: float = 0.06):
        # Handle PyTorch security changes for YOLO model loading
        import torch
        
        # Temporarily patch torch.load to allow loading YOLO models
        original_load = torch.load
        def patched_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
            if weights_only is None:
                weights_only = False  # Use False for YOLO models
            return original_load(f, map_location=map_location, pickle_module=pickle_module, 
                               weights_only=weights_only, **kwargs)
        
        torch.load = patched_load
        try:
            self.model = YOLO(weights)
        finally:
            torch.load = original_load  # Restore original function
            
        self.conf, self.iou = conf, iou
        self.imgsz_infer = imgsz_infer
        self.reading_rtl = rtl
        self.row_band_frac = row_band_frac

    def detect_panels(self, qimage: QImage, page_point_size: QSizeF) -> List[QRectF]:
        rgb = qimage_to_rgb(qimage)
        H, W = rgb.shape[:2]
        s = W / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0
        r = self.model.predict(source=rgb, imgsz=self.imgsz_infer, conf=self.conf, iou=self.iou, verbose=False)[0]
        panels: List[QRectF] = []
        titles = []
        
        # Check for bounding box detection results
        if r.boxes is not None and len(r.boxes) > 0:
            for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                x1,y1,x2,y2 = box.tolist()
                if cls == CLASS_PANEL:
                    panels.append(QRectF(x1/s, y1/s, (x2-x1)/s, (y2-y1)/s))
                elif cls == CLASS_TITLE:
                    titles.append((x1,y1,x2,y2))
        
        # Legacy segmentation mask support (if available)
        elif r.masks is not None:
            for box, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                x1,y1,x2,y2 = box.tolist()
                if cls == CLASS_PANEL:
                    panels.append(QRectF(x1/s, y1/s, (x2-x1)/s, (y2-y1)/s))
                elif cls == CLASS_TITLE:
                    titles.append((x1,y1,x2,y2))
        
        def overlaps_title(rect: QRectF) -> bool:
            a = (rect.left()*s, rect.top()*s, rect.right()*s, rect.bottom()*s)
            return any(iou_xyxy(a,t) > 0.50 for t in titles)
        
        keep = [r for r in panels if not overlaps_title(r)]

        # — filtre de taille relatif + ordre de lecture par rangées —
        page_area = float(page_point_size.width() * page_point_size.height() or 1.0)
        min_area = page_area * 0.008  # 0.8% de la page

        keep = [r for r in keep if (r.width()*r.height()) >= min_area]

        # Regroupement en rangées (cohérent avec l'heuristique)
        # Regroupement en rangées et ordre de lecture
        ordered = sort_reading_order(keep, page_point_size, self.reading_rtl, self.row_band_frac)
        
        # Post-traitement : snap sur les gouttières
        snapped = []
        for rect in ordered:
            snapped_rect = snap_rect_to_gutters_rgb(rgb, rect, page_point_size, s)
            snapped.append(snapped_rect)
        
        return snapped
