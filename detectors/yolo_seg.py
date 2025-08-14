from typing import List, Tuple
import numpy as np
import cv2
from ultralytics import YOLO
from PySide6.QtCore import QRectF, QSizeF
from PySide6.QtGui import QImage
from .base import BasePanelDetector

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
    def __init__(self, weights: str, conf: float = 0.25, iou: float = 0.5, rtl: bool=False):
        self.model = YOLO(weights)
        self.conf, self.iou = conf, iou
        self.reading_rtl = rtl

    def detect_panels(self, qimage: QImage, page_point_size: QSizeF) -> List[QRectF]:
        rgb = qimage_to_rgb(qimage)
        H, W = rgb.shape[:2]
        s = W / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0
        r = self.model.predict(source=rgb, imgsz=1280, conf=self.conf, iou=self.iou, verbose=False)[0]
        panels: List[QRectF] = []
        titles = []
        if r.masks is not None:
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
        keep.sort(key=lambda r: (r.top(), -r.left()) if self.reading_rtl else (r.top(), r.left()))
        return keep
