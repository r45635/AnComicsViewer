"""Smoke test for PanelDetector using a synthetic image.

Creates an image with 3 white rectangles on dark background and runs PanelDetector.detect_panels
to verify it finds at least one rectangle. Prints the detection result as JSON.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from AnComicsViewer import PanelDetector
from PySide6.QtGui import QImage
import numpy as np
import cv2


def make_test_qimage(w=800, h=1200):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    # white rectangles
    cv2.rectangle(img, (50, 100), (750, 400), (255,255,255,255), -1)
    cv2.rectangle(img, (60, 450), (380, 950), (255,255,255,255), -1)
    cv2.rectangle(img, (420, 450), (740, 950), (255,255,255,255), -1)
    # convert to QImage (note: numpy memory layout)
    qimg = QImage(img.data, w, h, QImage.Format.Format_RGBA8888)
    return qimg


def run():
    det = PanelDetector(debug=True)
    qimg = make_test_qimage()
    
    # Utiliser QSizeF au lieu d'un tuple
    from PySide6.QtCore import QSizeF
    page_size = QSizeF(800, 1200)
    
    rects = det.detect_panels(qimg, page_point_size=page_size)
    out = [{"x": r.x(), "y": r.y(), "w": r.width(), "h": r.height()} for r in rects]
    print(json.dumps({"count": len(out), "rects": out}, indent=2))


if __name__ == "__main__":
    run()
