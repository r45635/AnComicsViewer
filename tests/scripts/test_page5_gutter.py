"""Test gutter detection fix on real page 5 image."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
from PySide6.QtCore import QRectF, QSizeF
from PySide6.QtGui import QImage
from ancomicsviewer import PanelDetector
from ancomicsviewer.config import DetectorConfig

DEBUG_DIR = "debug_output/page_005_20260219_155113"

# Load the actual page render
img_bgr = cv2.imread(f"{DEBUG_DIR}/page_render.png")
if img_bgr is None:
    print("ERROR: Could not load page_render.png")
    sys.exit(1)

h, w = img_bgr.shape[:2]
img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
qimg = QImage(img_rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)

config = DetectorConfig(debug=True)
det = PanelDetector(config)
page_size = QSizeF(345.60, 487.80)

rects = det.detect_panels(qimg, page_point_size=page_size)

print(f"\n{'='*50}")
print(f"Page 5: detected {len(rects)} panels (expected 5)")
for i, r in enumerate(rects):
    print(f"  Panel {i+1}: ({r.x():.0f},{r.y():.0f}) {r.width():.0f}x{r.height():.0f}")

if len(rects) == 5:
    print("\nSUCCESS: 5 panels detected!")
elif len(rects) >= 4:
    print(f"\nPARTIAL: {len(rects)} panels (expected 5)")
else:
    print(f"\nFAIL: only {len(rects)} panels")
