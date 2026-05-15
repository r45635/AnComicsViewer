#!/usr/bin/env python3
"""Compare hierarchical gutter detection at different scales for page 13."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2
import numpy as np
from PySide6.QtCore import QSizeF, QRectF
from ancomicsviewer.config import DetectorConfig
from ancomicsviewer.detector.gutter import gutter_based_detection

DEBUG_DIR = "debug_output/page_013_20260219_160729"
img_bgr = cv2.imread(f"{DEBUG_DIR}/page_render.png")
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
page_size = QSizeF(345.60, 483.12)

config = DetectorConfig(debug=True)

for scale_factor in [0.6, 1.0, 1.5]:
    h, w = gray.shape
    if abs(scale_factor - 1.0) > 0.01:
        nw, nh = int(w * scale_factor), int(h * scale_factor)
        g = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LINEAR)
        bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LINEAR)
    else:
        g, bgr = gray, img_bgr
        nw, nh = w, h
    
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    
    print(f"\n{'='*60}")
    print(f"Scale {scale_factor}x: {nw}x{nh}")
    print(f"{'='*60}")
    
    rects = gutter_based_detection(g, L, nw, nh, page_size, config, bgr)
    
    px_scale = nw / page_size.width()
    print(f"\nGutter result: {len(rects)} panels")
    for i, r in enumerate(rects, 1):
        px_x = r.x() * px_scale
        px_y = r.y() * px_scale
        px_w = r.width() * px_scale
        px_h = r.height() * px_scale
        print(f"  Panel {i}: pts=({r.x():.1f},{r.y():.1f}) {r.width():.1f}x{r.height():.1f} "
              f"px=({px_x:.0f},{px_y:.0f}) {px_w:.0f}x{px_h:.0f}")
