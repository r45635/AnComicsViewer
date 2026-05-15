#!/usr/bin/env python3
"""Analyze page 13 debug output to understand detection issues."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2
import numpy as np

img = cv2.imread('debug_output/page_013_20260219_160729/page_render.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
print(f"Image: {w}x{h}")

# Row brightness profile
row_means = gray.mean(axis=1)
print(f"Row brightness: min={row_means.min():.0f}, max={row_means.max():.0f}, median={np.median(row_means):.0f}")

# Find bright bands (gutters)
thresh = np.median(row_means) + 0.6 * (row_means.max() - np.median(row_means))
print(f"Gutter threshold: {thresh:.0f}")

in_band = False
band_start = 0
bands = []
for y in range(h):
    if row_means[y] >= thresh and not in_band:
        band_start = y
        in_band = True
    elif row_means[y] < thresh and in_band:
        thickness = y - band_start
        if thickness >= 3:
            mean_b = row_means[band_start:y].mean()
            bands.append((band_start, y - 1, thickness, mean_b))
            print(f"  H-gutter y={band_start}-{y-1} (t={thickness}, brightness={mean_b:.0f})")
        in_band = False

# Column brightness for vertical gutters
col_means = gray.mean(axis=0)
col_thresh = np.median(col_means) + 0.6 * (col_means.max() - np.median(col_means))
print(f"\nCol brightness: min={col_means.min():.0f}, max={col_means.max():.0f}")
print(f"Col gutter threshold: {col_thresh:.0f}")

in_band = False
for x in range(w):
    if col_means[x] >= col_thresh and not in_band:
        band_start = x
        in_band = True
    elif col_means[x] < col_thresh and in_band:
        thickness = x - band_start
        if thickness >= 3:
            mean_b = col_means[band_start:x].mean()
            print(f"  V-gutter x={band_start}-{x-1} (t={thickness}, brightness={mean_b:.0f})")
        in_band = False

# Detected panels analysis
panels = [
    (16, 8, 673, 176),
    (63, 189, 521, 323),
    (23, 519, 666, 63),
    (16, 600, 673, 352),
]
print(f"\nDetected panels ({len(panels)}):")
for i, (x, y, pw, ph) in enumerate(panels, 1):
    region = gray[y:y+ph, x:x+pw]
    mean_b = region.mean()
    dark_pct = (region < 128).mean() * 100
    print(f"  Panel {i}: ({x},{y}) {pw}x{ph} - mean={mean_b:.0f}, dark_pct={dark_pct:.1f}%")

# Run actual detection to see full log
print("\n" + "="*60)
print("Running detection on page 13 render...")
print("="*60)

from PySide6.QtCore import QSizeF
from PySide6.QtGui import QImage
from ancomicsviewer import PanelDetector
from ancomicsviewer.config import DetectorConfig

config = DetectorConfig(debug=True)
detector = PanelDetector(config)

img_bgr = cv2.imread('debug_output/page_013_20260219_160729/page_render.png')
h, w = img_bgr.shape[:2]
img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
qimg = QImage(img_rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)

page_point_size = QSizeF(345.60, 483.12)

rects = detector.detect_panels(qimg, page_point_size=page_point_size)
print(f"\nFinal: {len(rects)} panels")
for i, r in enumerate(rects, 1):
    print(f"  Panel {i}: ({r.x():.0f},{r.y():.0f}) {r.width():.0f}x{r.height():.0f}")
