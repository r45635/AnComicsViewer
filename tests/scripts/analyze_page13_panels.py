#!/usr/bin/env python3
"""Detailed analysis of page 13 panel positions, especially row 2."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2
import numpy as np
from PySide6.QtCore import QSizeF
from PySide6.QtGui import QImage
from ancomicsviewer import PanelDetector
from ancomicsviewer.config import DetectorConfig

DEBUG_DIR = "debug_output/page_013_20260219_160729"

img_bgr = cv2.imread(f"{DEBUG_DIR}/page_render.png")
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
print(f"Image: {w}x{h}")

page_point_size = QSizeF(345.60, 483.12)
scale = w / page_point_size.width()
print(f"Scale: {scale:.4f} px/pt")

# Run detection
img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
qimg = QImage(img_rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
config = DetectorConfig(debug=True)
det = PanelDetector(config)
rects = det.detect_panels(qimg, page_point_size=page_point_size)

print(f"\n{'='*60}")
print(f"Detected {len(rects)} panels:")
for i, r in enumerate(rects, 1):
    # Convert to px for analysis
    px_x = r.x() * scale
    px_y = r.y() * scale
    px_w = r.width() * scale
    px_h = r.height() * scale
    print(f"  Panel {i}: pts=({r.x():.1f},{r.y():.1f}) {r.width():.1f}x{r.height():.1f}pt"
          f"  px=({px_x:.0f},{px_y:.0f}) {px_w:.0f}x{px_h:.0f}")

# Analyze row 2 structure precisely
print(f"\n{'='*60}")
print("Row 2 detailed structure analysis:")

# Find exact gutter positions by row brightness
margin = int(w * 0.03)
row_means = gray[:, margin:w-margin].mean(axis=1)
med = np.median(row_means)
mx = row_means.max()
thresh = med + 0.70 * (mx - med)

print(f"\nH-gutter positions (full page):")
in_band = False
h_gutters = []
for y in range(h):
    if row_means[y] >= thresh and not in_band:
        band_start = y
        in_band = True
    elif row_means[y] < thresh and in_band:
        t = y - band_start
        if t >= 3:
            h_gutters.append((band_start, y-1))
            print(f"  y={band_start}-{y-1} (t={t})")
        in_band = False

# Row 2: between gutter 1 and gutter 2
if len(h_gutters) >= 2:
    row2_y1 = h_gutters[0][1] + 1
    row2_y2 = h_gutters[1][0]
    print(f"\nRow 2 span: y={row2_y1}-{row2_y2} ({row2_y2-row2_y1}px)")
    
    row2 = gray[row2_y1:row2_y2, :]
    r2h, r2w = row2.shape
    
    # V-gutters in row 2
    col_means = row2[:, margin:w-margin].mean(axis=0)
    col_med = np.median(col_means)
    col_max = col_means.max()
    
    print(f"Col brightness: min={col_means.min():.0f} max={col_max:.0f} med={col_med:.0f}")
    
    # Find V-gutters with detailed position info
    v_thresh = 225.0
    print(f"\nV-gutter positions in row 2 (abs bright >= {v_thresh}):")
    in_band = False
    v_gutters = []
    for x in range(len(col_means)):
        abs_x = x + margin
        if col_means[x] >= v_thresh and not in_band:
            band_start = abs_x
            in_band = True
        elif col_means[x] < v_thresh and in_band:
            t = abs_x - band_start
            if t >= 3 and band_start > margin + 3 and abs_x < w - margin - 3:
                v_gutters.append((band_start, abs_x - 1))
                band_brightness = col_means[band_start-margin:abs_x-margin].mean()
                print(f"  x={band_start}-{abs_x-1} (t={t}, mean={band_brightness:.0f})")
            in_band = False
    
    # H-sub-gutters per column
    if v_gutters:
        col_boundaries = [(0, 0)] + v_gutters + [(w-1, w-1)]
        for j in range(len(col_boundaries) - 1):
            x1 = col_boundaries[j][1] + 1
            x2 = col_boundaries[j+1][0]
            cw = x2 - x1
            if cw < 30:
                continue
            
            cell = gray[row2_y1:row2_y2, x1:x2]
            cell_margin = max(3, int(cw * 0.05))
            interior = cell[:, cell_margin:cw-cell_margin]
            if interior.size == 0:
                continue
            
            crow_means = interior.mean(axis=1)
            cmed = np.median(crow_means)
            cmx = crow_means.max()
            
            print(f"\n  Column x={x1}-{x2} ({cw}px):")
            print(f"    Row brightness: min={crow_means.min():.0f} max={cmx:.0f} med={cmed:.0f}")
            
            if cmx - cmed > 35:
                sub_thresh = cmed + 0.60 * (cmx - cmed)
                print(f"    Sub-gutter threshold: {sub_thresh:.0f}")
                in_band = False
                for y in range(r2h):
                    abs_y = y + row2_y1
                    if crow_means[y] >= sub_thresh and not in_band:
                        band_start = y
                        in_band = True
                    elif crow_means[y] < sub_thresh and in_band:
                        t = y - band_start
                        sy = band_start + row2_y1
                        ey = y - 1 + row2_y1
                        if t >= 3:
                            band_b = crow_means[band_start:y].mean()
                            print(f"    H-sub-gutter abs_y={sy}-{ey} (t={t}, mean={band_b:.0f})")
                        in_band = False
            else:
                print(f"    No contrast for sub-gutters (contrast={cmx-cmed:.0f})")

# Also check overlaps between detected panels
print(f"\n{'='*60}")
print("Panel overlap analysis:")
for i in range(len(rects)):
    for j in range(i+1, len(rects)):
        ri, rj = rects[i], rects[j]
        inter = ri.intersected(rj)
        if not inter.isEmpty():
            overlap_area = inter.width() * inter.height()
            area_i = ri.width() * ri.height()
            area_j = rj.width() * rj.height()
            print(f"  Panel {i+1} & {j+1}: overlap={overlap_area:.0f}pt² "
                  f"({100*overlap_area/min(area_i,area_j):.1f}% of smaller)")
