#!/usr/bin/env python3
"""Deep structural analysis of page 13 to find internal panel divisions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import cv2
import numpy as np

img = cv2.imread('debug_output/page_013_20260219_160729/page_render.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
print(f"Image: {w}x{h}")

# Expected layout from user:
# Row 1 (y~8..184): 1 wide panel
# Row 2 (y~189..512): 3 panels + 2 stacked -> 5 cells
# Row 3 (y~519..?): 1 panel (possibly thin)
# Row 4 (y~600..952): 1 big panel
# Total expected: ~8 panels

# Analyze Row 2 region in detail for vertical divisions
row2_y1, row2_y2 = 189, 512
row2 = gray[row2_y1:row2_y2, :]
r2h, r2w = row2.shape
print(f"\n=== Row 2 region: y={row2_y1}-{row2_y2}, size={r2w}x{r2h} ===")

# Column-wise brightness for vertical gutters in row 2
col_means = row2.mean(axis=0)
print(f"Col brightness in row2: min={col_means.min():.0f}, max={col_means.max():.0f}, median={np.median(col_means):.0f}")

# Find vertical bright columns (gutters within row 2)
col_median = np.median(col_means)
col_max = col_means.max()
v_thresh = col_median + 0.5 * (col_max - col_median)
print(f"V-gutter threshold: {v_thresh:.0f}")

in_band = False
band_start = 0
for x in range(r2w):
    if col_means[x] >= v_thresh and not in_band:
        band_start = x
        in_band = True
    elif col_means[x] < v_thresh and in_band:
        t = x - band_start
        if t >= 3:
            mb = col_means[band_start:x].mean()
            print(f"  V-gutter x={band_start}-{x-1} (t={t}, brightness={mb:.0f})")
        in_band = False

# Row-wise brightness within Row 2 for horizontal sub-divisions
row_means_r2 = row2.mean(axis=1)
print(f"\nRow brightness in row2: min={row_means_r2.min():.0f}, max={row_means_r2.max():.0f}")
r2_median = np.median(row_means_r2)
r2_max = row_means_r2.max()
h_thresh = r2_median + 0.5 * (r2_max - r2_median)
print(f"H-gutter threshold within row2: {h_thresh:.0f}")

in_band = False
for y in range(r2h):
    abs_y = y + row2_y1
    if row_means_r2[y] >= h_thresh and not in_band:
        band_start = y
        in_band = True
    elif row_means_r2[y] < h_thresh and in_band:
        t = y - band_start
        if t >= 3:
            mb = row_means_r2[band_start:y].mean()
            print(f"  H-sub-gutter abs_y={band_start+row2_y1}-{y-1+row2_y1} (t={t}, brightness={mb:.0f})")
        in_band = False

# Edge detection within row 2 to find panel borders
print("\n=== Edge analysis in Row 2 ===")
edges = cv2.Canny(row2, 50, 150)

# Vertical line detection using HoughLinesP in row2
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=r2h//3, maxLineGap=10)
v_lines = []
h_lines_r2 = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx < 10 and dy > r2h * 0.3:  # near-vertical
            v_lines.append((x1, y1 + row2_y1, x2, y2 + row2_y1))
        elif dy < 10 and dx > r2w * 0.1:  # near-horizontal
            h_lines_r2.append((x1, y1 + row2_y1, x2, y2 + row2_y1))

print(f"Hough vertical lines in row2: {len(v_lines)}")
for vl in sorted(v_lines, key=lambda l: l[0]):
    print(f"  x={vl[0]}-{vl[2]}, y={vl[1]}-{vl[3]}")

print(f"Hough horizontal lines in row2: {len(h_lines_r2)}")
for hl in sorted(h_lines_r2, key=lambda l: l[1]):
    print(f"  x={hl[0]}-{hl[2]}, y={hl[1]}-{hl[3]}")

# Also check LSD (Line Segment Detector) for row 2
lsd = cv2.createLineSegmentDetector(0)
lines_lsd, _, _, _ = lsd.detect(row2)
v_lsd = []
h_lsd = []
if lines_lsd is not None:
    for seg in lines_lsd:
        x1, y1, x2, y2 = seg[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length < 30:
            continue
        angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
        if abs(angle) < 15 or abs(angle) > 165:  # horizontal
            if length > r2w * 0.1:
                h_lsd.append((x1, y1+row2_y1, x2, y2+row2_y1, length))
        elif abs(angle - 90) < 15 or abs(angle + 90) < 15:  # vertical
            if length > r2h * 0.15:
                v_lsd.append((x1, y1+row2_y1, x2, y2+row2_y1, length))

print(f"\nLSD vertical lines in row2: {len(v_lsd)}")
for vl in sorted(v_lsd, key=lambda l: l[0])[:10]:
    print(f"  x={vl[0]:.0f}-{vl[2]:.0f}, y={vl[1]:.0f}-{vl[3]:.0f}, len={vl[4]:.0f}")

print(f"LSD horizontal lines in row2: {len(h_lsd)}")
for hl in sorted(h_lsd, key=lambda l: l[1])[:10]:
    print(f"  x={hl[0]:.0f}-{hl[2]:.0f}, y={hl[1]:.0f}-{hl[3]:.0f}, len={hl[4]:.0f}")

# Analyze brightness across specific vertical columns within row 2
print("\n=== Brightness profile at key x positions in Row 2 ===")
for probe_x in range(0, r2w, 20):
    strip = row2[:, max(0,probe_x-2):probe_x+3]
    if strip.size > 0:
        m = strip.mean()
        if m > 200:
            print(f"  x={probe_x}: bright (mean={m:.0f})")

# Full page: check the row3 area more carefully (y=519-600)
print("\n=== Row 3 area analysis (y=513-600) ===")
row3_area = gray[513:600, :]
print(f"Region size: {row3_area.shape[1]}x{row3_area.shape[0]}")
row_means_r3 = row3_area.mean(axis=1)
for y in range(row3_area.shape[0]):
    abs_y = y + 513
    m = row_means_r3[y]
    if m > 200 or (y % 10 == 0):
        print(f"  y={abs_y}: mean={m:.0f}{'  <-- bright' if m > 220 else ''}")
