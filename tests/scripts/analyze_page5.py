"""Analyze the page render to understand actual panel layout."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

DEBUG_DIR = "debug_output/page_005_20260219_155113"

img = cv2.imread(f"{DEBUG_DIR}/page_render.png")
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Image: {w}x{h}")

# Find horizontal gutters (bright rows spanning most of the width)
h_profile = gray.mean(axis=1)
bright_thresh = 200
bright_rows = np.where(h_profile > bright_thresh)[0]

groups = []
if len(bright_rows) > 0:
    start = bright_rows[0]
    for i in range(1, len(bright_rows)):
        if bright_rows[i] != bright_rows[i-1] + 1:
            groups.append((start, bright_rows[i-1]))
            start = bright_rows[i]
    groups.append((start, bright_rows[-1]))

print(f"\nHorizontal bright bands (H gutters):")
for s, e in groups:
    thickness = e - s + 1
    if thickness >= 3:
        print(f"  rows {s}-{e}  thickness={thickness}  y_mid={((s+e)/2):.0f}")

# Find vertical gutters
v_profile = gray.mean(axis=0)
bright_cols = np.where(v_profile > bright_thresh)[0]
v_groups = []
if len(bright_cols) > 0:
    start = bright_cols[0]
    for i in range(1, len(bright_cols)):
        if bright_cols[i] != bright_cols[i-1] + 1:
            v_groups.append((start, bright_cols[i-1]))
            start = bright_cols[i]
    v_groups.append((start, bright_cols[-1]))

print(f"\nVertical bright bands (V gutters):")
for s, e in v_groups:
    thickness = e - s + 1
    if thickness >= 3:
        print(f"  cols {s}-{e}  thickness={thickness}  x_mid={((s+e)/2):.0f}")

# Check for vertical splits within horizontal bands
# Look at specific Y ranges for vertical separation
print("\n--- Analyzing row-by-row for vertical splits ---")
# Check the top panel area (y=58 to y=302) - is there a vertical gutter?
for label, y_start, y_end in [("Top band y=58-302", 58, 302),
                                ("Band2 y=306-440", 306, 440),
                                ("Band3 y=400-587", 400, 587),
                                ("Bottom y=593-983", 593, 983)]:
    y_s = max(0, y_start)
    y_e = min(h, y_end)
    strip = gray[y_s:y_e, :]
    col_avg = strip.mean(axis=0)
    # Find bright vertical columns in this strip
    bright_c = np.where(col_avg > bright_thresh)[0]
    if len(bright_c) > 0:
        vc_groups = []
        start = bright_c[0]
        for i in range(1, len(bright_c)):
            if bright_c[i] != bright_c[i-1] + 1:
                vc_groups.append((start, bright_c[i-1]))
                start = bright_c[i]
        vc_groups.append((start, bright_c[-1]))
        interior_gutters = [(s, e) for s, e in vc_groups if s > 30 and e < w-30 and (e-s+1) >= 3]
        if interior_gutters:
            print(f"  {label}: interior V gutters at {interior_gutters}")
        else:
            print(f"  {label}: no interior vertical gutter")
    else:
        print(f"  {label}: no bright columns")

# Also check with lower threshold for colored gutters
print("\n--- Lower threshold (170) for colored gutters ---")
for label, y_start, y_end in [("Top band y=58-302", 58, 302)]:
    strip = gray[y_start:y_end, :]
    col_avg = strip.mean(axis=0)
    # Find local minima = darker columns that could be borders
    # Or look for columns significantly brighter than neighbors
    col_min = col_avg.min()
    col_max = col_avg.max()
    print(f"  {label}: col brightness min={col_min:.0f} max={col_max:.0f}")
    # Check specific column values around mid-width
    mid = w // 2
    print(f"  Columns around mid ({mid}): {col_avg[mid-5:mid+5].astype(int)}")
    # Look for edges
    col_grad = np.abs(np.diff(col_avg))
    peaks = np.where(col_grad > 20)[0]
    if len(peaks) > 0:
        # Filter interior peaks  
        interior = peaks[(peaks > 50) & (peaks < w-50)]
        if len(interior) > 0:
            print(f"  Strong vertical edges at cols: {interior[:20]}")

print("\n--- Current detection ---")
print("Panel 1: x=18,y=58  w=605 h=244")
print("Panel 2: x=18,y=306 w=683 h=135")  
print("Panel 3: x=19,y=400 w=680 h=187")
print("Panel 4: x=19,y=593 w=681 h=390")
print("Expected: 5 panels")
