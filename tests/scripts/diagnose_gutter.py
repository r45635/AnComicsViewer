"""Diagnose gutter detection on page 5."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
from ancomicsviewer.config import DetectorConfig

DEBUG_DIR = "debug_output/page_005_20260219_155113"

img_bgr = cv2.imread(f"{DEBUG_DIR}/page_render.png")
img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
h, w = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Replicate LAB L-channel with CLAHE
lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
L = lab[:, :, 0]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
L = clahe.apply(L)

config = DetectorConfig(debug=True)

print(f"Image: {w}x{h}")
print(f"L range: {L.min()}-{L.max()}, mean={L.mean():.1f}")

# Step 1: Brightness threshold
bright_percentile = 94
bright_thresh = np.percentile(L, bright_percentile)
print(f"\nbright_thresh (p{bright_percentile}) = {bright_thresh:.1f}")
bright_mask = (L >= bright_thresh).astype(np.uint8) * 255
print(f"bright_mask pixels: {cv2.countNonZero(bright_mask)} ({cv2.countNonZero(bright_mask)/bright_mask.size*100:.1f}%)")

# Step 2: Gradient
grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
grad_mag = np.abs(grad_x) + np.abs(grad_y)
grad_percentile = config.gutter_grad_percentile
grad_low = np.percentile(grad_mag, grad_percentile)
print(f"grad_low (p{grad_percentile}) = {grad_low:.1f}")
uniform_mask = (grad_mag <= grad_low).astype(np.uint8) * 255
print(f"uniform_mask pixels: {cv2.countNonZero(uniform_mask)} ({cv2.countNonZero(uniform_mask)/uniform_mask.size*100:.1f}%)")

# Combined
gutter_raw = cv2.bitwise_and(bright_mask, uniform_mask)
print(f"gutter_raw (bright AND uniform): {cv2.countNonZero(gutter_raw)} ({cv2.countNonZero(gutter_raw)/gutter_raw.size*100:.1f}%)")

# Check gutter rows specifically
for label, y_s, y_e in [("Gutter1 y=133-139", 133, 139),
                          ("Gutter2 y=302-308", 302, 308),
                          ("Gutter3 y=441-447", 441, 447),
                          ("Gutter4 y=587-594", 587, 594)]:
    band_L = L[y_s:y_e+1, :]
    band_bright = bright_mask[y_s:y_e+1, :]
    band_uniform = uniform_mask[y_s:y_e+1, :]
    band_raw = gutter_raw[y_s:y_e+1, :]
    print(f"\n  {label}:")
    print(f"    L: min={band_L.min()} max={band_L.max()} mean={band_L.mean():.1f}")
    print(f"    bright: {cv2.countNonZero(band_bright)}/{band_bright.size} ({cv2.countNonZero(band_bright)/band_bright.size*100:.1f}%)")
    print(f"    uniform: {cv2.countNonZero(band_uniform)}/{band_uniform.size} ({cv2.countNonZero(band_uniform)/band_uniform.size*100:.1f}%)")
    print(f"    combined: {cv2.countNonZero(band_raw)}/{band_raw.size} ({cv2.countNonZero(band_raw)/band_raw.size*100:.1f}%)")

# Step 3: Morphological opening
open_kernel_len = max(int(w * config.gutter_open_kernel_frac), 21)
print(f"\nopen_kernel_len = {open_kernel_len}")

h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel_len, 3))
h_open = cv2.morphologyEx(gutter_raw, cv2.MORPH_OPEN, h_kernel, iterations=1)
print(f"After h_open: {cv2.countNonZero(h_open)} ({cv2.countNonZero(h_open)/h_open.size*100:.1f}%)")

v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, open_kernel_len))
v_open = cv2.morphologyEx(gutter_raw, cv2.MORPH_OPEN, v_kernel, iterations=1)
print(f"After v_open: {cv2.countNonZero(v_open)} ({cv2.countNonZero(v_open)/v_open.size*100:.1f}%)")

combined = cv2.bitwise_or(h_open, v_open)
print(f"After combined open: {cv2.countNonZero(combined)} ({cv2.countNonZero(combined)/combined.size*100:.1f}%)")

# Step 4: Closing
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
after_close = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=1)
print(f"After close: {cv2.countNonZero(after_close)} ({cv2.countNonZero(after_close)/after_close.size*100:.1f}%)")

# Step 5: Filter stripes
from ancomicsviewer.detector.gutter import _filter_stripes
final = _filter_stripes(after_close, w, h, config)
print(f"After stripe filter: {cv2.countNonZero(final)} ({cv2.countNonZero(final)/final.size*100:.1f}%)")

# Check gutter rows in the final mask
for label, y_s, y_e in [("Gutter1", 133, 139), ("Gutter2", 302, 308),
                          ("Gutter3", 441, 447), ("Gutter4", 587, 594)]:
    band = final[y_s:y_e+1, :]
    pct = cv2.countNonZero(band) / band.size * 100 if band.size > 0 else 0
    print(f"  {label}: {pct:.1f}% in final mask")

# Also check: what does panels_from_gutters require?
print("\n--- BUG CHECK: panels_from_gutters requires both h AND v lines ---")
print("This page has only H gutters, no V gutters!")
print("Current code: 'if h_lines and v_lines' => returns empty!")
