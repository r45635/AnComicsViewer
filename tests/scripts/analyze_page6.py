"""Analyse page 6 detection: gutter structure, panel boundaries, and offset issues."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import cv2
from ancomicsviewer.detector.gutter import gutter_based_detection, _detect_gutters_by_profile, _detect_gutters_hierarchical
from ancomicsviewer.config import DetectorConfig
from ancomicsviewer import PanelDetector
from PySide6.QtCore import QRectF, QSizeF
from PySide6.QtGui import QImage
import fitz

PDF_PATH = "samples_PDF/Le sang du dragon - 01 - Au-delà des brumes (2005).pdf"
PAGE_IDX = 5  # 0-indexed, page 6

def render_page(scale=1.0):
    doc = fitz.open(PDF_PATH)
    page = doc[PAGE_IDX]
    pps = page.rect
    mat = fitz.Matrix(scale * 150/72, scale * 150/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return img, QSizeF(pps.width, pps.height)

def analyze_gutter_structure(img, label):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    print(f"\n{'='*60}")
    print(f"{label}: {w}x{h}")
    print(f"{'='*60}")
    
    # Row means for H-gutter analysis
    row_means = gray.mean(axis=1)
    page_median = np.median(row_means)
    row_max = row_means.max()
    thresh_70 = page_median + 0.70 * (row_max - page_median)
    thresh_80 = page_median + 0.80 * (row_max - page_median)
    
    print(f"Row brightness: median={page_median:.1f}, max={row_max:.1f}")
    print(f"Threshold 0.70: {thresh_70:.1f}")
    print(f"Threshold 0.80: {thresh_80:.1f}")
    
    # Find bright rows
    bright_rows_70 = np.where(row_means >= thresh_70)[0]
    bright_rows_80 = np.where(row_means >= thresh_80)[0]
    
    def group_runs(arr, min_t=3):
        if len(arr) == 0:
            return []
        groups = []
        start = arr[0]
        prev = arr[0]
        for v in arr[1:]:
            if v - prev > 3:
                if prev - start + 1 >= min_t:
                    groups.append((start, prev))
                start = v
            prev = v
        if prev - start + 1 >= min_t:
            groups.append((start, prev))
        return groups
    
    h_gutters_70 = group_runs(bright_rows_70)
    h_gutters_80 = group_runs(bright_rows_80)
    
    print(f"\nH-gutters (0.70 thresh): {len(h_gutters_70)}")
    for y1, y2 in h_gutters_70:
        mean_b = row_means[y1:y2+1].mean()
        print(f"  y={y1}-{y2} (t={y2-y1+1}, brightness={mean_b:.0f})")
    
    print(f"\nH-gutters (0.80 thresh): {len(h_gutters_80)}")
    for y1, y2 in h_gutters_80:
        mean_b = row_means[y1:y2+1].mean()
        print(f"  y={y1}-{y2} (t={y2-y1+1}, brightness={mean_b:.0f})")
    
    # Column analysis
    col_means = gray.mean(axis=0)
    col_median = np.median(col_means)
    col_max = col_means.max()
    col_thresh = col_median + 0.80 * (col_max - col_median)
    
    bright_cols = np.where(col_means >= col_thresh)[0]
    v_gutters = group_runs(bright_cols)
    print(f"\nV-gutters (0.80 thresh): {len(v_gutters)}")
    for x1, x2 in v_gutters:
        mean_b = col_means[x1:x2+1].mean()
        print(f"  x={x1}-{x2} (t={x2-x1+1}, brightness={mean_b:.0f})")
    
    return h_gutters_80, v_gutters

def analyze_bottom_row(img, h_gutters):
    """Analyze the bottom row of panels to find vertical splits."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    if len(h_gutters) < 2:
        print("\nNot enough H-gutters to find bottom row")
        return
    
    # Bottom row is between second-last and last H-gutter
    # (last H-gutter is bottom margin)
    row_start = h_gutters[-2][1] + 1
    row_end = h_gutters[-1][0] - 1
    if row_end <= row_start:
        row_end = h - 1
    print(f"\n--- Bottom row analysis: y={row_start}-{row_end} ({row_end-row_start+1}px) ---")
    
    row_crop = gray[row_start:row_end+1, :]
    col_means = row_crop.mean(axis=0)
    col_median = np.median(col_means)
    col_max = col_means.max()
    
    print(f"Column brightness in bottom row: median={col_median:.1f}, max={col_max:.1f}")
    
    # Find V-gutters in this row
    for factor in [0.70, 0.80]:
        thresh = col_median + factor * (col_max - col_median)
        print(f"\n  V-gutters in bottom row (factor={factor}, thresh={thresh:.1f}):")
        bright_cols = np.where(col_means >= thresh)[0]
        groups = []
        if len(bright_cols) > 0:
            start = bright_cols[0]
            prev = bright_cols[0]
            for v in bright_cols[1:]:
                if v - prev > 3:
                    if prev - start + 1 >= 3:
                        groups.append((start, prev))
                    start = v
                prev = v
            if prev - start + 1 >= 3:
                groups.append((start, prev))
        
        for x1, x2 in groups:
            band = row_crop[:, x1:x2+1]
            band_mean = band.mean()
            # Check coverage (how much of band is bright)
            bright_pct = (band >= 225).mean()
            print(f"    x={x1}-{x2} (t={x2-x1+1}, mean={band_mean:.0f}, bright_pct={bright_pct:.1%})")
    
    # Detailed column profile in bottom row for panel 5 area (x=387-530)
    print(f"\n  Column means in panel 5 area (x=370-550):")
    for x in range(370, min(550, w), 10):
        print(f"    x={x}: {col_means[x]:.1f}", end="")
    print()
    
    # Check for H sub-gutters in each cell of bottom row
    # First find V-gutters in bottom row
    thresh = col_median + 0.70 * (col_max - col_median)
    bright_cols = np.where(col_means >= thresh)[0]
    groups = []
    if len(bright_cols) > 0:
        start = bright_cols[0]
        prev = bright_cols[0]
        for v in bright_cols[1:]:
            if v - prev > 3:
                if prev - start + 1 >= 3:
                    groups.append((start, prev))
                start = v
            prev = v
        if prev - start + 1 >= 3:
            groups.append((start, prev))
    
    # Build cells from V-gutter splits
    v_splits = [0] + [(x1+x2)//2 for x1, x2 in groups] + [w]
    print(f"\n  Bottom row cells from V-gutters:")
    for i in range(len(v_splits)-1):
        x1, x2 = v_splits[i], v_splits[i+1]
        cell = row_crop[:, x1:x2]
        cell_h, cell_w = cell.shape
        # Row means within cell
        cell_row_means = cell.mean(axis=1)
        cell_median = np.median(cell_row_means)
        cell_max = cell_row_means.max()
        contrast = cell_max - cell_median
        
        print(f"    Cell {i+1}: x={x1}-{x2} ({cell_w}px)")
        print(f"      Row brightness: median={cell_median:.1f}, max={cell_max:.1f}, contrast={contrast:.1f}")
        
        if contrast >= 30:
            sub_thresh = cell_median + 0.65 * contrast
            bright_rows = np.where(cell_row_means >= sub_thresh)[0]
            sub_groups = []
            if len(bright_rows) > 0:
                s = bright_rows[0]
                p = bright_rows[0]
                for v in bright_rows[1:]:
                    if v - p > 3:
                        if p - s + 1 >= 3:
                            sub_groups.append((s, p))
                        s = v
                    p = v
                if p - s + 1 >= 3:
                    sub_groups.append((s, p))
            
            for sy1, sy2 in sub_groups:
                band = cell[sy1:sy2+1, :]
                band_mean = band.mean()
                bright_pct = (band >= 225).mean()
                abs_y = row_start + sy1
                print(f"      Sub-H-gutter: y={sy1}-{sy2} (abs_y={abs_y}-{row_start+sy2}, t={sy2-sy1+1}, mean={band_mean:.0f}, bright225%={bright_pct:.1%})")

def run_full_pipeline(img, page_point_size):
    """Run full detection to get final panels."""
    h, w = img.shape[:2]
    qimg = QImage(img.data, w, h, 3*w, QImage.Format.Format_RGB888)
    
    config = DetectorConfig(debug=True)
    det = PanelDetector(config)
    rects = det.detect_panels(qimg, page_point_size=page_point_size)
    
    print(f"\n{'='*60}")
    print(f"Full pipeline: {len(rects)} panels")
    print(f"{'='*60}")
    for i, r in enumerate(rects, 1):
        px_x = r.x() * w / page_point_size.width()
        px_y = r.y() * h / page_point_size.height()
        px_w = r.width() * w / page_point_size.width()
        px_h = r.height() * h / page_point_size.height()
        print(f"  Panel {i}: pts=({r.x():.1f},{r.y():.1f}) {r.width():.1f}x{r.height():.1f}  px=({px_x:.0f},{px_y:.0f}) {px_w:.0f}x{px_h:.0f}")
    return rects

def run_gutter_only(img, page_point_size, scale_label):
    """Run just gutter detection to see what it produces."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    config = DetectorConfig(debug=True)
    
    # Create L channel from LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    
    rects = gutter_based_detection(gray, L, w, h, page_point_size, config)
    print(f"\n--- Gutter-only ({scale_label}): {len(rects)} panels ---")
    for i, r in enumerate(rects, 1):
        px_x = r.x() * w / page_point_size.width()
        px_y = r.y() * h / page_point_size.height()
        px_w = r.width() * w / page_point_size.width()
        px_h = r.height() * h / page_point_size.height()
        print(f"  Panel {i}: pts=({r.x():.1f},{r.y():.1f}) {r.width():.1f}x{r.height():.1f}  px=({px_x:.0f},{px_y:.0f}) {px_w:.0f}x{px_h:.0f}")
    return rects

def main():
    for scale in [0.6, 1.0, 1.5]:
        img, pps = render_page(scale)
        label = f"Scale {scale}x"
        h_gutters, v_gutters = analyze_gutter_structure(img, label)
        
        if scale == 1.0:
            analyze_bottom_row(img, h_gutters)
        
        run_gutter_only(img, pps, label)
    
    # Full pipeline at native scale
    img, pps = render_page(1.0)
    run_full_pipeline(img, pps)

if __name__ == "__main__":
    main()
