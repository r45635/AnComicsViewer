#!/usr/bin/env python3
"""Diagnostic tool to visualize detection steps."""

import sys
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtPdf import QPdfDocument
from PySide6.QtCore import QSize

from ancomicsviewer.detector import PanelDetector
from ancomicsviewer.config import DetectorConfig
from ancomicsviewer.image_utils import qimage_to_numpy_rgba


def diagnose_page(pdf_path: str, page_num: int):
    """Show intermediate steps of panel detection."""
    print(f"=== Diagnosing page {page_num + 1} ===\n")
    
    doc = QPdfDocument()
    doc.load(pdf_path)
    page_size = doc.pagePointSize(page_num)
    
    # Render page
    dpi = 150.0
    scale = dpi / 72.0
    image_size = QSize(int(page_size.width() * scale), int(page_size.height() * scale))
    qimage = doc.render(page_num, image_size)
    
    # Convert to numpy
    rgba = qimage_to_numpy_rgba(qimage)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    print(f"Image size: {gray.shape[1]}x{gray.shape[0]}")
    print(f"Page size: {page_size.width():.0f}x{page_size.height():.0f}pt\n")
    
    # Apply detection with intermediate steps
    config = DetectorConfig(debug=True)
    
    # Adaptive threshold
    k = config.adaptive_block | 1
    gray_smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(
        gray_smooth, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        k, config.adaptive_C
    )
    
    # Morphology
    kernel = np.ones((config.morph_kernel, config.morph_kernel), np.uint8)
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=config.morph_iter)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Detection steps:")
    print(f"  1. Bilateral filter: {gray_smooth.shape}")
    print(f"  2. Adaptive threshold: {th.shape}, kernel={k}, C={config.adaptive_C}")
    print(f"  3. Morphology CLOSE: kernel={config.morph_kernel}x{config.morph_kernel}, iter={config.morph_iter}")
    print(f"  4. Found {len(contours)} contours\n")
    
    # Analyze contours
    page_area = page_size.width() * page_size.height()
    scale_to_points = page_size.width() / gray.shape[1]
    
    print(f"Contour analysis (threshold: {config.min_area_pct * 100:.1f}% = {page_area * config.min_area_pct:.0f}pt²):")
    
    kept = 0
    filtered_small = 0
    filtered_fill = 0
    
    for i, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)
        rect = cv2.boundingRect(cnt)
        rect_area_px = rect[2] * rect[3]
        fill_ratio = area_px / rect_area_px if rect_area_px > 0 else 0
        
        # Convert to points
        area_pt = area_px * (scale_to_points ** 2)
        area_pct = area_pt / page_area
        
        w_pt = rect[2] * scale_to_points
        h_pt = rect[3] * scale_to_points
        
        # Check filters
        passes_area = area_pct >= config.min_area_pct
        passes_fill = fill_ratio >= config.min_fill_ratio
        passes = passes_area and passes_fill
        
        if passes:
            status = "✓ KEEP"
            kept += 1
        elif not passes_area:
            status = "✗ TOO SMALL"
            filtered_small += 1
        else:
            status = "✗ LOW FILL"
            filtered_fill += 1
        
        if i < 20 or passes:  # Show first 20 or all kept
            print(f"  [{i:2d}] {status:12s} area={area_pct*100:5.2f}% ({w_pt:5.0f}x{h_pt:5.0f}pt) fill={fill_ratio:.2f}")
    
    print(f"\nSummary:")
    print(f"  Total contours: {len(contours)}")
    print(f"  Kept: {kept}")
    print(f"  Filtered (too small): {filtered_small}")
    print(f"  Filtered (low fill): {filtered_fill}")
    
    # Save debug images
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / f"page{page_num+1}_1_gray.png"), gray)
    cv2.imwrite(str(output_dir / f"page{page_num+1}_2_smooth.png"), gray_smooth)
    cv2.imwrite(str(output_dir / f"page{page_num+1}_3_threshold.png"), th)
    cv2.imwrite(str(output_dir / f"page{page_num+1}_4_morph.png"), morph)
    
    # Draw rectangles
    debug_img = bgr.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area_px = cv2.contourArea(cnt)
        area_pt = area_px * (scale_to_points ** 2)
        area_pct = area_pt / page_area
        fill_ratio = area_px / (w * h) if w * h > 0 else 0
        
        if area_pct >= config.min_area_pct and fill_ratio >= config.min_fill_ratio:
            color = (0, 255, 0)  # Green for kept
        else:
            color = (0, 0, 255)  # Red for filtered
        
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
    
    cv2.imwrite(str(output_dir / f"page{page_num+1}_5_rectangles.png"), debug_img)
    
    print(f"\nDebug images saved to {output_dir}/")
    
    doc.close()


if __name__ == "__main__":
    samples_dir = Path(__file__).parent / "samples_PDF"
    pdf_files = list(samples_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("Error: No PDF files found in samples_PDF/")
        sys.exit(1)
    
    pdf_path = str(pdf_files[0])
    page_num = 3  # Page 4 (0-indexed)
    
    diagnose_page(pdf_path, page_num)
