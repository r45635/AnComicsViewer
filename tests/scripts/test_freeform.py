#!/usr/bin/env python3
"""Test freeform detection fallback on a specific page."""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PySide6.QtCore import QSizeF
from PySide6.QtPdf import QPdfDocument

from ancomicsviewer.config import DetectorConfig
from ancomicsviewer.detector import PanelDetector


def test_page(pdf_path: str, page_num: int):
    """Test panel detection on a specific page."""
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    # Load PDF
    doc = QPdfDocument()
    err = doc.load(pdf_path)
    
    if doc.pageCount() == 0:
        print(f"Error: Could not load PDF: {pdf_path}")
        return
    
    if page_num < 0 or page_num >= doc.pageCount():
        print(f"Error: Page {page_num} out of range (0-{doc.pageCount()-1})")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing page {page_num} of {os.path.basename(pdf_path)}")
    print(f"{'='*60}\n")
    
    # Create detector with debug enabled
    config = DetectorConfig(debug=True, use_freeform_fallback=True)
    detector = PanelDetector(config)
    
    # Render page
    pt = doc.pagePointSize(page_num)
    dpi = 150.0
    scale = dpi / 72.0
    qsize = QSizeF(pt.width() * scale, pt.height() * scale).toSize()
    qimg = doc.render(page_num, qsize)
    
    print(f"Page size: {pt.width():.1f} x {pt.height():.1f} points")
    print(f"Rendered at {dpi} DPI: {qimg.width()} x {qimg.height()} pixels\n")
    
    # Detect panels
    rects = detector.detect_panels(qimg, pt)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(rects)} panels detected")
    print(f"{'='*60}\n")
    
    for i, rect in enumerate(rects, 1):
        print(f"Panel {i}: ({rect.left():.1f}, {rect.top():.1f}) "
              f"{rect.width():.1f} x {rect.height():.1f} points")
    
    # Check for debug images
    debug_dir = "debug_output"
    if os.path.exists(debug_dir):
        debug_files = [f for f in os.listdir(debug_dir) if f.startswith('freeform_')]
        if debug_files:
            print(f"\nDebug images saved in {debug_dir}/:")
            for f in sorted(debug_files):
                print(f"  - {f}")
    
    doc.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_freeform.py <pdf_path> <page_number>")
        print("\nExample:")
        print("  python test_freeform.py samples_PDF/Gremillet.pdf 6")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2])
    
    test_page(pdf_path, page_num)
