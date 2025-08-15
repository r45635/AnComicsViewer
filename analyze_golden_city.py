#!/usr/bin/env python3
"""
Simple Golden City Comic Analysis (no GUI)
Analyzes the PDF file structure without creating GUI components.
"""

import os
import sys
from pathlib import Path

def analyze_pdf_file(pdf_path: str):
    """Analyze PDF file using basic file information."""
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    file_info = os.stat(pdf_path)
    file_size_mb = file_info.st_size / (1024 * 1024)
    
    print(f"=== Golden City - File Analysis ===")
    print(f"File: {Path(pdf_path).name}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Path: {pdf_path}")
    
    # Basic PDF header analysis
    try:
        with open(pdf_path, 'rb') as f:
            header = f.read(1024)
            if header.startswith(b'%PDF'):
                version_line = header.split(b'\n')[0].decode('ascii', errors='ignore')
                print(f"PDF Version: {version_line}")
            else:
                print("Warning: File doesn't appear to be a valid PDF")
    except Exception as e:
        print(f"Error reading file: {e}")

def create_test_script():
    """Create a script to test Golden City with the current detector."""
    
    script_content = '''#!/usr/bin/env python3
"""
Test Golden City comic with AnComicsViewer heuristic detector
"""

import sys
import os
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AnComicsViewer import PanelDetector
from PySide6.QtCore import QSizeF
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication
from PySide6.QtPdf import QPdfDocument

def test_golden_city_detection(pdf_path: str, page_num: int = 0):
    """Test panel detection on a specific page of Golden City."""
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Load PDF
    doc = QPdfDocument()
    if doc.load(pdf_path) != 0:
        print(f"Error: Could not load PDF: {pdf_path}")
        return
    
    if page_num >= doc.pageCount():
        print(f"Error: Page {page_num} not found (total pages: {doc.pageCount()})")
        return
    
    print(f"Testing Golden City page {page_num + 1}...")
    
    # Get page at detection DPI
    det_dpi = 200.0  # Good for Franco-Belge comics
    page_size_pt = doc.pagePointSize(page_num)
    scale = det_dpi / 72.0
    render_size = QSizeF(
        page_size_pt.width() * scale,
        page_size_pt.height() * scale
    ).toSize()
    
    # Render page
    image = doc.render(page_num, render_size)
    print(f"Page size: {image.width()}x{image.height()} pixels at {det_dpi} DPI")
    
    # Create detector with Franco-Belge preset
    detector = PanelDetector(debug=True)
    
    # Apply Franco-Belge preset settings
    detector.adaptive_block, detector.adaptive_C = 51, 5
    detector.morph_kernel, detector.morph_iter = 7, 2
    detector.min_rect_px = detector.min_panel_px = 60
    detector.light_col_rel, detector.light_row_rel = 0.12, 0.12
    detector.gutter_cov_min = 0.90
    detector.min_gutter_px, detector.max_gutter_px_frac = 8, 0.06
    detector.edge_margin_frac = 0.03
    detector.filter_title_rows = True
    detector.title_row_top_frac, detector.title_row_max_h_frac = 0.20, 0.12
    detector.title_row_min_boxes, detector.title_row_min_meanL = 4, 0.80
    detector.max_panels_per_page = 20
    detector.reading_rtl = False
    
    print("Applied Franco-Belge preset settings")
    
    # Run detection
    panels = detector.detect_panels(image, page_size_pt, det_dpi)
    
    print(f"\\nDetection Results:")
    print(f"Found {len(panels)} panels")
    
    for i, panel in enumerate(panels):
        print(f"Panel {i+1}: ({panel.x():.1f}, {panel.y():.1f}) {panel.width():.1f}x{panel.height():.1f} pts")
    
    return panels

if __name__ == "__main__":
    pdf_path = "Golden City - T01 - Pilleurs d'épaves.pdf"
    if len(sys.argv) > 1:
        page_num = int(sys.argv[1]) - 1  # Convert to 0-based
    else:
        page_num = 0
    
    test_golden_city_detection(pdf_path, page_num)
'''
    
    with open("test_golden_city_detection.py", "w") as f:
        f.write(script_content)
    
    print("Created test_golden_city_detection.py")

def main():
    pdf_path = "Golden City - T01 - Pilleurs d'épaves.pdf"
    
    analyze_pdf_file(pdf_path)
    print()
    create_test_script()
    
    print(f"\\n=== Next Steps ===")
    print(f"1. Load the PDF in AnComicsViewer (Ctrl+O)")
    print(f"2. Test panel detection: python test_golden_city_detection.py [page_number]")
    print(f"3. Use panel navigation (Ctrl+2 to toggle overlay, N/Shift+N for navigation)")
    print(f"4. Try different presets: Franco-Belge, Manga, Newspaper")

if __name__ == "__main__":
    main()
