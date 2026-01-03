#!/usr/bin/env python3
"""Script to tune panel detection parameters for optimal results."""

import sys
from pathlib import Path
from PySide6.QtPdf import QPdfDocument
from PySide6.QtCore import QUrl

from ancomicsviewer.detector import PanelDetector
from ancomicsviewer.config import DetectorConfig


def test_detection(pdf_path: str, page_num: int, expected_panels: int, config: DetectorConfig) -> tuple[int, list]:
    """Test detection with given config and return panel count."""
    from PySide6.QtGui import QImage
    from PySide6.QtCore import QSize
    
    doc = QPdfDocument()
    doc.load(pdf_path)
    
    if page_num >= doc.pageCount():
        print(f"Error: Page {page_num} not found (document has {doc.pageCount()} pages)")
        return 0, []
    
    detector = PanelDetector(config)
    page_size = doc.pagePointSize(page_num)
    
    # Render page to image at detection DPI (150)
    dpi = 150.0
    scale = dpi / 72.0
    image_size = QSize(
        int(page_size.width() * scale),
        int(page_size.height() * scale)
    )
    image = doc.render(page_num, image_size)
    
    # Detect panels
    panels = detector.detect_panels(image, page_size)
    
    doc.close()
    return len(panels), panels


def grid_search(pdf_path: str, page_num: int, expected_panels: int):
    """Test multiple parameter combinations to find optimal settings."""
    print(f"=== Tuning parameters for page {page_num} (expected: {expected_panels} panels) ===\n")
    
    # Parameter ranges to test
    morph_kernels = [5, 6, 7, 8]
    morph_iters = [1, 2]
    min_area_pcts = [0.025, 0.03, 0.035, 0.04]
    min_fill_ratios = [0.35, 0.40, 0.45, 0.50]
    
    best_config = None
    best_score = float('inf')
    results = []
    
    total_tests = len(morph_kernels) * len(morph_iters) * len(min_area_pcts) * len(min_fill_ratios)
    test_count = 0
    
    for mk in morph_kernels:
        for mi in morph_iters:
            for map in min_area_pcts:
                for mfr in min_fill_ratios:
                    test_count += 1
                    
                    # Create config
                    config = DetectorConfig(
                        morph_kernel=mk,
                        morph_iter=mi,
                        min_area_pct=map,
                        min_fill_ratio=mfr,
                        debug=False
                    )
                    
                    # Test detection
                    panel_count, panels = test_detection(pdf_path, page_num, expected_panels, config)
                    
                    # Calculate score (absolute difference from expected)
                    score = abs(panel_count - expected_panels)
                    
                    result = {
                        'config': config,
                        'count': panel_count,
                        'score': score,
                        'mk': mk,
                        'mi': mi,
                        'map': map,
                        'mfr': mfr
                    }
                    results.append(result)
                    
                    # Update best
                    if score < best_score or (score == best_score and panel_count == expected_panels):
                        best_score = score
                        best_config = config
                    
                    # Progress
                    if test_count % 10 == 0 or score == 0:
                        status = "✓" if panel_count == expected_panels else "✗"
                        print(f"[{test_count}/{total_tests}] {status} mk={mk} mi={mi} area={map:.3f} fill={mfr:.2f} → {panel_count} panels (diff={score})")
    
    # Sort results by score
    results.sort(key=lambda r: (r['score'], abs(r['count'] - expected_panels)))
    
    print(f"\n=== TOP 5 CONFIGURATIONS ===\n")
    for i, r in enumerate(results[:5]):
        status = "✓✓✓" if r['count'] == expected_panels else f"({r['count']})"
        print(f"{i+1}. {status} morph_kernel={r['mk']}, morph_iter={r['mi']}, "
              f"min_area_pct={r['map']:.3f}, min_fill_ratio={r['mfr']:.2f} "
              f"→ {r['count']} panels (diff={r['score']})")
    
    if best_config:
        print(f"\n=== BEST CONFIGURATION ===")
        print(f"morph_kernel: {best_config.morph_kernel}")
        print(f"morph_iter: {best_config.morph_iter}")
        print(f"min_area_pct: {best_config.min_area_pct}")
        print(f"min_fill_ratio: {best_config.min_fill_ratio}")
        print(f"\nTo apply, update config.py with these values.")


if __name__ == "__main__":
    # Find sample PDF
    samples_dir = Path(__file__).parent / "samples_PDF"
    pdf_files = list(samples_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("Error: No PDF files found in samples_PDF/")
        sys.exit(1)
    
    pdf_path = str(pdf_files[0])
    page_num = 3  # Page 4 (0-indexed)
    expected_panels = 11
    
    print(f"Testing with: {pdf_path}")
    print(f"Target: Page {page_num + 1} (0-indexed: {page_num})")
    print(f"Expected panels: {expected_panels}\n")
    
    grid_search(pdf_path, page_num, expected_panels)
