#!/usr/bin/env python3
"""Regression test for panel detection modes.

Tests panel detection on reference pages to ensure:
- AUTO mode correctly classifies pages
- CLASSIC mode preserves Tintin performance
- MODERN mode improves modern page detection

Usage:
    python tests/scripts/regress_panels.py <pdf_path> <classic_page> <modern_page>
    
Example:
    python tests/scripts/regress_panels.py "samples_PDF/Tintin.pdf" 4 18
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PySide6.QtCore import QSizeF
from PySide6.QtGui import QImage
from PySide6.QtPdf import QPdfDocument

from ancomicsviewer.detector import PanelDetector
from ancomicsviewer.config import DetectorConfig


def test_page(pdf_path: str, page_num: int, panel_mode: str, label: str) -> dict:
    """Test panel detection on a single page.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        panel_mode: Detection mode
        label: Label for output
        
    Returns:
        Dictionary with metrics
    """
    # Load PDF
    doc = QPdfDocument()
    err = doc.load(pdf_path)
    if doc.status() != QPdfDocument.Status.Ready:
        print(f"Error loading PDF: {err}")
        return {}
    
    if page_num >= doc.pageCount():
        print(f"Page {page_num} out of range (0-{doc.pageCount()-1})")
        return {}
    
    # Get page
    pt = doc.pagePointSize(page_num)
    page_point_size = QSizeF(pt.width(), pt.height())
    
    # Render at detection DPI
    dpi = 150.0
    scale = dpi / 72.0
    qsize = QSizeF(pt.width() * scale, pt.height() * scale).toSize()
    qimg = doc.render(page_num, qsize)
    
    # Create detector with specified mode
    config = DetectorConfig(panel_mode=panel_mode, debug=True)
    detector = PanelDetector(config)
    
    # Detect panels
    rects = detector.detect_panels(qimg, page_point_size)
    
    # Extract metrics from decision context if available
    decision_context = getattr(detector, '_decision_context', {})
    
    # Compute coverage
    w, h = qimg.width(), qimg.height()
    coverage = detector._rects_union_coverage(rects, w, h) if rects else 0.0
    small_ratio = detector._small_rect_ratio(rects, w, h) if rects else 0.0
    margin_ratio = detector._margin_rect_ratio(rects, w, h) if rects else 0.0
    
    # Build result
    result = {
        "label": label,
        "page": page_num,
        "panel_mode": panel_mode,
        "mode_used": decision_context.get("panel_mode_used", panel_mode),
        "route_chosen": decision_context.get("route_chosen", "unknown"),
        "panel_count": len(rects),
        "coverage": round(coverage, 3),
        "small_ratio": round(small_ratio, 3),
        "margin_ratio": round(margin_ratio, 3),
    }
    
    # Add candidate info if available
    if "adaptive_count" in decision_context:
        result["adaptive_count"] = decision_context["adaptive_count"]
        result["adaptive_coverage"] = round(decision_context.get("adaptive_coverage", 0.0), 3)
    
    if "freeform_count" in decision_context:
        result["freeform_count"] = decision_context["freeform_count"]
        result["freeform_coverage"] = round(decision_context.get("freeform_coverage", 0.0), 3)
        result["freeform_selected"] = decision_context.get("freeform_selected", False)
    
    doc.close()
    
    return result


def main():
    if len(sys.argv) < 4:
        print("Usage: regress_panels.py <pdf_path> <classic_page> <modern_page>")
        print("Example: regress_panels.py 'Tintin.pdf' 4 18")
        return 1
    
    pdf_path = sys.argv[1]
    classic_page = int(sys.argv[2])
    modern_page = int(sys.argv[3])
    
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return 1
    
    print(f"Testing: {pdf_path}")
    print(f"Classic page: {classic_page}, Modern page: {modern_page}")
    print("=" * 70)
    
    # Test configurations
    tests = [
        # Classic page tests
        ("classic-auto", classic_page, "auto", "Classic page with AUTO"),
        ("classic-classic", classic_page, "classic_franco_belge", "Classic page with CLASSIC mode"),
        ("classic-modern", classic_page, "modern", "Classic page with MODERN mode"),
        
        # Modern page tests
        ("modern-auto", modern_page, "auto", "Modern page with AUTO"),
        ("modern-classic", modern_page, "classic_franco_belge", "Modern page with CLASSIC mode"),
        ("modern-modern", modern_page, "modern", "Modern page with MODERN mode"),
    ]
    
    results = {}
    
    for test_id, page, mode, label in tests:
        print(f"\n{label}:")
        print("-" * 70)
        result = test_page(pdf_path, page, mode, label)
        results[test_id] = result
        
        # Print results
        print(f"  Mode Input:      {result.get('panel_mode')}")
        print(f"  Mode Used:       {result.get('mode_used')}")
        print(f"  Route Chosen:    {result.get('route_chosen')}")
        print(f"  Panel Count:     {result.get('panel_count')}")
        print(f"  Coverage:        {result.get('coverage')}")
        print(f"  Small Ratio:     {result.get('small_ratio')}")
        print(f"  Margin Ratio:    {result.get('margin_ratio')}")
        
        if "freeform_count" in result:
            print(f"  Freeform Count:  {result.get('freeform_count')}")
            print(f"  Freeform Sel:    {result.get('freeform_selected')}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("-" * 70)
    
    # Check classic page with auto
    classic_auto = results.get("classic-auto", {})
    print(f"\nClassic page AUTO classification: {classic_auto.get('mode_used')}")
    if classic_auto.get('mode_used') == 'classic_franco_belge':
        print("  ✓ Correctly classified as classic")
    else:
        print("  ✗ WARNING: Classified as modern (potential regression)")
    
    # Check modern page with auto
    modern_auto = results.get("modern-auto", {})
    print(f"\nModern page AUTO classification: {modern_auto.get('mode_used')}")
    if modern_auto.get('mode_used') == 'modern':
        print("  ✓ Correctly classified as modern")
    else:
        print("  ℹ Classified as classic (conservative, may be acceptable)")
    
    # Save results
    output_file = "debug_output/regression_results.json"
    os.makedirs("debug_output", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
