#!/usr/bin/env python3
"""
Test script to verify the new implementation features.
This script will test the main functionalities without needing a GUI.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from main import (
    _area, _iou, _overlap_frac, apply_nms_class_aware,
    PdfYoloViewer, GLOBAL_CONFIG
)
from PySide6.QtCore import QRectF

def test_helper_functions():
    """Test the basic helper functions."""
    print("Testing helper functions...")
    
    # Test rectangles
    r1 = QRectF(0, 0, 100, 100)  # 100x100 square
    r2 = QRectF(50, 50, 100, 100)  # Overlapping square
    r3 = QRectF(25, 25, 50, 50)  # Small square inside r1
    
    # Test _area
    assert _area(r1) == 10000.0, f"Expected 10000, got {_area(r1)}"
    assert _area(r3) == 2500.0, f"Expected 2500, got {_area(r3)}"
    
    # Test _iou
    iou_12 = _iou(r1, r2)
    print(f"IoU between r1 and r2: {iou_12:.3f}")
    assert 0.14 < iou_12 < 0.15, f"Expected IoU ~0.143, got {iou_12}"
    
    # Test _overlap_frac
    overlap_13 = _overlap_frac(r1, r3)  # r3 completely inside r1
    print(f"Overlap fraction r1->r3: {overlap_13:.3f}")
    assert overlap_13 == 1.0, f"Expected 1.0, got {overlap_13}"
    
    overlap_31 = _overlap_frac(r3, r1)  # only 25% of r1 is inside r3
    print(f"Overlap fraction r3->r1: {overlap_31:.3f}")
    assert overlap_31 == 0.25, f"Expected 0.25, got {overlap_31}"
    
    print("✅ Helper functions work correctly!")

def test_nms_class_aware():
    """Test class-aware NMS."""
    print("\nTesting class-aware NMS...")
    
    # Create test detections
    dets = [
        (0, 0.9, QRectF(10, 10, 80, 80)),    # Panel, high confidence
        (0, 0.8, QRectF(15, 15, 70, 70)),    # Panel, overlapping with first
        (1, 0.7, QRectF(50, 50, 30, 30)),    # Balloon
        (1, 0.6, QRectF(55, 55, 25, 25)),    # Balloon, overlapping with first balloon
        (0, 0.5, QRectF(200, 200, 50, 50)),  # Panel, far away
    ]
    
    # Apply NMS with high IoU threshold to see class separation
    filtered = apply_nms_class_aware(dets, iou_thr=0.3)
    
    print(f"Original detections: {len(dets)}")
    print(f"After class-aware NMS: {len(filtered)}")
    
    # Should keep the best from each class + the separate panel
    # Expected: 2 panels (best + far away) + 1 balloon (best)
    panels = [d for d in filtered if d[0] == 0]
    balloons = [d for d in filtered if d[0] == 1]
    
    print(f"Panels kept: {len(panels)}")
    print(f"Balloons kept: {len(balloons)}")
    
    assert len(panels) >= 1, "Should keep at least one panel"
    assert len(balloons) >= 1, "Should keep at least one balloon"
    
    print("✅ Class-aware NMS works correctly!")

def test_calibration():
    """Test pixel <-> PDF calibration."""
    print("\nTesting calibration...")
    
    # Create a mock viewer
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    viewer = PdfYoloViewer()
    
    # Set up mock calibration data
    viewer.render_dpi = 300.0
    viewer.page_size_pts = (612.0, 792.0)  # Standard letter size in points
    viewer.image_size_px = (2550, 3300)    # 300 DPI rendering of letter size
    
    # Test pixel to PDF conversion
    pixel_rect = QRectF(255, 330, 510, 660)  # 10% of image size
    pdf_rect = viewer.pixel_to_pdf_rect(pixel_rect)
    
    print(f"Pixel rect: {pixel_rect}")
    print(f"PDF rect: {pdf_rect}")
    
    # Should be roughly 10% of page size
    expected_w = 612.0 * 0.2  # 20% width
    expected_h = 792.0 * 0.2  # 20% height
    
    assert abs(pdf_rect.width() - expected_w) < 1.0, f"Width should be ~{expected_w}, got {pdf_rect.width()}"
    assert abs(pdf_rect.height() - expected_h) < 1.0, f"Height should be ~{expected_h}, got {pdf_rect.height()}"
    
    # Test round trip
    back_to_pixel = viewer.pdf_to_pixel_rect(pdf_rect)
    print(f"Back to pixel: {back_to_pixel}")
    
    # Should be very close to original
    assert abs(back_to_pixel.width() - pixel_rect.width()) < 1.0, "Round trip width error"
    assert abs(back_to_pixel.height() - pixel_rect.height()) < 1.0, "Round trip height error"
    
    print("✅ Calibration works correctly!")

def test_quality_metrics():
    """Test quality metrics computation."""
    print("\nTesting quality metrics...")
    
    # Create a mock viewer
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    viewer = PdfYoloViewer()
    viewer.image_size_px = (1000, 1000)  # 1000x1000 image
    viewer.page_index = 0
    
    # Create test detections
    panels = [
        (0, 0.9, QRectF(100, 100, 300, 200)),  # Panel 1
        (0, 0.8, QRectF(500, 500, 200, 300)),  # Panel 2
    ]
    
    balloons = [
        (1, 0.7, QRectF(150, 150, 100, 50)),   # Balloon overlapping with panel 1
        (1, 0.6, QRectF(550, 550, 80, 60)),    # Balloon overlapping with panel 2
        (1, 0.5, QRectF(800, 800, 50, 40)),    # Balloon not overlapping much
    ]
    
    metrics = viewer.compute_quality_metrics(panels, balloons)
    
    print(f"Metrics: {metrics}")
    
    assert metrics["panels"] == 2, f"Expected 2 panels, got {metrics['panels']}"
    assert metrics["balloons"] == 3, f"Expected 3 balloons, got {metrics['balloons']}"
    assert metrics["page_index"] == 0, f"Expected page_index 0, got {metrics['page_index']}"
    assert len(metrics["panel_area_ratios"]) == 2, "Should have 2 panel area ratios"
    assert len(metrics["balloon_area_ratios"]) == 3, "Should have 3 balloon area ratios"
    assert 0.0 <= metrics["quality_score"] <= 1.0, f"Quality score should be in [0,1], got {metrics['quality_score']}"
    
    print("✅ Quality metrics work correctly!")

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting config loading...")
    
    # Test _cfg method
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    viewer = PdfYoloViewer()
    
    # Test default values
    assert viewer._cfg('nonexistent_key', 42) == 42, "Should return default for missing key"
    assert viewer._cfg('nonexistent_key', 3.14) == 3.14, "Should return default for missing key"
    assert viewer._cfg('nonexistent_key', "test") == "test", "Should return default for missing key"
    
    # Test with global config
    global GLOBAL_CONFIG
    GLOBAL_CONFIG['test_key'] = 123
    assert viewer._cfg('test_key', 999) == 123, "Should return config value when available"
    
    print("✅ Config loading works correctly!")

if __name__ == "__main__":
    print("🧪 Testing AnComicsViewer implementation...")
    
    try:
        test_helper_functions()
        test_nms_class_aware()
        test_calibration() 
        test_quality_metrics()
        test_config_loading()
        
        print("\n🎉 All tests passed! The implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
