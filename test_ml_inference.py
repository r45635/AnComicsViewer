#!/usr/bin/env python3
"""Test script to validate ML detector inference."""

import sys
import os
import numpy as np

# Set matplotlib backend before any other imports
import matplotlib
matplotlib.use('Agg')

from PySide6.QtCore import QSizeF
from PySide6.QtGui import QImage

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ml_detection():
    print("Testing ML detector inference...")
    
    try:
        from detectors.yolo_seg import YoloSegPanelDetector
        
        # Create detector
        detector = YoloSegPanelDetector(weights="yolov8n-seg.pt", rtl=False)
        print("✓ ML detector created successfully")
        
        # Create a dummy image for testing
        width, height = 800, 600
        dummy_image = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
        
        # Convert to QImage
        qimage = QImage(dummy_image.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
        page_size = QSizeF(width, height)
        
        print("✓ Created test image (800x600)")
        
        # Run detection
        panels = detector.detect_panels(qimage, page_size)
        print(f"✓ Detection completed successfully")
        print(f"  - Found {len(panels)} panels")
        
        if panels:
            for i, panel in enumerate(panels):
                print(f"  - Panel {i+1}: ({panel.x():.1f}, {panel.y():.1f}) {panel.width():.1f}x{panel.height():.1f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ml_detection()
    sys.exit(0 if success else 1)
