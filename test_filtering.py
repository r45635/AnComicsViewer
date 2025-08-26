#!/usr/bin/env python3
"""Test script for enhanced detection filtering."""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ancomicsviewer.detectors.multibd_detector import MultiBDPanelDetector
from PySide6.QtCore import QSizeF

def test_enhanced_filtering():
    """Test the new class-specific filtering capabilities."""
    
    # Set up environment for enhanced filtering
    os.environ["ACV_DEBUG"] = "1"
    os.environ["ACV_CONF_PANEL"] = "0.40"
    os.environ["ACV_CONF_INSET"] = "0.50"
    os.environ["ACV_CONF_BALLOON"] = "0.55"
    os.environ["ACV_MAX_ASPECT"] = "4.0"
    os.environ["ACV_MIN_PIXEL_SIZE"] = "25"
    
    print("🧪 Testing Enhanced Detection Filtering")
    print("=" * 50)
    
    # Initialize detector
    detector = MultiBDPanelDetector()
    print(f"✅ Detector loaded: {detector.get_model_info()['name']}")
    print(f"📊 Class thresholds:")
    for class_name, threshold in detector.class_conf.items():
        print(f"   {class_name}: {threshold:.2f}")
    print(f"🔧 Geometric filters: max_aspect={os.environ['ACV_MAX_ASPECT']}, min_pixel={os.environ['ACV_MIN_PIXEL_SIZE']}")
    print()
    
    # Test images
    test_images = [
        "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/detection_result_p0002.jpg",
        "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/detection_result_p0004.jpg"
    ]
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"⚠️  Skipping {os.path.basename(img_path)} (not found)")
            continue
            
        print(f"🖼️  Testing: {os.path.basename(img_path)}")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Failed to load {img_path}")
            continue
            
        # Convert to RGB then QImage for detection
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        
        # Convert to QImage
        from PySide6.QtGui import QImage
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        page_size = QSizeF(w, h)
        
        print(f"   📐 Image size: {w}x{h}")
        
        # Run detection
        panels = detector.detect_panels(qimg, page_size)
        
        print(f"   🎯 Detected {len(panels)} panels")
        
        # Show panel info
        for i, panel in enumerate(panels, 1):
            area_frac = (panel.width() * panel.height()) / (w * h)
            aspect = panel.width() / max(1e-6, panel.height())
            print(f"      Panel {i}: {panel.width():.0f}x{panel.height():.0f} (area: {area_frac:.3f}, aspect: {aspect:.2f})")
        
        print()

if __name__ == "__main__":
    test_enhanced_filtering()
