#!/usr/bin/env python3
"""
Quick Panel Detection Test
Tests the YOLO detector integration directly.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '.')

try:
    from PySide6.QtGui import QImage
    from PySide6.QtCore import QSizeF
    from detectors.yolo_seg import YoloSegPanelDetector
    
    def test_yolo_detector():
        """Test the YOLO detector integration."""
        print("🧪 Testing YOLO Panel Detector Integration")
        print("=" * 45)
        
        # Check if model exists
        model_path = "runs/detect/overfit_small/weights/best.pt"
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        # Check if test image exists
        test_image = "dataset/images/train/p0003.png"
        if not Path(test_image).exists():
            print(f"❌ Test image not found: {test_image}")
            return False
        
        try:
            # Initialize detector
            print(f"📦 Loading detector with model: {model_path}")
            detector = YoloSegPanelDetector(weights=model_path, conf=0.1)
            
            # Load test image
            print(f"🖼️  Loading test image: {test_image}")
            qimage = QImage(test_image)
            if qimage.isNull():
                print("❌ Failed to load image")
                return False
            
            # Set page size (simulate PDF page)
            page_size = QSizeF(qimage.width(), qimage.height())
            
            # Detect panels
            print("🔍 Running panel detection...")
            panels = detector.detect_panels(qimage, page_size)
            
            print(f"✅ Detection completed!")
            print(f"📊 Results:")
            print(f"   Image size: {qimage.width()}x{qimage.height()}")
            print(f"   Panels found: {len(panels)}")
            
            if panels:
                print(f"📋 Panel coordinates:")
                for i, panel in enumerate(panels[:5]):  # Show first 5
                    print(f"   Panel {i+1}: ({panel.x():.0f}, {panel.y():.0f}) {panel.width():.0f}x{panel.height():.0f}")
                if len(panels) > 5:
                    print(f"   ... and {len(panels)-5} more panels")
            else:
                print("⚠️  No panels detected. Try:")
                print("   • Lower confidence threshold")
                print("   • Check image format")
                print("   • Verify model training")
            
            return len(panels) > 0
            
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    if __name__ == "__main__":
        success = test_yolo_detector()
        if success:
            print("\n🎉 Integration test passed!")
            print("💡 In AnComicsViewer:")
            print("   1. Open a comic PDF")
            print("   2. Go to Panels > YOLOv8 Seg (ML)")
            print("   3. Press Space to toggle panel view")
        else:
            print("\n❌ Integration test failed!")
            
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from AnComicsViewer directory")
