#!/usr/bin/env python3
"""Test script to validate ML detector functionality."""

import sys
import os

# Set matplotlib backend before any other imports
import matplotlib
matplotlib.use('Agg')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ml_detector():
    print("Testing ML detector loading...")
    
    try:
        from detectors.yolo_seg import YoloSegPanelDetector
        print("✓ YoloSegPanelDetector import successful")
        
        # Test with the downloaded model
        model_path = "yolov8n-seg.pt"
        if os.path.exists(model_path):
            try:
                detector = YoloSegPanelDetector(weights=model_path, rtl=False)
                print(f"✓ Successfully loaded YOLOv8 model from {model_path}")
                print(f"  - Confidence threshold: {detector.conf}")
                print(f"  - IoU threshold: {detector.iou}")
                print(f"  - Reading RTL: {detector.reading_rtl}")
                return True
            except Exception as e:
                print(f"✗ Failed to load YOLOv8 model: {e}")
                return False
        else:
            print(f"✗ Model file {model_path} not found")
            return False
            
    except Exception as e:
        print(f"✗ Failed to import YoloSegPanelDetector: {e}")
        return False

if __name__ == "__main__":
    success = test_ml_detector()
    sys.exit(0 if success else 1)
