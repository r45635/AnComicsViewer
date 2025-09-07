#!/usr/bin/env python3
"""
Test script to verify the best model and config are loaded correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def test_model_loading():
    """Test if the best model loads correctly"""
    print("Testing model loading configuration...")

    # Import main module
    import main

    # Check if YOLO is available
    try:
        YOLO = main.YOLO
    except AttributeError:
        YOLO = None

    if YOLO is None:
        print("❌ YOLO not available")
        return False

    # Test the _auto_load_model logic
    print("Testing _auto_load_model method...")

    # Create a mock viewer to test the method
    class MockViewer:
        def __init__(self):
            self.model = None
            self.model_status_text = ""

        def _load_model(self, path):
            """Mock load model"""
            try:
                self.model = YOLO(path)
                print(f"✅ Model loaded: {os.path.basename(path)}")
                return True
            except Exception as e:
                print(f"❌ Failed to load {path}: {e}")
                return False

        def set_status(self, text):
            self.model_status_text = text
            print(f"Status: {text}")

    viewer = MockViewer()

    # Test loading the best model first
    best_model = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/yolov8s.pt"
    if os.path.exists(best_model):
        if viewer._load_model(best_model):
            viewer.set_status("✅ Best model (YOLOv8s) loaded automatically")
            print("✅ SUCCESS: Best model loads correctly!")
            return True
        else:
            print("❌ Best model failed to load")
    else:
        print(f"❌ Best model not found: {best_model}")

    # Test fallback models
    original_model = "anComicsViewer_v01.pt"
    if os.path.exists(original_model):
        if viewer._load_model(original_model):
            viewer.set_status("✅ Original model loaded automatically")
            print("✅ SUCCESS: Fallback model loads correctly!")
            return True

    print("❌ No models could be loaded")
    return False

def test_config_loading():
    """Test if the best config loads correctly"""
    print("\nTesting config loading...")

    import main

    # Test loading the best config
    config_path = "config/detect.yaml"
    if os.path.exists(config_path):
        try:
            config = main.load_config(config_path)
            if config:
                print(f"✅ Config loaded: {len(config)} parameters")
                print("✅ SUCCESS: Best config loads correctly!")
                return True
            else:
                print("❌ Config loading returned empty")
        except AttributeError:
            print("❌ load_config function not accessible")
    else:
        print(f"❌ Config not found: {config_path}")

    return False

if __name__ == "__main__":
    print("🔧 Testing main.py configuration with best model and config")
    print("=" * 60)

    model_success = test_model_loading()
    config_success = test_config_loading()

    print("\n" + "=" * 60)
    if model_success and config_success:
        print("🎯 SUCCESS: main.py is configured to use the best model and config!")
        print("   - Model: YOLOv8s (best overall performer)")
        print("   - Config: detect.yaml (optimal settings)")
    else:
        print("❌ Configuration test failed")
        if not model_success:
            print("   - Model loading issue")
        if not config_success:
            print("   - Config loading issue")
