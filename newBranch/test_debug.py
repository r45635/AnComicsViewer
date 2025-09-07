#!/usr/bin/env python3
"""
Test script for AnComicsViewer debug functionality
"""

import sys
import os
import time
from PySide6.QtCore import QRectF

# Mock the missing modules for testing
class MockModule:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

sys.modules['yaml'] = MockModule()
sys.modules['ultralytics'] = MockModule()
sys.modules['fitz'] = MockModule()
sys.modules['cv2'] = MockModule()

# Import our functions
try:
    import main
    print("✅ Successfully imported main module")
    print(f"DEBUG_DETECT from main: {main.DEBUG_DETECT}")
    print(f"DEBUG_OVERLAY_DIR from main: {main.DEBUG_OVERLAY_DIR}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Access functions and variables
debug_detection_stats = main.debug_detection_stats
save_debug_overlay = main.save_debug_overlay
save_detection_data = main.save_detection_data

def test_debug_functions():
    """Test the debug functions"""

    # Test data
    panels = [
        (0, 0.85, QRectF(100, 100, 200, 300)),
        (0, 0.92, QRectF(350, 150, 180, 250))
    ]
    balloons = [
        (1, 0.78, QRectF(120, 120, 50, 30)),
        (1, 0.65, QRectF(380, 180, 40, 25))
    ]
    page_area = 1000 * 800  # Mock page area

    print("=== Testing Debug Functions ===")
    print(f"Initial DEBUG_DETECT: {DEBUG_DETECT}")
    print(f"Initial DEBUG_OVERLAY_DIR: {DEBUG_OVERLAY_DIR}")

    # Test 1: Check function exists and can be called
    print("\n1. Testing function existence")
    print(f"debug_detection_stats function: {callable(debug_detection_stats)}")
    print(f"save_debug_overlay function: {callable(save_debug_overlay)}")
    print(f"save_detection_data function: {callable(save_detection_data)}")

    # Test 2: Debug stats with debug enabled
    print("\n2. Testing debug_detection_stats with DEBUG_DETECT = True")
    DEBUG_DETECT = True
    print(f"DEBUG_DETECT set to: {DEBUG_DETECT}")
    try:
        debug_detection_stats("TEST_ENABLED", panels, balloons, page_area)
        print("✅ debug_detection_stats called successfully")
    except Exception as e:
        print(f"❌ Error calling debug_detection_stats: {e}")

    # Test 3: Save detection data
    print("\n3. Testing save_detection_data")
    page_name = f"test_{int(time.time())}"
    try:
        save_detection_data(panels + balloons, page_name, "test_data")
        print("✅ save_detection_data called successfully")
    except Exception as e:
        print(f"❌ Error calling save_detection_data: {e}")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_debug_functions()
