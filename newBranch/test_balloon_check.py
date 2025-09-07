#!/usr/bin/env python3
"""
Test script for balloon detection validation
"""

import sys
import os

# Mock the missing modules for testing
class MockModule:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

sys.modules['yaml'] = MockModule()
sys.modules['ultralytics'] = MockModule()
sys.modules['fitz'] = MockModule()
sys.modules['cv2'] = MockModule()

# Mock YOLO model with different scenarios
class MockYOLO:
    def __init__(self, has_balloon=True):
        self.names = {0: "panel", 1: "balloon"} if has_balloon else {0: "panel"}
        self.has_balloon = has_balloon

    def predict(self, *args, **kwargs):
        return [MockResult()]

class MockResult:
    def __init__(self):
        self.boxes = MockBoxes()

class MockBoxes:
    def __init__(self):
        self.xyxy = MockTensor([[10, 10, 100, 100]])
        self.cls = MockTensor([0])  # panel
        self.conf = MockTensor([0.9])

class MockTensor:
    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def numpy(self):
        return self.data

def test_balloon_detection_check():
    """Test the balloon detection validation logic"""

    print("=== Testing Balloon Detection Validation ===")

    # Test 1: Model with balloon class
    print("\n1. Testing model WITH balloon class")
    mock_model_with_balloon = MockYOLO(has_balloon=True)

    # Simulate the check from _load_model
    if hasattr(mock_model_with_balloon, 'names') and mock_model_with_balloon.names:
        if "balloon" not in mock_model_with_balloon.names.values():
            print("⚠️  WARNING: balloon class not found in model.names -> balloon detection disabled")
            balloon_detection_disabled = True
        else:
            balloon_detection_disabled = False
            balloon_class_index = None
            for idx, name in mock_model_with_balloon.names.items():
                if name == "balloon":
                    balloon_class_index = idx
                    break
    else:
        balloon_detection_disabled = False
        balloon_class_index = 1

    print(f"Balloon detection disabled: {balloon_detection_disabled}")
    print(f"Balloon class index: {balloon_class_index}")

    # Test 2: Model without balloon class
    print("\n2. Testing model WITHOUT balloon class")
    mock_model_without_balloon = MockYOLO(has_balloon=False)

    # Simulate the check from _load_model
    if hasattr(mock_model_without_balloon, 'names') and mock_model_without_balloon.names:
        if "balloon" not in mock_model_without_balloon.names.values():
            print("⚠️  WARNING: balloon class not found in model.names -> balloon detection disabled")
            balloon_detection_disabled = True
        else:
            balloon_detection_disabled = False
            balloon_class_index = None
            for idx, name in mock_model_without_balloon.names.items():
                if name == "balloon":
                    balloon_class_index = idx
                    break
    else:
        balloon_detection_disabled = False
        balloon_class_index = 1

    print(f"Balloon detection disabled: {balloon_detection_disabled}")
    print(f"Balloon class index: {balloon_class_index}")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_balloon_detection_check()
