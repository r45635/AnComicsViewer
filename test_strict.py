#!/usr/bin/env python3
"""Quick test with strict thresholds"""

import os
import sys
from pathlib import Path

# Set strict environment before importing
os.environ["ACV_DEBUG"] = "1"
os.environ["ACV_CONF_PANEL"] = "0.55"
os.environ["ACV_CONF_INSET"] = "0.65"
os.environ["ACV_CONF_BALLOON"] = "0.70"
os.environ["ACV_MAX_ASPECT"] = "3.0"
os.environ["ACV_MIN_PIXEL_SIZE"] = "40"

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import cv2
from ancomicsviewer.detectors.multibd_detector import MultiBDPanelDetector
from PySide6.QtCore import QSizeF
from PySide6.QtGui import QImage

def test_strict():
    print("🔒 Testing STRICT filtering parameters")
    print("=" * 50)
    
    detector = MultiBDPanelDetector()
    print(f"📊 Class thresholds from environment:")
    for class_name, threshold in detector.class_conf.items():
        print(f"   {class_name}: {threshold:.2f}")
    print()
    
    # Test one image
    img_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/detection_result_p0004.jpg"
    image = cv2.imread(img_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    page_size = QSizeF(w, h)
    
    print(f"🖼️  Testing: {os.path.basename(img_path)} ({w}x{h})")
    panels = detector.detect_panels(qimg, page_size)
    print(f"🎯 STRICT filtering result: {len(panels)} panels")
    
    for i, panel in enumerate(panels, 1):
        area_frac = (panel.width() * panel.height()) / (w * h)
        aspect = panel.width() / max(1e-6, panel.height())
        print(f"   Panel {i}: {panel.width():.0f}x{panel.height():.0f} (area: {area_frac:.3f}, aspect: {aspect:.2f})")

if __name__ == "__main__":
    test_strict()
