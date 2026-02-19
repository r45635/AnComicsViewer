"""Test cover/splash page detection returns a single full-page panel."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import cv2
from PySide6.QtCore import QSizeF
from PySide6.QtGui import QImage
from ancomicsviewer import PanelDetector
from ancomicsviewer.config import DetectorConfig


def make_cover_image(w=720, h=1007):
    """Continuous illustration without panel borders (cover page).
    
    Uses blurred shapes so no clean countours are detected by hierarchy,
    simulating real cover art with gradients and painted elements.
    """
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :] = [120, 80, 60, 255]          # warm background
    # Soft organic shapes (blurred heavily = no hard contours)
    overlay = img[:, :, :3].copy()
    cv2.circle(overlay, (360, 550), 300, (200, 150, 100), -1)   # body
    cv2.circle(overlay, (360, 250), 120, (220, 170, 130), -1)   # head
    # Heavy blur to destroy any clean edges
    overlay = cv2.GaussianBlur(overlay, (61, 61), 0)
    img[:, :, :3] = overlay
    img[:, :, 3] = 255
    return QImage(img.data, w, h, QImage.Format.Format_RGBA8888)


def make_multi_panel_image(w=720, h=1007):
    """White background with 3 clear panels separated by black gutters."""
    img = np.ones((h, w, 4), dtype=np.uint8) * 255
    img[:, :, 3] = 255
    # Black gutters
    img[330:340, :] = [0, 0, 0, 255]   # horizontal gutter
    img[:, 355:365] = [0, 0, 0, 255]   # vertical gutter
    # Content blobs
    cv2.rectangle(img, (20, 20), (340, 320), (180, 120, 80, 255), -1)
    cv2.rectangle(img, (380, 20), (700, 320), (80, 120, 180, 255), -1)
    cv2.rectangle(img, (20, 360), (700, 980), (80, 180, 80, 255), -1)
    return QImage(img.data, w, h, QImage.Format.Format_RGBA8888)


def run():
    config = DetectorConfig(debug=False)
    det = PanelDetector(config)
    page_size = QSizeF(345.6, 483.5)

    # Test 1: cover page -> must return exactly 1 full-page panel
    qimg = make_cover_image()
    rects = det.detect_panels(qimg, page_point_size=page_size)
    print(f"Cover page: {len(rects)} panel(s)")
    for i, r in enumerate(rects):
        print(f"  Panel {i+1}: ({r.x():.0f},{r.y():.0f}) {r.width():.0f}x{r.height():.0f}")
    assert len(rects) == 1, f"Cover: expected 1 panel, got {len(rects)}"
    # Full page dimensions
    assert rects[0].width() >= page_size.width() * 0.95
    assert rects[0].height() >= page_size.height() * 0.95
    print("  OK: cover correctly detected as 1 full-page panel")

    # Test 2: multi-panel page -> must return >1 panel
    qimg2 = make_multi_panel_image()
    rects2 = det.detect_panels(qimg2, page_point_size=page_size)
    print(f"\nMulti-panel page: {len(rects2)} panel(s)")
    assert len(rects2) >= 2, f"Multi-panel: expected >=2 panels, got {len(rects2)}"
    print("  OK: multi-panel page correctly detected")

    print("\nALL COVER PAGE TESTS PASSED")


if __name__ == "__main__":
    run()
