"""Adaptive threshold detection route.

Primary detection method using:
- Bilateral filter for edge-preserving smoothing
- Adaptive thresholding
- Morphological operations
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QRectF, QSizeF

from .utils import pdebug, merge_rects, HAS_CV2, HAS_NUMPY

if HAS_CV2:
    import cv2
if HAS_NUMPY:
    import numpy as np
    from numpy.typing import NDArray


def adaptive_threshold_route(
    gray: NDArray,
    config: "DetectorConfig",
) -> NDArray:
    """Primary detection route using adaptive threshold.

    Uses bilateral filter + adaptive threshold + morphological closing.
    
    Args:
        gray: Grayscale image
        config: Detector configuration
        
    Returns:
        Binary mask with potential panel regions
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return np.zeros_like(gray) if HAS_NUMPY else None
    
    h, w = gray.shape[:2]
    
    # Adapt block size to image resolution
    relative_block = 99 * (w / 1200.0)
    block_size = int(relative_block) | 1  # Make odd
    block_size = max(51, min(201, block_size))
    
    # Use bilateral filter to preserve edges while smoothing text
    gray_smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    th = cv2.adaptiveThreshold(
        gray_smooth, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, 5
    )

    # Morphological closing to connect borders
    kernel_size = int(config.morph_kernel * (w / 1200.0)) | 1
    kernel_size = max(3, min(15, kernel_size))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=config.morph_iter)


def lab_route(L: NDArray) -> NDArray:
    """LAB L-channel fallback route.
    
    Args:
        L: L channel from LAB color space
        
    Returns:
        Binary mask
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return np.zeros_like(L) if HAS_NUMPY else None
    
    th = cv2.adaptiveThreshold(
        L, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 5
    )
    kernel = np.ones((7, 7), np.uint8)
    return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)


def canny_route(gray: NDArray) -> NDArray:
    """Canny edge-based fallback route.
    
    Args:
        gray: Grayscale image
        
    Returns:
        Binary mask
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return np.zeros_like(gray) if HAS_NUMPY else None
    
    edges = cv2.Canny(gray, 60, 180)
    kernel = np.ones((5, 5), np.uint8)
    dil = cv2.dilate(edges, kernel, iterations=2)
    return cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kernel, iterations=2)


def rects_from_mask(
    mask: NDArray, 
    w: int, 
    h: int, 
    page_point_size: QSizeF,
    config: "DetectorConfig",
) -> List[QRectF]:
    """Extract and filter rectangles from binary mask.
    
    Uses relaxed thresholds for initial extraction to catch fragments,
    proper filtering happens later after merging.
    
    Args:
        mask: Binary mask
        w, h: Image dimensions
        page_point_size: Page size in points
        config: Detector configuration
        
    Returns:
        List of rectangles in page point coordinates
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return []
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    page_area_px = float(w * h)
    scale = w / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0

    # Relaxed threshold for initial extraction
    min_area_initial = config.min_area_pct / 2.0
    min_px_dyn = max(config.min_rect_px // 2, int(config.min_rect_frac * min(w, h) / 2))

    rects: List[QRectF] = []
    for contour in contours:
        # Rotated bbox then clamp to image bounds
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        xs, ys = box[:, 0], box[:, 1]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)

        cw = max(1, x1 - x0)
        ch = max(1, y1 - y0)

        # Area filter
        area_px = cw * ch
        area_pct = area_px / page_area_px
        if area_pct < min_area_initial or area_pct > config.max_area_pct:
            continue

        # Fill ratio filter
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= 0:
            continue
        fill = contour_area / float(area_px)
        if fill < (config.min_fill_ratio * 0.8):
            continue

        # Size filter
        if cw < min_px_dyn or ch < min_px_dyn:
            continue

        # Convert to page points
        px = x0 / scale
        py = y0 / scale
        pw = cw / scale
        ph = ch / scale
        rects.append(QRectF(px, py, pw, ph))

    return merge_rects(rects, iou_thresh=0.25)


# Import for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config import DetectorConfig
