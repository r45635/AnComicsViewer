"""Shared utilities and data structures for panel detection.

Contains:
- PanelRegion dataclass for freeform detection
- DebugInfo dataclass for visualization
- Common helper functions
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

from PySide6.QtCore import QRectF

# Optional dependencies
try:
    import numpy as np
    from numpy.typing import NDArray
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    NDArray = None  # type: ignore
    HAS_NUMPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    HAS_CV2 = False


def pdebug(*parts: object) -> None:
    """Thread-safe debug logger for panel detection."""
    try:
        msg = "[Panels] " + " ".join(map(str, parts)) + "\n"
        sys.stdout.write(msg)
        sys.stdout.flush()
    except Exception:
        pass


@dataclass
class PanelRegion:
    """Represents a detected panel region with multiple representations.
    
    Used in freeform detection for complex layouts (parallelograms, tinted backgrounds, etc.)
    """
    contour: NDArray          # Original contour (Nx1x2)
    poly: NDArray             # Simplified polygon approximation
    bbox: Tuple[int, int, int, int]  # Axis-aligned bounding box (x, y, w, h)
    obb: NDArray              # Oriented bounding box (4 points from minAreaRect)
    area: float               # Contour area
    fill_ratio: float         # area / (bbox_w * bbox_h)
    touches_border: bool      # Whether region touches image border
    centroid: Tuple[float, float]  # (cx, cy)
    
    def to_qrectf(self, scale: float) -> QRectF:
        """Convert bbox to QRectF in page points."""
        x, y, w, h = self.bbox
        return QRectF(x / scale, y / scale, w / scale, h / scale)


@dataclass
class DebugInfo:
    """Debug information from detection pass."""
    vertical_splits: List[Tuple[float, float, float, float]]  # (x, y, w, h) in page points
    horizontal_splits: List[Tuple[float, float, float, float]]

    @classmethod
    def empty(cls) -> "DebugInfo":
        return cls(vertical_splits=[], horizontal_splits=[])


def estimate_bg_lab(img_bgr: NDArray, border_pct: float = 0.04) -> NDArray:
    """Estimate background color from image borders in Lab color space.
    
    Args:
        img_bgr: Input image in BGR format
        border_pct: Percentage of image dimensions to use for border sampling
        
    Returns:
        Array of shape (3,) containing (L, a, b) median values
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return np.array([255.0, 128.0, 128.0])  # Default white
    
    h, w = img_bgr.shape[:2]
    border_size = max(int(min(h, w) * border_pct), 5)
    
    # Sample borders: top, bottom, left, right
    samples = []
    samples.append(img_bgr[0:border_size, :])  # Top
    samples.append(img_bgr[h-border_size:h, :])  # Bottom
    samples.append(img_bgr[:, 0:border_size])  # Left
    samples.append(img_bgr[:, w-border_size:w])  # Right
    
    # Concatenate all border pixels
    border_pixels = np.vstack([s.reshape(-1, 3) for s in samples])
    
    # Convert to Lab
    border_lab = cv2.cvtColor(border_pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2Lab)
    border_lab = border_lab.reshape(-1, 3).astype(np.float32)
    
    # Compute median (robust to outliers)
    bg_lab = np.median(border_lab, axis=0)
    
    return bg_lab


def non_bg_ratio(img_bgr_roi: NDArray, bg_lab: NDArray, delta: float = 12.0) -> float:
    """Calculate ratio of non-background pixels in a ROI.
    
    Args:
        img_bgr_roi: ROI in BGR format
        bg_lab: Background color in Lab (shape (3,))
        delta: Maximum Lab distance to be considered background
        
    Returns:
        Ratio of pixels that are NOT background (0.0 to 1.0)
    """
    if not HAS_CV2 or not HAS_NUMPY or img_bgr_roi.size == 0:
        return 0.0
    
    # Convert ROI to Lab
    roi_lab = cv2.cvtColor(img_bgr_roi, cv2.COLOR_BGR2Lab).astype(np.float32)
    
    # Compute Euclidean distance to background color
    dist = np.linalg.norm(roi_lab - bg_lab, axis=2)
    
    # Non-background pixels are those far from bg color
    non_bg = dist > delta
    
    return float(np.mean(non_bg))


def iou(a: QRectF, b: QRectF) -> float:
    """Calculate Intersection over Union."""
    inter = a.intersected(b)
    if inter.isEmpty():
        return 0.0
    ia = inter.width() * inter.height()
    ua = a.width() * a.height() + b.width() * b.height() - ia
    return ia / ua if ua > 0 else 0.0


def union(a: QRectF, b: QRectF) -> QRectF:
    """Calculate bounding box union."""
    return QRectF(
        min(a.left(), b.left()),
        min(a.top(), b.top()),
        max(a.right(), b.right()) - min(a.left(), b.left()),
        max(a.bottom(), b.bottom()) - min(a.top(), b.top()),
    )


def merge_rects(rects: List[QRectF], iou_thresh: float = 0.25) -> List[QRectF]:
    """Merge overlapping rectangles using IoU threshold."""
    if not rects:
        return []

    merged: List[QRectF] = []
    for r in rects:
        did_merge = False
        for i, m in enumerate(merged):
            if iou(r, m) >= iou_thresh:
                merged[i] = union(r, m)
                did_merge = True
                break
        if not did_merge:
            merged.append(r)
    return merged
