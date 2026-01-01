"""Image conversion utilities for AnComicsViewer.

Optimized QImage <-> NumPy conversions with proper memory handling.
"""

from __future__ import annotations

import sys
from typing import Optional

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

from PySide6.QtGui import QImage


def pdebug(*parts: object) -> None:
    """Thread-safe debug logger for panel detection."""
    try:
        sys.stdout.write("[Panels] " + " ".join(map(str, parts)) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def qimage_to_numpy_rgba(img: QImage) -> Optional[NDArray]:
    """Convert QImage to NumPy array (HxWx4 RGBA uint8).

    Handles PySide6 memoryview correctly and avoids data copies where possible.

    Args:
        img: QImage to convert

    Returns:
        NumPy array of shape (height, width, 4) or None on failure
    """
    if not HAS_NUMPY:
        pdebug("NumPy not available")
        return None

    if img.isNull():
        return None

    # Convert to RGBA8888 if needed
    if img.format() != QImage.Format.Format_RGBA8888:
        img = img.convertToFormat(QImage.Format.Format_RGBA8888)

    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()

    # Get raw bytes from memoryview
    mv = img.constBits()
    try:
        buf = bytes(mv)
        arr = np.frombuffer(buf, dtype=np.uint8)

        # Handle stride padding (bytesPerLine may be > width * 4)
        arr = arr.reshape((h, bpl))
        arr = arr[:, :w * 4]
        arr = arr.reshape((h, w, 4))

        # Return a contiguous copy for OpenCV compatibility
        return np.ascontiguousarray(arr)
    except Exception as e:
        pdebug(f"QImage conversion failed: {e}")
        return None


def rgba_to_grayscale(arr: NDArray) -> NDArray:
    """Convert RGBA array to grayscale using OpenCV.

    Args:
        arr: RGBA array of shape (H, W, 4)

    Returns:
        Grayscale array of shape (H, W)
    """
    if not HAS_CV2:
        # Fallback: simple luminance calculation
        return (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)


def rgba_to_lab_l(arr: NDArray, apply_clahe: bool = True) -> NDArray:
    """Convert RGBA to LAB and extract L channel with optional CLAHE.

    Args:
        arr: RGBA array of shape (H, W, 4)
        apply_clahe: Whether to apply CLAHE for contrast enhancement

    Returns:
        L channel as uint8 array of shape (H, W)
    """
    if not HAS_CV2:
        # Fallback to grayscale
        return rgba_to_grayscale(arr)

    rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L = clahe.apply(L)

    return L


def check_dependencies() -> tuple[bool, bool]:
    """Check availability of optional dependencies.

    Returns:
        Tuple of (has_numpy, has_cv2)
    """
    return HAS_NUMPY, HAS_CV2
