"""Gutter-based panel detection.

Detects panels by finding white/light separations (gutters) between panels.
Better for layouts where panels are separated by color transitions, not black lines.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

from PySide6.QtCore import QRectF, QSizeF

from .utils import pdebug, merge_rects, HAS_CV2, HAS_NUMPY

if HAS_CV2:
    import cv2
if HAS_NUMPY:
    import numpy as np
    from numpy.typing import NDArray


def make_gutter_mask(
    gray: NDArray, 
    L: NDArray, 
    config: "DetectorConfig",
    img_bgr: NDArray = None,
) -> NDArray:
    """Build a robust gutter mask using brightness + gradient uniformity.
    
    Identifies gutters as pixels that are BOTH very bright AND locally uniform
    (low gradient). This avoids picking up bright regions inside panels.
    
    Args:
        gray: Grayscale image
        L: LAB L-channel
        config: Detector configuration
        img_bgr: Optional color image
        
    Returns:
        Binary gutter mask (0=not gutter, 255=gutter)
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return np.zeros_like(L) if HAS_NUMPY else None
    
    h, w = L.shape
    
    # Adaptive brightness threshold
    bright_percentile = getattr(config, 'gutter_bright_percentile', 94)
    bright_thresh = np.percentile(L, bright_percentile)
    
    # Compute gradient magnitude (Sobel)
    grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.abs(grad_x) + np.abs(grad_y)
    
    # Adaptive gradient threshold
    grad_percentile = config.gutter_grad_percentile
    grad_low = np.percentile(grad_mag, grad_percentile)
    
    if config.debug:
        pdebug(f"[gutter_mask] bright_thresh (p{bright_percentile})={bright_thresh:.1f}")
        pdebug(f"[gutter_mask] grad_low (p{grad_percentile})={grad_low:.1f}")
    
    # Combine: both bright AND uniform
    bright_mask = (L >= bright_thresh).astype(np.uint8) * 255
    uniform_mask = (grad_mag <= grad_low).astype(np.uint8) * 255
    gutter_mask = cv2.bitwise_and(bright_mask, uniform_mask)
    
    # Apply morphological opening with elongated kernels
    open_kernel_len = max(int(w * config.gutter_open_kernel_frac), 21)
    
    h_kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel_len, 3))
    h_open = cv2.morphologyEx(gutter_mask, cv2.MORPH_OPEN, h_kernel_open, iterations=1)
    
    v_kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, open_kernel_len))
    v_open = cv2.morphologyEx(gutter_mask, cv2.MORPH_OPEN, v_kernel_open, iterations=1)
    
    gutter_mask = cv2.bitwise_or(h_open, v_open)
    
    # Small closing to reconnect small gaps
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    gutter_mask = cv2.morphologyEx(gutter_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    
    # Filter connected components to keep only stripe-like structures
    gutter_mask = _filter_stripes(gutter_mask, w, h, config)
    
    white_ratio = cv2.countNonZero(gutter_mask) / gutter_mask.size
    pdebug(f"[gutter_mask] white_ratio={white_ratio:.4f}")
    
    return gutter_mask


def _filter_stripes(
    mask: NDArray, 
    w: int, 
    h: int, 
    config: "DetectorConfig"
) -> NDArray:
    """Filter connected components to keep only stripe-like structures."""
    min_stripe_w = config.gutter_min_stripe_width
    min_length_frac = config.gutter_stripe_length_frac
    
    min_h_stripe_len = int(w * min_length_frac)
    min_v_stripe_len = int(h * min_length_frac)
    
    num_labels, labels = cv2.connectedComponents(mask)
    result = np.zeros_like(mask)
    
    for label_id in range(1, num_labels):
        component_mask = (labels == label_id).astype(np.uint8) * 255
        x, y, cw, ch = cv2.boundingRect(component_mask)
        
        # Keep horizontal stripes
        if cw >= min_h_stripe_len and ch <= min_stripe_w:
            result[component_mask > 0] = 255
        # Keep vertical stripes
        elif ch >= min_v_stripe_len and cw <= min_stripe_w:
            result[component_mask > 0] = 255
    
    return result


def validate_gutter_lines(
    gutter_mask: NDArray,
    h_lines: List[Tuple[int, int]],
    v_lines: List[Tuple[int, int]],
    config: "DetectorConfig",
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Validate gutter candidates based on coverage and thickness.
    
    Args:
        gutter_mask: Binary gutter mask
        h_lines: Horizontal gutter candidates (y_start, y_end)
        v_lines: Vertical gutter candidates (x_start, x_end)
        config: Detector configuration
        
    Returns:
        Tuple of (validated_h_lines, validated_v_lines)
    """
    min_coverage = config.gutter_cov_min
    min_thickness = config.min_gutter_px
    
    h, w = gutter_mask.shape
    validated_h = []
    validated_v = []
    
    for y_start, y_end in h_lines:
        thickness = y_end - y_start + 1
        if thickness < min_thickness:
            continue
        
        band = gutter_mask[y_start:y_end+1, :]
        if band.size == 0:
            continue
        
        coverage = np.mean(band == 255)
        if coverage >= min_coverage:
            validated_h.append((y_start, y_end))
    
    for x_start, x_end in v_lines:
        thickness = x_end - x_start + 1
        if thickness < min_thickness:
            continue
        
        band = gutter_mask[:, x_start:x_end+1]
        if band.size == 0:
            continue
        
        coverage = np.mean(band == 255)
        if coverage >= min_coverage:
            validated_v.append((x_start, x_end))
    
    return validated_h, validated_v


def detect_gutter_lines(
    gutter_mask: NDArray,
    config: "DetectorConfig",
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Detect horizontal and vertical gutter lines from mask.
    
    Args:
        gutter_mask: Binary gutter mask
        config: Detector configuration
        
    Returns:
        Tuple of (h_lines, v_lines) where each line is (start, end)
    """
    if not HAS_NUMPY:
        return [], []
    
    h, w = gutter_mask.shape
    
    # Projections
    h_proj = np.sum(gutter_mask, axis=1)
    v_proj = np.sum(gutter_mask, axis=0)
    
    h_proj_norm = h_proj / w if w > 0 else h_proj
    v_proj_norm = v_proj / h if h > 0 else v_proj
    
    # Try scipy for smoothing if available
    try:
        from scipy import signal
        from scipy.signal import find_peaks
        
        h_smooth = signal.savgol_filter(h_proj_norm, window_length=min(51, len(h_proj_norm)//2 | 1), polyorder=3)
        v_smooth = signal.savgol_filter(v_proj_norm, window_length=min(51, len(v_proj_norm)//2 | 1), polyorder=3)
        
        h_distance = int(max(30, 0.015 * h))
        v_distance = int(max(30, 0.015 * w))
        
        h_threshold = np.percentile(h_smooth, 85)
        v_threshold = np.percentile(v_smooth, 85)
        
        h_peaks, _ = find_peaks(h_smooth, height=h_threshold, distance=h_distance, prominence=0.015)
        v_peaks, _ = find_peaks(v_smooth, height=v_threshold, distance=v_distance, prominence=0.015)
    except ImportError:
        # Fallback without scipy
        h_smooth = h_proj_norm
        v_smooth = v_proj_norm
        h_peaks = np.where(h_smooth > np.percentile(h_smooth, 85))[0]
        v_peaks = np.where(v_smooth > np.percentile(v_smooth, 85))[0]
    
    # Limit peaks
    if len(h_peaks) > 10:
        peak_heights = h_smooth[h_peaks]
        top_indices = np.argsort(peak_heights)[-10:]
        h_peaks = h_peaks[np.sort(top_indices)]
    
    if len(v_peaks) > 10:
        peak_heights = v_smooth[v_peaks]
        top_indices = np.argsort(peak_heights)[-10:]
        v_peaks = v_peaks[np.sort(top_indices)]
    
    # Group into lines
    h_lines = _group_peaks_to_lines(h_peaks)
    v_lines = _group_peaks_to_lines(v_peaks)
    
    return h_lines, v_lines


def _group_peaks_to_lines(peaks: NDArray) -> List[Tuple[int, int]]:
    """Group consecutive peaks into line regions."""
    if len(peaks) == 0:
        return []
    
    lines = []
    start = peaks[0]
    prev = start
    
    for pixel in peaks[1:]:
        if pixel > prev + 1:
            lines.append((start, prev))
            start = pixel
        prev = pixel
    
    lines.append((start, prev))
    
    # Expand thin lines
    expanded = []
    for y_start, y_end in lines:
        thickness = y_end - y_start + 1
        if thickness < 3:
            expand = (3 - thickness + 1) // 2
            y_start = max(0, y_start - expand)
            y_end = min(y_end + expand, 4096)
        expanded.append((y_start, y_end))
    
    return expanded


def panels_from_gutters(
    h_lines: List[Tuple[int, int]],
    v_lines: List[Tuple[int, int]],
    w: int,
    h: int,
    page_point_size: QSizeF,
    config: "DetectorConfig",
) -> List[QRectF]:
    """Generate panel rectangles from gutter positions.
    
    Args:
        h_lines: Horizontal gutter lines
        v_lines: Vertical gutter lines
        w, h: Image dimensions
        page_point_size: Page size in points
        config: Detector configuration
        
    Returns:
        List of panel rectangles
    """
    rects = []
    scale = w / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0
    
    # Add boundaries
    h_boundaries = [(0, 0)] + h_lines + [(h-1, h-1)]
    v_boundaries = [(0, 0)] + v_lines + [(w-1, w-1)]
    
    min_panel_w = max(config.min_rect_px, int(0.025 * w))
    min_panel_h = max(config.min_rect_px, int(0.025 * h))
    
    for i in range(len(h_boundaries) - 1):
        y_start = h_boundaries[i][1] + 1
        y_end = h_boundaries[i+1][0]
        h_cell = y_end - y_start
        if h_cell < min_panel_h:
            continue
        
        for j in range(len(v_boundaries) - 1):
            x_start = v_boundaries[j][1] + 1
            x_end = v_boundaries[j+1][0]
            w_cell = x_end - x_start
            if w_cell < min_panel_w:
                continue
            
            rect_w = x_end - x_start
            rect_h = y_end - y_start
            
            x_pts = x_start / scale
            y_pts = y_start / scale
            w_pts = rect_w / scale
            h_pts = rect_h / scale
            
            rects.append(QRectF(x_pts, y_pts, w_pts, h_pts))
    
    if config.debug:
        pdebug(f"[gutters] generated {len(rects)} panels from gutter grid")
    
    return rects


def gutter_based_detection(
    gray: NDArray,
    L: NDArray,
    w: int,
    h: int,
    page_point_size: QSizeF,
    config: "DetectorConfig",
    img_bgr: NDArray = None,
) -> List[QRectF]:
    """Full gutter-based detection pipeline.
    
    Args:
        gray: Grayscale image
        L: LAB L-channel
        w, h: Image dimensions
        page_point_size: Page size in points
        config: Detector configuration
        img_bgr: Optional color image
        
    Returns:
        List of detected panel rectangles
    """
    # Build gutter mask
    gutter_mask = make_gutter_mask(gray, L, config, img_bgr)
    
    # Detect gutter lines
    h_lines, v_lines = detect_gutter_lines(gutter_mask, config)
    pdebug(f"[gutters] h_lines raw={len(h_lines)} v_lines raw={len(v_lines)}")
    
    # Validate
    h_lines, v_lines = validate_gutter_lines(gutter_mask, h_lines, v_lines, config)
    pdebug(f"[gutters] h_lines valid={len(h_lines)} v_lines valid={len(v_lines)}")
    
    # Generate panels
    if h_lines and v_lines:
        rects = panels_from_gutters(h_lines, v_lines, w, h, page_point_size, config)
    else:
        rects = []
    
    return rects


# Import for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config import DetectorConfig
