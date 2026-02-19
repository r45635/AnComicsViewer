"""Line Segment Detection (LSD) based panel detection.

Detects panel borders by finding actual line segments in the image,
then building enclosed rectangular regions from intersections.

This is more robust than threshold-based detection because it detects
borders regardless of color (black, blue, red, gray gutters all
produce strong gradient lines).

Pipeline:
1. LSD/HoughLinesP to detect line segments
2. Filter to keep near-horizontal and near-vertical lines
3. Cluster lines into groups (merge close parallel lines)
4. Find intersections to build grid of regions
5. Validate regions as panels
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSizeF

from .utils import pdebug, merge_rects, HAS_CV2, HAS_NUMPY

if HAS_CV2:
    import cv2
if HAS_NUMPY:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class LineSegment:
    """A detected line segment."""
    x1: float
    y1: float
    x2: float
    y2: float
    angle: float       # Angle in degrees (0=horizontal, 90=vertical)
    length: float      # Pixel length
    is_horizontal: bool
    is_vertical: bool


def detect_line_segments(
    gray: NDArray,
    min_length_frac: float = 0.08,
    angle_tolerance: float = 8.0,
) -> List[LineSegment]:
    """Detect line segments using LSD with HoughLinesP fallback.

    Args:
        gray: Grayscale image
        min_length_frac: Minimum line length as fraction of image dimension
        angle_tolerance: Maximum deviation from H/V in degrees

    Returns:
        List of detected LineSegment objects (H or V only)
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return []

    h, w = gray.shape[:2]
    min_length = int(min(w, h) * min_length_frac)

    segments: List[LineSegment] = []

    # Try LSD first (parameter-free, more robust)
    lsd_segments = _detect_lsd(gray, min_length)

    # HoughLinesP as complement (catches strong lines LSD may miss)
    hough_segments = _detect_hough(gray, min_length)

    # Combine and deduplicate
    all_segments = lsd_segments + hough_segments
    segments = _filter_hv_lines(all_segments, angle_tolerance)
    segments = _deduplicate_segments(segments, merge_dist=max(8, int(0.005 * min(w, h))))

    pdebug(f"[LSD] Detected {len(segments)} H/V line segments "
           f"(LSD={len(lsd_segments)}, Hough={len(hough_segments)})")

    return segments


def _detect_lsd(gray: NDArray, min_length: int) -> List[LineSegment]:
    """Detect lines using Line Segment Detector."""
    if not HAS_CV2:
        return []

    segments = []
    try:
        lsd = cv2.createLineSegmentDetector(
            cv2.LSD_REFINE_STD,
            _scale=0.8,
            _sigma_scale=0.6,
            _quant=2.0,
            _ang_th=22.5,
            _log_eps=0,
            _density_th=0.7,
            _n_bins=1024,
        )
        lines, _widths, _precs, _nfas = lsd.detect(gray)
    except Exception:
        # Fallback: simpler LSD creation
        try:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
            lines, _widths, _precs, _nfas = lsd.detect(gray)
        except Exception as e:
            pdebug(f"[LSD] LSD detection failed: {e}")
            return []

    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        seg = _make_segment(float(x1), float(y1), float(x2), float(y2))
        if seg.length >= min_length:
            segments.append(seg)

    return segments


def _detect_hough(gray: NDArray, min_length: int) -> List[LineSegment]:
    """Detect lines using probabilistic Hough transform."""
    if not HAS_CV2:
        return []

    segments = []

    # Multi-scale edge detection for robustness
    for low_t, high_t in [(30, 100), (50, 150), (80, 200)]:
        edges = cv2.Canny(gray, low_t, high_t, apertureSize=3)

        # Dilate to connect broken edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(50, min_length // 3),
            minLineLength=min_length,
            maxLineGap=max(5, min_length // 8),
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                seg = _make_segment(float(x1), float(y1), float(x2), float(y2))
                segments.append(seg)

    return segments


def _make_segment(x1: float, y1: float, x2: float, y2: float) -> LineSegment:
    """Create a LineSegment from endpoints."""
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    angle = np.degrees(np.arctan2(abs(dy), abs(dx))) if HAS_NUMPY else 0.0

    return LineSegment(
        x1=x1, y1=y1, x2=x2, y2=y2,
        angle=angle, length=length,
        is_horizontal=(angle < 15.0),
        is_vertical=(angle > 75.0),
    )


def _filter_hv_lines(
    segments: List[LineSegment],
    angle_tolerance: float = 8.0,
) -> List[LineSegment]:
    """Keep only near-horizontal and near-vertical segments."""
    filtered = []
    for seg in segments:
        if seg.angle <= angle_tolerance:
            seg.is_horizontal = True
            seg.is_vertical = False
            filtered.append(seg)
        elif seg.angle >= (90.0 - angle_tolerance):
            seg.is_horizontal = False
            seg.is_vertical = True
            filtered.append(seg)
    return filtered


def _deduplicate_segments(
    segments: List[LineSegment],
    merge_dist: int = 8,
) -> List[LineSegment]:
    """Merge segments that are very close and parallel."""
    if not segments or not HAS_NUMPY:
        return segments

    # Separate H and V
    h_segs = [s for s in segments if s.is_horizontal]
    v_segs = [s for s in segments if s.is_vertical]

    h_merged = _merge_parallel_segments(h_segs, merge_dist, horizontal=True)
    v_merged = _merge_parallel_segments(v_segs, merge_dist, horizontal=False)

    return h_merged + v_merged


def _merge_parallel_segments(
    segments: List[LineSegment],
    merge_dist: int,
    horizontal: bool,
) -> List[LineSegment]:
    """Merge parallel segments that are close together."""
    if not segments:
        return []

    # Sort by perpendicular position
    if horizontal:
        segments.sort(key=lambda s: (s.y1 + s.y2) / 2)
    else:
        segments.sort(key=lambda s: (s.x1 + s.x2) / 2)

    merged = []
    group = [segments[0]]

    for seg in segments[1:]:
        if horizontal:
            prev_pos = (group[-1].y1 + group[-1].y2) / 2
            curr_pos = (seg.y1 + seg.y2) / 2
        else:
            prev_pos = (group[-1].x1 + group[-1].x2) / 2
            curr_pos = (seg.x1 + seg.x2) / 2

        if abs(curr_pos - prev_pos) <= merge_dist:
            group.append(seg)
        else:
            merged.append(_merge_segment_group(group, horizontal))
            group = [seg]

    merged.append(_merge_segment_group(group, horizontal))
    return merged


def _merge_segment_group(
    group: List[LineSegment],
    horizontal: bool,
) -> LineSegment:
    """Merge a group of close parallel segments into one."""
    if len(group) == 1:
        return group[0]

    # Take longest segment as base, extend with group extents
    longest = max(group, key=lambda s: s.length)

    if horizontal:
        y_avg = np.mean([(s.y1 + s.y2) / 2 for s in group])
        x_min = min(min(s.x1, s.x2) for s in group)
        x_max = max(max(s.x1, s.x2) for s in group)
        return _make_segment(x_min, y_avg, x_max, y_avg)
    else:
        x_avg = np.mean([(s.x1 + s.x2) / 2 for s in group])
        y_min = min(min(s.y1, s.y2) for s in group)
        y_max = max(max(s.y1, s.y2) for s in group)
        return _make_segment(x_avg, y_min, x_avg, y_max)


def cluster_lines_to_positions(
    segments: List[LineSegment],
    w: int,
    h: int,
    min_gap_frac: float = 0.04,
) -> Tuple[List[int], List[int]]:
    """Cluster line positions into discrete H/V positions.

    Groups lines that are close together into single positions,
    representing panel borders.

    Args:
        segments: Detected line segments
        w, h: Image dimensions
        min_gap_frac: Minimum gap between clusters as fraction of image size

    Returns:
        (h_positions, v_positions) - Y coords of H lines, X coords of V lines
    """
    if not HAS_NUMPY:
        return [], []

    min_h_gap = int(h * min_gap_frac)
    min_v_gap = int(w * min_gap_frac)

    # Collect positions weighted by line length
    h_positions_raw = []
    v_positions_raw = []

    for seg in segments:
        if seg.is_horizontal:
            pos = int((seg.y1 + seg.y2) / 2)
            h_positions_raw.append((pos, seg.length))
        elif seg.is_vertical:
            pos = int((seg.x1 + seg.x2) / 2)
            v_positions_raw.append((pos, seg.length))

    h_positions = _cluster_1d(h_positions_raw, min_h_gap)
    v_positions = _cluster_1d(v_positions_raw, min_v_gap)

    # Filter positions near borders (likely page edges, not gutters)
    border_h = int(h * 0.03)
    border_v = int(w * 0.03)
    h_positions = [p for p in h_positions if border_h < p < h - border_h]
    v_positions = [p for p in v_positions if border_v < p < w - border_v]

    pdebug(f"[LSD] Clustered: {len(h_positions)} H-lines, {len(v_positions)} V-lines")

    return h_positions, v_positions


def _cluster_1d(
    positions_weights: List[Tuple[int, float]],
    min_gap: int,
) -> List[int]:
    """Cluster 1D positions using weighted averaging."""
    if not positions_weights or not HAS_NUMPY:
        return []

    # Sort by position
    sorted_pw = sorted(positions_weights, key=lambda x: x[0])

    clusters = []
    current_cluster = [sorted_pw[0]]

    for pos, weight in sorted_pw[1:]:
        last_pos = current_cluster[-1][0]
        if pos - last_pos <= min_gap:
            current_cluster.append((pos, weight))
        else:
            clusters.append(current_cluster)
            current_cluster = [(pos, weight)]
    clusters.append(current_cluster)

    # Weighted average for each cluster
    result = []
    for cluster in clusters:
        total_weight = sum(w for _, w in cluster)
        if total_weight > 0:
            weighted_pos = sum(p * w for p, w in cluster) / total_weight
            result.append(int(weighted_pos))

    return sorted(result)


def panels_from_line_grid(
    h_positions: List[int],
    v_positions: List[int],
    w: int,
    h: int,
    page_point_size: QSizeF,
    min_panel_frac: float = 0.03,
) -> List[QRectF]:
    """Generate panel rectangles from line positions.

    Creates a grid of panels between detected line positions.

    Args:
        h_positions: Y coordinates of horizontal lines
        v_positions: X coordinates of vertical lines
        w, h: Image dimensions
        page_point_size: Page size in points
        min_panel_frac: Minimum panel dimension as fraction

    Returns:
        List of panel rectangles in page point coordinates
    """
    scale = w / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0

    # Add page boundaries
    h_bounds = [0] + h_positions + [h]
    v_bounds = [0] + v_positions + [w]

    min_panel_w = int(w * min_panel_frac)
    min_panel_h = int(h * min_panel_frac)

    rects = []
    for i in range(len(h_bounds) - 1):
        y_start = h_bounds[i]
        y_end = h_bounds[i + 1]
        panel_h = y_end - y_start
        if panel_h < min_panel_h:
            continue

        for j in range(len(v_bounds) - 1):
            x_start = v_bounds[j]
            x_end = v_bounds[j + 1]
            panel_w = x_end - x_start
            if panel_w < min_panel_w:
                continue

            # Small inset to exclude the border lines themselves
            inset = 2
            x_pts = (x_start + inset) / scale
            y_pts = (y_start + inset) / scale
            w_pts = max(1, (panel_w - 2 * inset)) / scale
            h_pts = max(1, (panel_h - 2 * inset)) / scale

            rects.append(QRectF(x_pts, y_pts, w_pts, h_pts))

    pdebug(f"[LSD] Generated {len(rects)} panels from line grid")
    return rects


def line_based_detection(
    gray: NDArray,
    w: int,
    h: int,
    page_point_size: QSizeF,
    config: "DetectorConfig",
    img_bgr: NDArray = None,
) -> List[QRectF]:
    """Full LSD-based panel detection pipeline.

    Args:
        gray: Grayscale image
        w, h: Image dimensions
        page_point_size: Page size in points
        config: Detector configuration
        img_bgr: Optional color image for gradient analysis

    Returns:
        List of detected panel rectangles
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return []

    # Detect line segments
    min_len = getattr(config, 'lsd_min_length_frac', 0.08)
    angle_tol = getattr(config, 'lsd_angle_tolerance', 8.0)
    segments = detect_line_segments(gray, min_length_frac=min_len, angle_tolerance=angle_tol)

    if not segments:
        pdebug("[LSD] No line segments detected")
        return []

    # Also detect gradient-based border lines for colored borders
    if img_bgr is not None:
        gradient_segments = detect_gradient_borders(img_bgr, min_length_frac=min_len)
        if gradient_segments:
            segments = segments + gradient_segments
            segments = _deduplicate_segments(segments, merge_dist=max(8, int(0.005 * min(w, h))))
            pdebug(f"[LSD] After gradient merge: {len(segments)} segments")

    # Cluster into positions
    min_gap = getattr(config, 'lsd_min_gap_frac', 0.04)
    h_positions, v_positions = cluster_lines_to_positions(segments, w, h, min_gap_frac=min_gap)

    if not h_positions and not v_positions:
        pdebug("[LSD] No valid line positions found")
        return []

    # Generate panels
    min_frac = getattr(config, 'lsd_min_panel_frac', 0.03)
    rects = panels_from_line_grid(h_positions, v_positions, w, h, page_point_size, min_panel_frac=min_frac)

    return rects


def detect_gradient_borders(
    img_bgr: NDArray,
    min_length_frac: float = 0.08,
) -> List[LineSegment]:
    """Detect panel borders using gradient magnitude analysis.

    Works for colored borders (blue, red, gray) that threshold misses.
    Detects strong-gradient bands in both H and V directions.

    Args:
        img_bgr: Image in BGR format
        min_length_frac: Minimum line length as fraction

    Returns:
        List of detected line segments
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return []

    h, w = img_bgr.shape[:2]
    min_length = int(min(w, h) * min_length_frac)

    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Compute gradient magnitude using Scharr (more accurate than Sobel)
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize to 0-255
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold strong gradients
    thresh = np.percentile(grad_norm, 92)
    _, strong_grad = cv2.threshold(grad_norm, thresh, 255, cv2.THRESH_BINARY)

    # Morphological operations to extract directional structures
    segments = []

    # Horizontal borders: use horizontal kernel
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length // 2, 1))
    h_lines_mask = cv2.morphologyEx(strong_grad, cv2.MORPH_OPEN, h_kernel)

    # Extract horizontal line segments from mask
    h_contours, _ = cv2.findContours(h_lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in h_contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw >= min_length and ch < max(10, int(0.01 * h)):
            y_mid = y + ch // 2
            seg = _make_segment(float(x), float(y_mid), float(x + cw), float(y_mid))
            segments.append(seg)

    # Vertical borders: use vertical kernel
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length // 2))
    v_lines_mask = cv2.morphologyEx(strong_grad, cv2.MORPH_OPEN, v_kernel)

    v_contours, _ = cv2.findContours(v_lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in v_contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if ch >= min_length and cw < max(10, int(0.01 * w)):
            x_mid = x + cw // 2
            seg = _make_segment(float(x_mid), float(y), float(x_mid), float(y + ch))
            segments.append(seg)

    pdebug(f"[Gradient] Detected {len(segments)} gradient-based border segments")
    return segments


# Type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config import DetectorConfig
