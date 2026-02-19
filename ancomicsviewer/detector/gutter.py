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
    
    # Adaptive brightness threshold (cap at 240 so near-white gutters are included)
    bright_percentile = getattr(config, 'gutter_bright_percentile', 94)
    bright_thresh = min(np.percentile(L, bright_percentile), 240.0)
    
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
    # Use a shorter kernel to avoid destroying narrow gutters
    open_kernel_len = max(int(w * config.gutter_open_kernel_frac * 0.5), 15)
    
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
        
        # Keep horizontal stripes (wide and thin)
        if cw >= min_h_stripe_len and ch <= min_stripe_w:
            result[component_mask > 0] = 255
        # Keep vertical stripes (tall and thin)
        elif ch >= min_v_stripe_len and cw <= min_stripe_w:
            result[component_mask > 0] = 255
        # Also keep components with high aspect ratio (stripe-like)
        # even if shorter than min_length, as long as aspect > 4:1
        elif cw > 0 and ch > 0:
            aspect = max(cw / ch, ch / cw)
            if aspect >= 4.0 and max(cw, ch) >= min(min_h_stripe_len, min_v_stripe_len) * 0.4:
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
    
    Uses three complementary approaches:
    1. Mask-based: brightness + gradient uniformity + morphological filtering
    2. Profile-based: row/column mean brightness to find bright bands directly
    3. Hierarchical: first H-gutters (rows), then per-row V-gutters (columns)
    
    The hierarchical approach handles classic BD layouts where different rows
    have different column divisions (e.g., row 1 = 1 panel, row 2 = 3+2 panels).
    
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
    # Method 1: Mask-based detection (original)
    gutter_mask = make_gutter_mask(gray, L, config, img_bgr)
    
    h_lines, v_lines = detect_gutter_lines(gutter_mask, config)
    pdebug(f"[gutters] h_lines raw={len(h_lines)} v_lines raw={len(v_lines)}")
    
    h_lines, v_lines = validate_gutter_lines(gutter_mask, h_lines, v_lines, config)
    pdebug(f"[gutters] h_lines valid={len(h_lines)} v_lines valid={len(v_lines)}")
    
    # Method 2: Profile-based detection (simpler, more robust)
    # Always run and merge — profile catches gutters that mask misses
    h_prof, v_prof = _detect_gutters_by_profile(gray, w, h, config)
    pdebug(f"[gutters] profile: h={len(h_prof)} v={len(v_prof)}")
    h_lines = _merge_gutter_lines(h_lines, h_prof)
    v_lines = _merge_gutter_lines(v_lines, v_prof)
    pdebug(f"[gutters] merged: h={len(h_lines)} v={len(v_lines)}")
    
    # Method 3: Hierarchical detection — per-row V-gutter detection
    # This is the primary approach for layouts with varying columns per row
    if h_lines:
        hier_rects = _detect_gutters_hierarchical(
            gray, w, h, h_lines, page_point_size, config
        )
        if hier_rects:
            pdebug(f"[gutters] hierarchical: {len(hier_rects)} panels")
        
        # Also generate flat-grid panels for comparison
        flat_rects = []
        if h_lines or v_lines:
            flat_rects = panels_from_gutters(
                h_lines, v_lines, w, h, page_point_size, config
            )
        
        # Choose between hierarchical and flat results
        # Hierarchical is better when it finds row-specific columns,
        # but can over-segment when rows have varying content (false V-gutters).
        # Prefer hierarchical only if it finds modestly more panels.
        if hier_rects and flat_rects:
            hier_n = len(hier_rects)
            flat_n = len(flat_rects)
            # Hierarchical is preferred when:
            # - It found more structure (more panels)
            # - But not wildly more (< 2x flat + 3, to avoid over-segmentation)
            if hier_n > flat_n and hier_n <= flat_n * 2 + 3:
                pdebug(f"[gutters] using hierarchical ({hier_n}) over flat ({flat_n})")
                return hier_rects
            else:
                return flat_rects
        elif flat_rects:
            return flat_rects
        elif hier_rects:
            return hier_rects
        else:
            return []
    
    # No H-gutters: fall back to flat grid
    if h_lines or v_lines:
        rects = panels_from_gutters(h_lines, v_lines, w, h, page_point_size, config)
    else:
        rects = []
    
    return rects


def _detect_gutters_hierarchical(
    gray: NDArray,
    w: int,
    h: int,
    h_gutters: List[Tuple[int, int]],
    page_point_size: QSizeF,
    config: "DetectorConfig",
) -> List[QRectF]:
    """Hierarchical gutter detection: H-gutters define rows, then find
    V-gutters within each row independently.
    
    This handles classic BD layouts where each row can have a different
    number of columns (e.g., row 1 = 1 wide panel, row 2 = 3 columns).
    
    Args:
        gray: Grayscale image
        w, h: Image dimensions
        h_gutters: Horizontal gutters already detected
        page_point_size: Page size in points
        config: Detector configuration
        
    Returns:
        List of panel rectangles
    """
    if not HAS_NUMPY:
        return []
    
    scale = w / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0
    min_panel_w = max(config.min_rect_px, int(0.025 * w))
    min_panel_h = max(config.min_rect_px, int(0.025 * h))
    margin_w = int(w * 0.03)
    
    # Build row boundaries from H-gutters
    # rows = [(y_start, y_end), ...]
    boundaries = [(0, 0)] + list(h_gutters) + [(h - 1, h - 1)]
    rows = []
    for i in range(len(boundaries) - 1):
        y_start = boundaries[i][1] + 1
        y_end = boundaries[i + 1][0]
        row_h = y_end - y_start
        if row_h >= min_panel_h:
            rows.append((y_start, y_end))
    
    if not rows:
        return []
    
    rects = []
    for row_y1, row_y2 in rows:
        row_strip = gray[row_y1:row_y2, :]
        row_h_px = row_y2 - row_y1
        
        if row_h_px < 10 or row_strip.size == 0:
            continue
        
        # Find V-gutters within this row using column mean brightness
        interior = row_strip[:, margin_w:w - margin_w]
        if interior.size == 0:
            continue
            
        col_means = interior.mean(axis=0)
        col_median = np.median(col_means)
        col_max = col_means.max()
        
        # Need meaningful contrast to detect gutters
        min_thickness = max(config.min_gutter_px, 3)
        
        v_gutters_in_row = []
        # Require both relative contrast AND absolute brightness.
        # Real gutters are near-white (>225), not just brighter than content.
        # Page 13 real V-gutters: mean 241-253. Page 5 noise: 202-216.
        abs_bright_min = 225  # absolute minimum brightness for gutter pixels
        if col_max - col_median > 30 and col_max > abs_bright_min:
            v_thresh = col_median + 0.55 * (col_max - col_median)
            # Only consider columns that are absolutely bright
            bright_cols = np.where(
                (col_means >= v_thresh) & (col_means >= abs_bright_min)
            )[0] + margin_w
            v_candidates = _group_bright_pixels(bright_cols, min_thickness, min_gap=5)
            
            # Validate: gutter must span most of the row height
            for x_start, x_end in v_candidates:
                # Skip margins (left/right edges of page)
                if x_start < margin_w + 3 or x_end > w - margin_w - 3:
                    continue
                band = row_strip[:, x_start:x_end + 1]
                if band.size == 0:
                    continue
                # Check absolute brightness AND coverage
                bright_pct = (band >= abs_bright_min).mean()
                if bright_pct >= 0.50:
                    v_gutters_in_row.append((x_start, x_end))
        
        # Now build cells for this row
        # Also check for horizontal sub-gutters within each cell (stacked panels)
        col_boundaries = [(0, 0)] + v_gutters_in_row + [(w - 1, w - 1)]
        
        for j in range(len(col_boundaries) - 1):
            x_start = col_boundaries[j][1] + 1
            x_end = col_boundaries[j + 1][0]
            cell_w = x_end - x_start
            if cell_w < min_panel_w:
                continue
            
            # Check for horizontal sub-gutters within this cell
            cell = gray[row_y1:row_y2, x_start:x_end]
            sub_h_gutters = _find_sub_gutters_in_cell(
                cell, row_h_px, cell_w, config
            )
            
            if sub_h_gutters:
                # Split cell vertically
                sub_boundaries = [(0, 0)] + sub_h_gutters + [(row_h_px - 1, row_h_px - 1)]
                for k in range(len(sub_boundaries) - 1):
                    sy = sub_boundaries[k][1] + 1
                    ey = sub_boundaries[k + 1][0]
                    sh = ey - sy
                    if sh < min_panel_h:
                        continue
                    abs_y = row_y1 + sy
                    rects.append(QRectF(
                        x_start / scale,
                        abs_y / scale,
                        cell_w / scale,
                        sh / scale,
                    ))
            else:
                # Single cell, no sub-division
                rects.append(QRectF(
                    x_start / scale,
                    row_y1 / scale,
                    cell_w / scale,
                    row_h_px / scale,
                ))
    
    return rects


def _find_sub_gutters_in_cell(
    cell: NDArray,
    cell_h: int,
    cell_w: int,
    config: "DetectorConfig",
) -> List[Tuple[int, int]]:
    """Find horizontal sub-gutters within a single cell.
    
    Used to detect stacked panels within a column of a row.
    Only returns sub-gutters if they represent clear divisions.
    
    Args:
        cell: Grayscale cell image
        cell_h: Cell height in pixels
        cell_w: Cell width in pixels
        config: Detector configuration
        
    Returns:
        List of (y_start, y_end) sub-gutters in cell-local coordinates
    """
    if not HAS_NUMPY or cell.size == 0:
        return []
    
    # Don't subdivide cells that are already small
    if cell_h < 60 or cell_w < 30:
        return []
    
    # Margin: skip a few pixels from left/right edges of the cell
    cell_margin = max(3, int(cell_w * 0.05))
    interior = cell[:, cell_margin:cell_w - cell_margin]
    if interior.size == 0:
        return []
    
    row_means = interior.mean(axis=1)
    median_val = np.median(row_means)
    max_val = row_means.max()
    
    # Need strong contrast for sub-gutters
    if max_val - median_val < 40:
        return []
    
    # Sub-gutters must be truly bright (near-white), not just brighter
    abs_bright_min = 225
    if max_val < abs_bright_min:
        return []
    
    thresh = median_val + 0.65 * (max_val - median_val)
    min_thickness = max(config.min_gutter_px, 3)
    
    bright_rows = np.where((row_means >= thresh) & (row_means >= abs_bright_min))[0]
    candidates = _group_bright_pixels(bright_rows, min_thickness, min_gap=3)
    
    # Filter: sub-gutter must span most of cell width and not be at edges
    validated = []
    min_margin = int(cell_h * 0.12)  # don't split too close to edges
    for y_start, y_end in candidates:
        if y_start < min_margin or y_end > cell_h - min_margin:
            continue
        band = cell[y_start:y_end + 1, cell_margin:cell_w - cell_margin]
        if band.size == 0:
            continue
        bright_pct = (band >= abs_bright_min).mean()
        if bright_pct >= 0.55:
            validated.append((y_start, y_end))
    
    return validated


def _detect_gutters_by_profile(
    gray: NDArray,
    w: int,
    h: int,
    config: "DetectorConfig",
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Detect gutters by analyzing row/column mean brightness profiles.
    
    Simpler approach: a row of pixels with high mean brightness is likely
    a horizontal gutter. Works even when mask-based approach fails due
    to morphological filtering.
    
    Returns:
        (h_lines, v_lines)
    """
    if not HAS_NUMPY:
        return [], []
    
    margin_h = int(h * 0.03)  # ignore top/bottom margins
    margin_w = int(w * 0.03)  # ignore left/right margins
    min_thickness = max(config.min_gutter_px, 3)
    
    # Horizontal gutters: mean brightness per row
    # Use the interior portion (skip left/right margins)
    interior = gray[margin_h:h - margin_h, margin_w:w - margin_w]
    row_means = interior.mean(axis=1)
    
    # A gutter row should be significantly brighter than the page median
    page_median = np.median(row_means)
    # Threshold: 80% of the way from median to max
    # Real gutters are near-white (250+), this excludes bright content (230s)
    row_max = row_means.max()
    thresh = page_median + 0.80 * (row_max - page_median)
    # Only trigger if there's meaningful contrast
    if row_max - page_median < 30:
        return [], []
    
    bright_rows = np.where(row_means >= thresh)[0] + margin_h
    h_lines = _group_bright_pixels(bright_rows, min_thickness, min_gap=5)
    
    # Filter: gutter must span at least 60% of interior width
    validated_h = []
    for y_start, y_end in h_lines:
        band = gray[y_start:y_end + 1, margin_w:w - margin_w]
        # Check that at least 60% of pixels in the band are bright
        bright_pct = (band >= thresh * 0.9).mean()
        if bright_pct >= 0.5:
            validated_h.append((y_start, y_end))
    
    # Vertical gutters: mean brightness per column  
    col_means = interior.mean(axis=0)
    bright_cols = np.where(col_means >= thresh)[0] + margin_w
    v_lines = _group_bright_pixels(bright_cols, min_thickness, min_gap=5)
    
    # Filter: gutter must span at least 60% of interior height
    validated_v = []
    for x_start, x_end in v_lines:
        band = gray[margin_h:h - margin_h, x_start:x_end + 1]
        bright_pct = (band >= thresh * 0.9).mean()
        if bright_pct >= 0.5:
            validated_v.append((x_start, x_end))
    
    return validated_h, validated_v


def _group_bright_pixels(
    indices: NDArray,
    min_thickness: int,
    min_gap: int = 5,
) -> List[Tuple[int, int]]:
    """Group consecutive pixel indices into line regions."""
    if len(indices) == 0:
        return []
    
    groups = []
    start = indices[0]
    prev = start
    for idx in indices[1:]:
        if idx > prev + min_gap:
            if prev - start + 1 >= min_thickness:
                groups.append((int(start), int(prev)))
            start = idx
        prev = idx
    if prev - start + 1 >= min_thickness:
        groups.append((int(start), int(prev)))
    
    return groups


def _merge_gutter_lines(
    existing: List[Tuple[int, int]],
    new: List[Tuple[int, int]],
    overlap_threshold: int = 15,
) -> List[Tuple[int, int]]:
    """Merge two lists of gutter lines, avoiding duplicates."""
    if not new:
        return existing
    if not existing:
        return new
    
    merged = list(existing)
    for ns, ne in new:
        n_mid = (ns + ne) / 2
        is_duplicate = False
        for es, ee in existing:
            e_mid = (es + ee) / 2
            if abs(n_mid - e_mid) < overlap_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append((ns, ne))
    
    merged.sort(key=lambda x: x[0])
    return merged


# Import for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config import DetectorConfig
