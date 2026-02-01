"""Post-processing filters for panel detection.

Filters:
- Title row removal
- Empty panel filtering
- Nested rectangle suppression
- Reading order sorting
"""

from __future__ import annotations

from typing import List

from PySide6.QtCore import QRectF, QSizeF

from .utils import pdebug, union, HAS_CV2, HAS_NUMPY

if HAS_CV2:
    import cv2
if HAS_NUMPY:
    import numpy as np
    from numpy.typing import NDArray


def filter_title_rows(
    rects: List[QRectF],
    page_point_size: QSizeF,
    debug: bool = False,
) -> List[QRectF]:
    """Filter out title boxes at top of page.
    
    Only removes small, wide boxes in top 15% of page.
    
    Args:
        rects: Panel rectangles in page points
        page_point_size: Page dimensions
        debug: Enable debug logging
        
    Returns:
        Filtered list of rectangles
    """
    if not rects:
        return []
    
    keep = []
    for r in rects:
        # Only check boxes in top 15% of page
        in_top = r.top() < 0.15 * page_point_size.height()
        
        if in_top:
            h_ratio = r.height() / page_point_size.height()
            w_ratio = r.width() / page_point_size.width()
            
            is_short = h_ratio < 0.12
            is_wide = w_ratio > 0.35
            
            if is_short and is_wide:
                if debug:
                    pdebug(f"[title-row] Removed title at y={r.top():.0f}")
                continue
        
        keep.append(r)
    
    return keep


def filter_by_area(
    rects: List[QRectF],
    page_point_size: QSizeF,
    min_area_pct: float,
    max_area_pct: float,
) -> List[QRectF]:
    """Filter rectangles by area percentage.
    
    Args:
        rects: Panel rectangles
        page_point_size: Page dimensions
        min_area_pct: Minimum area percentage
        max_area_pct: Maximum area percentage
        
    Returns:
        Filtered list of rectangles
    """
    page_area = page_point_size.width() * page_point_size.height()
    if page_area <= 0:
        return rects
    
    min_area = page_area * min_area_pct
    max_area = page_area * max_area_pct
    
    return [r for r in rects
            if min_area <= (r.width() * r.height()) <= max_area]


def filter_empty_rects(
    rects: List[QRectF],
    gray: NDArray,
    w: int,
    h: int,
    page_point_size: QSizeF,
    min_content_ratio: float = 0.03,
) -> List[QRectF]:
    """Filter out rectangles that contain mostly empty background.
    
    Args:
        rects: Panel rectangles
        gray: Grayscale image
        w, h: Image dimensions
        page_point_size: Page dimensions
        min_content_ratio: Minimum ratio of dark pixels
        
    Returns:
        Filtered list of rectangles
    """
    if not rects or gray is None or not HAS_CV2 or not HAS_NUMPY:
        return rects
    
    scale = w / page_point_size.width() if page_point_size.width() > 0 else 1.0
    
    # Binary mask: dark pixels = content
    _, content_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    filtered = []
    for rect in rects:
        x = int(rect.left() * scale)
        y = int(rect.top() * scale)
        rw = int(rect.width() * scale)
        rh = int(rect.height() * scale)
        
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))
        
        region_mask = content_mask[y:y+rh, x:x+rw]
        
        if region_mask.size == 0:
            continue
        
        content_pixels = np.sum(region_mask == 255)
        content_ratio = content_pixels / region_mask.size
        
        if content_ratio >= min_content_ratio:
            filtered.append(rect)
        else:
            pdebug(f"[Filter] Rejected empty rect at ({rect.left():.0f},{rect.top():.0f})")
    
    return filtered


def filter_by_lab_content(
    rects: List[QRectF],
    img_bgr: NDArray,
    gray: NDArray,
    bg_lab: NDArray,
    w: int,
    h: int,
    page_point_size: QSizeF,
    min_non_bg_ratio: float,
    min_dim_ratio: float,
    delta: float,
) -> List[QRectF]:
    """Filter panels by Lab color distance from background.
    
    Args:
        rects: Panel rectangles
        img_bgr: BGR image
        gray: Grayscale image
        bg_lab: Background color in Lab
        w, h: Image dimensions
        page_point_size: Page dimensions
        min_non_bg_ratio: Minimum non-background ratio
        min_dim_ratio: Minimum dimension ratio vs median
        delta: Lab distance threshold
        
    Returns:
        Filtered list of rectangles
    """
    if not rects or img_bgr is None or not HAS_CV2 or not HAS_NUMPY:
        return rects
    
    scale = w / page_point_size.width() if page_point_size.width() > 0 else 1.0
    
    # Calculate median dimensions
    if len(rects) > 1:
        widths = [r.width() for r in rects]
        heights = [r.height() for r in rects]
        median_w = np.median(widths)
        median_h = np.median(heights)
    else:
        median_w = median_h = 0
    
    filtered = []
    for rect in rects:
        x = int(rect.left() * scale)
        y = int(rect.top() * scale)
        rw = int(rect.width() * scale)
        rh = int(rect.height() * scale)
        
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))
        
        roi_bgr = img_bgr[y:y+rh, x:x+rw]
        roi_gray = gray[y:y+rh, x:x+rw]
        
        if roi_bgr.size == 0:
            continue
        
        # Calculate non-background ratio
        roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        dist = np.linalg.norm(roi_lab - bg_lab, axis=2)
        non_bg = float(np.mean(dist > delta))
        
        # Check if too empty
        if non_bg < min_non_bg_ratio:
            gray_std = np.std(roi_gray) if roi_gray.size > 0 else 0.0
            if gray_std < 12.0:
                pdebug(f"[Lab] Dropped mostly-bg panel at ({rect.left():.0f},{rect.top():.0f})")
                continue
        
        # Check if too thin
        if median_w > 0 and median_h > 0:
            is_thin = (rect.height() < min_dim_ratio * median_h or
                      rect.width() < min_dim_ratio * median_w)
            if is_thin:
                content_threshold = max(min_non_bg_ratio * 1.8, 0.18)
                if non_bg < content_threshold:
                    pdebug(f"[Lab] Dropped thin+low-content panel")
                    continue
        
        filtered.append(rect)
    
    return filtered


def suppress_nested_rects(
    rects: List[QRectF],
    img_bgr: NDArray,
    bg_lab: NDArray,
    w: int,
    h: int,
    page_point_size: QSizeF,
    delta: float = 12.0,
    contain_thr: float = 0.90,
    area_ratio_thr: float = 0.25,
) -> List[QRectF]:
    """Handle nested rectangles by merging them.
    
    Args:
        rects: Panel rectangles
        img_bgr: BGR image
        bg_lab: Background color
        w, h: Image dimensions
        page_point_size: Page dimensions
        delta: Lab distance threshold
        contain_thr: Containment threshold
        area_ratio_thr: Area ratio threshold
        
    Returns:
        List with nested rectangles merged
    """
    if len(rects) <= 1 or img_bgr is None:
        return rects
    
    merged_indices = {}
    to_remove = set()
    
    for i, small in enumerate(rects):
        if i in to_remove:
            continue
        
        small_area = small.width() * small.height()
        
        for j, big in enumerate(rects):
            if i == j or j in to_remove:
                continue
            
            big_area = big.width() * big.height()
            
            if small_area >= big_area:
                continue
            
            area_ratio = small_area / big_area if big_area > 0 else 0
            if area_ratio > area_ratio_thr:
                continue
            
            inter = small.intersected(big)
            if inter.isEmpty():
                continue
            
            inter_area = inter.width() * inter.height()
            containment = inter_area / small_area if small_area > 0 else 0
            
            if containment >= contain_thr:
                merged_rect = small.united(big)
                merged_indices[j] = merged_rect
                to_remove.add(i)
                
                pdebug(f"[Nested] Merged at ({small.left():.0f},{small.top():.0f})")
                break
    
    result = []
    for i, r in enumerate(rects):
        if i in to_remove:
            continue
        if i in merged_indices:
            result.append(merged_indices[i])
        else:
            result.append(r)
    
    return result


def remove_header_footer_strips(
    rects: List[QRectF],
    page_h: float,
) -> List[QRectF]:
    """Remove header/footer strips (page numbers, title fragments).
    
    Args:
        rects: Panel rectangles in page points
        page_h: Page height in points
        
    Returns:
        Filtered list of rectangles
    """
    if not rects:
        return rects
    
    filtered = []
    for rect in rects:
        if rect.height() < 0.10 * page_h:
            if rect.top() < 0.06 * page_h or (rect.top() + rect.height()) > 0.94 * page_h:
                pdebug(f"[Strip] Removed header/footer at y={rect.top():.1f}")
                continue
        filtered.append(rect)
    
    return filtered


def sort_by_reading_order(
    rects: List[QRectF],
    rtl: bool = False,
) -> List[QRectF]:
    """Sort rectangles by reading order.
    
    Args:
        rects: Panel rectangles
        rtl: Right-to-left reading order
        
    Returns:
        Sorted list of rectangles
    """
    if not rects:
        return []
    
    sorted_by_top = sorted(rects, key=lambda r: r.top())
    
    rows = []
    current_row = [sorted_by_top[0]]
    row_threshold = 20
    
    for rect in sorted_by_top[1:]:
        if abs(rect.top() - current_row[0].top()) < row_threshold:
            current_row.append(rect)
        else:
            rows.append(current_row)
            current_row = [rect]
    rows.append(current_row)
    
    result = []
    for row in rows:
        if rtl:
            row_sorted = sorted(row, key=lambda r: -r.left())
        else:
            row_sorted = sorted(row, key=lambda r: r.left())
        result.extend(row_sorted)
    
    return result


def merge_overlapping_rects(rects: List[QRectF]) -> List[QRectF]:
    """Merge rectangles that overlap significantly.
    
    Args:
        rects: Panel rectangles
        
    Returns:
        Merged list of rectangles
    """
    if not rects:
        return []
    
    rects_sorted = sorted(rects, key=lambda r: r.width() * r.height(), reverse=True)
    
    merged = []
    used = set()
    
    for i, rect in enumerate(rects_sorted):
        if i in used:
            continue
        
        merged.append(rect)
        
        for j in range(i + 1, len(rects_sorted)):
            if j in used:
                continue
            
            other = rects_sorted[j]
            inter_rect = rect.intersected(other)
            
            if not inter_rect.isEmpty():
                overlap = inter_rect.width() * inter_rect.height()
                area_min = min(rect.width() * rect.height(), other.width() * other.height())
                overlap_ratio = overlap / area_min if area_min > 0 else 0
                
                if overlap_ratio > 0.4:
                    used.add(j)
    
    return merged
