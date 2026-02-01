"""Freeform panel detection using watershed segmentation.

For complex layouts with:
- Parallelograms
- Tinted backgrounds
- Watercolor pages
- Non-rectangular panels
"""

from __future__ import annotations

from typing import List, Tuple, Optional

from PySide6.QtCore import QRectF, QSizeF

from .utils import PanelRegion, pdebug, HAS_CV2, HAS_NUMPY

if HAS_CV2:
    import cv2
if HAS_NUMPY:
    import numpy as np
    from numpy.typing import NDArray


def estimate_background_color_lab(
    img_bgr: NDArray, 
    sample_pct: float = 0.03
) -> Tuple[float, float, float]:
    """Estimate background color by sampling image borders in Lab color space.
    
    Args:
        img_bgr: Input image in BGR format
        sample_pct: Percentage of image dimensions to use for border sampling
        
    Returns:
        Tuple of (L, a, b) median values
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return (255.0, 128.0, 128.0)
    
    h, w = img_bgr.shape[:2]
    border_size = max(int(min(h, w) * sample_pct), 5)
    
    samples = []
    samples.append(img_bgr[0:border_size, :])
    samples.append(img_bgr[h-border_size:h, :])
    samples.append(img_bgr[:, 0:border_size])
    samples.append(img_bgr[:, w-border_size:w])
    
    border_pixels = np.vstack([s.reshape(-1, 3) for s in samples])
    border_lab = cv2.cvtColor(border_pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2Lab)
    border_lab = border_lab.reshape(-1, 3).astype(np.float32)
    
    L_med = np.median(border_lab[:, 0])
    a_med = np.median(border_lab[:, 1])
    b_med = np.median(border_lab[:, 2])
    
    pdebug(f"[Freeform] Background Lab: L={L_med:.1f}, a={a_med:.1f}, b={b_med:.1f}")
    return (L_med, a_med, b_med)


def make_background_mask(
    img_bgr: NDArray, 
    bg_lab: Tuple[float, float, float], 
    delta: float = 12.0,
) -> NDArray:
    """Create binary mask of background pixels based on Lab distance.
    
    Args:
        img_bgr: Input image in BGR format
        bg_lab: Background color in Lab (L, a, b)
        delta: Maximum Lab distance to be considered background
        
    Returns:
        Binary mask (uint8, 255 = background, 0 = foreground)
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8) if HAS_NUMPY else None
    
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    bg_array = np.array(bg_lab, dtype=np.float32)
    dist = np.linalg.norm(img_lab - bg_array, axis=2)
    
    mask_bg = (dist < delta).astype(np.uint8) * 255
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_OPEN, kernel_open)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, kernel_close)
    
    bg_pct = (np.sum(mask_bg == 255) / mask_bg.size) * 100
    pdebug(f"[Freeform] Background mask: {bg_pct:.1f}% of image")
    
    return mask_bg


def segment_panels_watershed(
    img_bgr: NDArray, 
    mask_bg: NDArray, 
    sure_fg_ratio: float = 0.45,
) -> NDArray:
    """Segment panels using watershed algorithm.
    
    Args:
        img_bgr: Input image in BGR format
        mask_bg: Background mask (255 = background)
        sure_fg_ratio: Ratio of max distance for sure foreground
        
    Returns:
        Labeled regions (int32, 0 = background, 1+ = regions)
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return np.zeros(img_bgr.shape[:2], dtype=np.int32) if HAS_NUMPY else None
    
    mask_fg = cv2.bitwise_not(mask_bg)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    sure_bg = cv2.dilate(mask_bg, kernel, iterations=2)
    
    dist = cv2.distanceTransform(mask_fg, cv2.DIST_L2, 5)
    dist_max = dist.max()
    threshold_val = sure_fg_ratio * dist_max
    _, sure_fg = cv2.threshold(dist, threshold_val, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    
    sure_bg_bin = (sure_bg == 255).astype(np.uint8)
    sure_fg_bin = (sure_fg == 255).astype(np.uint8)
    unknown = cv2.subtract(sure_bg_bin, sure_fg_bin)
    
    n_labels, markers = cv2.connectedComponents(sure_fg)
    pdebug(f"[Freeform] Connected components: {n_labels - 1}")
    
    markers = markers + 1
    markers[unknown == 1] = 0
    markers = cv2.watershed(img_bgr, markers)
    
    unique_labels = np.unique(markers)
    valid_labels = unique_labels[(unique_labels > 1)]
    pdebug(f"[Freeform] Watershed regions: {len(valid_labels)}")
    
    return markers


def has_content(
    region: PanelRegion, 
    mask_bg: NDArray, 
    min_content_ratio: float = 0.05
) -> bool:
    """Check if a region contains actual content.
    
    Args:
        region: Panel region to check
        mask_bg: Background mask
        min_content_ratio: Minimum ratio of non-background pixels
        
    Returns:
        True if region has sufficient content
    """
    x, y, w, h = region.bbox
    region_mask_bg = mask_bg[y:y+h, x:x+w]
    
    if region_mask_bg.size == 0:
        return False
    
    foreground_pixels = np.sum(region_mask_bg == 0)
    content_ratio = foreground_pixels / region_mask_bg.size
    
    return content_ratio >= min_content_ratio


def extract_panel_regions(
    markers: NDArray, 
    img_shape: Tuple[int, int],
    img_bgr: NDArray,
    mask_bg: NDArray,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.95,
    min_fill_ratio: float = 0.25,
    min_content_ratio: float = 0.05,
    approx_eps_ratio: float = 0.01,
) -> List[PanelRegion]:
    """Extract and filter panel regions from watershed markers.
    
    Args:
        markers: Watershed output
        img_shape: Image dimensions (h, w)
        img_bgr: Original image
        mask_bg: Background mask
        min_area_ratio: Minimum region area as fraction
        max_area_ratio: Maximum region area as fraction
        min_fill_ratio: Minimum fill ratio
        min_content_ratio: Minimum content ratio
        approx_eps_ratio: Epsilon for polygon approximation
        
    Returns:
        List of filtered PanelRegion objects
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return []
    
    h, w = img_shape
    img_area = h * w
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio
    
    regions = []
    unique_labels = np.unique(markers)
    
    for label in unique_labels:
        if label <= 1:
            continue
        
        region_mask = (markers == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        if area < min_area or area > max_area:
            continue
        
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox = (x, y, bw, bh)
        bbox_area = bw * bh
        
        if bbox_area == 0:
            continue
        
        fill_ratio = area / bbox_area
        if fill_ratio < min_fill_ratio:
            continue
        
        touches_border = (x == 0 or y == 0 or x + bw >= w or y + bh >= h)
        
        rect = cv2.minAreaRect(contour)
        obb = cv2.boxPoints(rect)
        obb = np.intp(obb)
        
        perimeter = cv2.arcLength(contour, True)
        epsilon = approx_eps_ratio * perimeter
        poly = cv2.approxPolyDP(contour, epsilon, True)
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = x + bw / 2, y + bh / 2
        
        region = PanelRegion(
            contour=contour,
            poly=poly,
            bbox=bbox,
            obb=obb,
            area=area,
            fill_ratio=fill_ratio,
            touches_border=touches_border,
            centroid=(cx, cy)
        )
        
        if not has_content(region, mask_bg, min_content_ratio):
            continue
        
        regions.append(region)
    
    pdebug(f"[Freeform] Extracted {len(regions)} regions")
    return regions


def merge_overlapping_regions(
    regions: List[PanelRegion], 
    iou_thr: float = 0.20
) -> List[PanelRegion]:
    """Merge regions with significant overlap.
    
    Args:
        regions: List of PanelRegion objects
        iou_thr: IoU threshold for merging
        
    Returns:
        List of merged regions
    """
    if len(regions) <= 1 or not HAS_CV2 or not HAS_NUMPY:
        return regions
    
    merged = []
    used = set()
    
    for i, reg in enumerate(regions):
        if i in used:
            continue
        
        to_merge = [reg]
        to_merge_indices = {i}
        
        for j, other in enumerate(regions):
            if j <= i or j in used:
                continue
            
            x1, y1, w1, h1 = reg.bbox
            x2, y2, w2, h2 = other.bbox
            
            xi = max(x1, x2)
            yi = max(y1, y2)
            wi = max(0, min(x1 + w1, x2 + w2) - xi)
            hi = max(0, min(y1 + h1, y2 + h2) - yi)
            
            inter_area = wi * hi
            union_area = w1 * h1 + w2 * h2 - inter_area
            
            if union_area == 0:
                continue
            
            iou = inter_area / union_area
            
            if iou > iou_thr:
                to_merge.append(other)
                to_merge_indices.add(j)
        
        if len(to_merge) > 1:
            all_points = np.vstack([r.contour for r in to_merge])
            hull = cv2.convexHull(all_points)
            
            area = cv2.contourArea(hull)
            x, y, bw, bh = cv2.boundingRect(hull)
            bbox = (x, y, bw, bh)
            fill_ratio = area / (bw * bh) if (bw * bh) > 0 else 0
            
            rect = cv2.minAreaRect(hull)
            obb = cv2.boxPoints(rect)
            obb = np.intp(obb)
            
            perimeter = cv2.arcLength(hull, True)
            poly = cv2.approxPolyDP(hull, 0.01 * perimeter, True)
            
            M = cv2.moments(hull)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + bw / 2, y + bh / 2
            
            merged_reg = PanelRegion(
                contour=hull,
                poly=poly,
                bbox=bbox,
                obb=obb,
                area=area,
                fill_ratio=fill_ratio,
                touches_border=False,
                centroid=(cx, cy)
            )
            
            merged.append(merged_reg)
            used.update(to_merge_indices)
        else:
            merged.append(reg)
            used.add(i)
    
    return merged


def sort_reading_order(
    regions: List[PanelRegion], 
    rtl: bool = False
) -> List[PanelRegion]:
    """Sort regions in reading order.
    
    Args:
        regions: List of PanelRegion objects
        rtl: Right-to-left reading order
        
    Returns:
        Sorted list of regions
    """
    if not regions or not HAS_NUMPY:
        return regions
    
    sorted_by_y = sorted(regions, key=lambda r: r.centroid[1])
    
    heights = [r.bbox[3] for r in regions]
    median_height = np.median(heights) if heights else 100
    
    rows = []
    current_row = []
    current_y = None
    
    for reg in sorted_by_y:
        cy = reg.centroid[1]
        
        if current_y is None:
            current_y = cy
            current_row = [reg]
        elif abs(cy - current_y) < 0.5 * median_height:
            current_row.append(reg)
        else:
            rows.append(current_row)
            current_row = [reg]
            current_y = cy
    
    if current_row:
        rows.append(current_row)
    
    result = []
    for row in rows:
        row_sorted = sorted(row, key=lambda r: r.centroid[0], reverse=rtl)
        result.extend(row_sorted)
    
    return result


def freeform_detection(
    img_bgr: NDArray,
    w: int,
    h: int,
    page_point_size: QSizeF,
    config: "DetectorConfig",
) -> List[QRectF]:
    """Full freeform detection pipeline.
    
    Args:
        img_bgr: Image in BGR format
        w, h: Image dimensions
        page_point_size: Page size in points
        config: Detector configuration
        
    Returns:
        List of detected panel rectangles
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return []
    
    # Estimate background
    bg_lab = estimate_background_color_lab(img_bgr)
    
    # Create background mask
    mask_bg = make_background_mask(img_bgr, bg_lab, delta=config.freeform_bg_delta)
    
    # Watershed segmentation
    markers = segment_panels_watershed(img_bgr, mask_bg, sure_fg_ratio=config.sure_fg_ratio)
    
    # Extract regions
    regions = extract_panel_regions(
        markers, (h, w), img_bgr, mask_bg,
        min_area_ratio=config.min_area_ratio_freeform,
        max_area_ratio=config.max_area_pct,
        min_fill_ratio=config.min_fill_ratio_freeform,
        min_content_ratio=0.05,
        approx_eps_ratio=config.approx_eps_ratio
    )
    
    # Merge overlapping
    regions = merge_overlapping_regions(regions, iou_thr=config.iou_merge_thr)
    
    # Sort by reading order
    regions = sort_reading_order(regions, rtl=config.reading_rtl)
    
    # Convert to QRectF
    scale = w / page_point_size.width()
    rects = [region.to_qrectf(scale) for region in regions]
    
    return rects


# Import for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config import DetectorConfig
