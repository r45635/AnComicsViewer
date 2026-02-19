"""Contour hierarchy analysis for panel detection.

Uses cv2.RETR_TREE instead of RETR_EXTERNAL to get parent-child
relationships between contours. This enables:
- Finding panel borders as large closed contours containing children
- Filtering out speech bubbles (too small/irregular)
- Filtering out text (too many small children)
- Detecting panels even when they overlap

Pipeline:
1. Edge detection + morphological closing
2. Find contours with full hierarchy (RETR_TREE)
3. Analyze parent-child relationships
4. Select contours that look like panels:
   - Large enough (> min_area)
   - Rectangular enough (high fill ratio)
   - Contains children (has content inside)
   - Not too many children (not the page itself)
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
class ContourInfo:
    """Information about a contour and its hierarchy."""
    index: int
    contour: NDArray
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: float
    fill_ratio: float           # contour_area / bbox_area
    rectangularity: float       # How rectangular (0-1)
    children_count: int         # Direct children
    depth: int                  # Hierarchy depth (0 = root)
    parent_index: int           # -1 if no parent
    has_content: bool           # Contains child contours


def analyze_contour_hierarchy(
    gray: NDArray,
    w: int,
    h: int,
    config: "DetectorConfig",
) -> List[ContourInfo]:
    """Extract contours with hierarchy analysis.

    Args:
        gray: Grayscale image
        w, h: Image dimensions
        config: Detector configuration

    Returns:
        List of ContourInfo objects for potential panels
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return []

    page_area = float(w * h)

    # Multi-method edge detection for robustness
    edges = _robust_edge_detection(gray)

    # Morphological closing to connect panel borders
    k_size = max(3, int(0.005 * min(w, h)) | 1)
    kernel = np.ones((k_size, k_size), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours with FULL hierarchy
    contours, hierarchy = cv2.findContours(
        closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours is None or hierarchy is None:
        return []

    hierarchy = hierarchy[0]  # Shape: (N, 4) - [next, prev, child, parent]

    # Build hierarchy info
    children_count = _count_children(hierarchy)
    depths = _compute_depths(hierarchy)

    # Analyze each contour
    contour_infos = []

    min_area = page_area * getattr(config, 'hierarchy_min_area_pct', 0.008)
    max_area = page_area * getattr(config, 'hierarchy_max_area_pct', 0.95)
    min_fill = getattr(config, 'hierarchy_min_fill', 0.40)
    min_rect = getattr(config, 'hierarchy_min_rectangularity', 0.50)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        bbox_area = bw * bh
        if bbox_area == 0:
            continue

        fill_ratio = area / bbox_area
        if fill_ratio < min_fill:
            continue

        # Rectangularity: how close to a rectangle
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]
        rectangularity = area / rect_area if rect_area > 0 else 0

        if rectangularity < min_rect:
            continue

        parent_idx = hierarchy[i][3]
        depth = depths[i]
        n_children = children_count.get(i, 0)

        # Must have content (children) but not too many (that's the page)
        has_content_flag = n_children > 0
        is_page = (area / page_area > 0.85)

        if is_page:
            continue

        contour_infos.append(ContourInfo(
            index=i,
            contour=contour,
            bbox=(x, y, bw, bh),
            area=area,
            fill_ratio=fill_ratio,
            rectangularity=rectangularity,
            children_count=n_children,
            depth=depth,
            parent_index=parent_idx,
            has_content=has_content_flag,
        ))

    pdebug(f"[Hierarchy] Found {len(contour_infos)} panel-like contours")
    return contour_infos


def _robust_edge_detection(gray: NDArray) -> NDArray:
    """Multi-method edge detection combining Canny at multiple thresholds."""
    if not HAS_CV2 or not HAS_NUMPY:
        return gray

    h, w = gray.shape[:2]

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Multi-threshold Canny
    edges_low = cv2.Canny(blurred, 30, 90)
    edges_mid = cv2.Canny(blurred, 50, 150)
    edges_high = cv2.Canny(blurred, 80, 200)

    # Combine: union of all edges
    combined = cv2.bitwise_or(edges_low, edges_mid)
    combined = cv2.bitwise_or(combined, edges_high)

    return combined


def _count_children(hierarchy: NDArray) -> dict:
    """Count direct children for each contour."""
    children_count = {}

    for i in range(len(hierarchy)):
        parent = hierarchy[i][3]
        if parent >= 0:
            children_count[parent] = children_count.get(parent, 0) + 1

    return children_count


def _compute_depths(hierarchy: NDArray) -> List[int]:
    """Compute depth of each contour in the hierarchy."""
    n = len(hierarchy)
    depths = [0] * n

    for i in range(n):
        depth = 0
        parent = hierarchy[i][3]
        while parent >= 0:
            depth += 1
            parent = hierarchy[parent][3]
            if depth > 20:  # Safety limit
                break
        depths[i] = depth

    return depths


def hierarchy_based_detection(
    gray: NDArray,
    w: int,
    h: int,
    page_point_size: QSizeF,
    config: "DetectorConfig",
) -> List[QRectF]:
    """Full hierarchy-based panel detection pipeline.

    Args:
        gray: Grayscale image
        w, h: Image dimensions
        page_point_size: Page size in points
        config: Detector configuration

    Returns:
        List of detected panel rectangles
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return []

    scale = w / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0

    # Analyze hierarchy
    contour_infos = analyze_contour_hierarchy(gray, w, h, config)

    if not contour_infos:
        return []

    # Score and filter contours
    scored = _score_panel_candidates(contour_infos, w, h)

    # Convert to QRectF
    rects = []
    for info, score in scored:
        x, y, bw, bh = info.bbox
        px = x / scale
        py = y / scale
        pw = bw / scale
        ph = bh / scale
        rects.append(QRectF(px, py, pw, ph))

    # Merge overlapping
    rects = merge_rects(rects, iou_thresh=0.30)

    # Remove nested (keep larger)
    rects = _remove_nested(rects)

    pdebug(f"[Hierarchy] Final: {len(rects)} panels")
    return rects


def _score_panel_candidates(
    contour_infos: List[ContourInfo],
    w: int,
    h: int,
) -> List[Tuple[ContourInfo, float]]:
    """Score contour candidates as potential panels.

    Higher score = more likely to be a panel.
    """
    page_area = float(w * h)
    scored = []

    for info in contour_infos:
        score = 0.0

        # Rectangularity bonus
        score += info.rectangularity * 2.0

        # Fill ratio bonus
        score += info.fill_ratio * 1.5

        # Area sweet spot (5-35% of page)
        area_frac = info.area / page_area
        if 0.05 <= area_frac <= 0.35:
            score += 2.0
        elif 0.02 <= area_frac <= 0.50:
            score += 1.0

        # Having children = has content
        if info.has_content:
            score += 1.0

        # Moderate children count (2-50 is typical for a panel with drawings)
        if 2 <= info.children_count <= 50:
            score += 0.5
        elif info.children_count > 100:
            score -= 0.5  # Probably too noisy

        # Depth 1-2 is ideal (direct children of page)
        if info.depth <= 2:
            score += 0.5
        elif info.depth > 4:
            score -= 0.5

        # Aspect ratio: panels are usually wider than tall or roughly square
        bw, bh = info.bbox[2], info.bbox[3]
        aspect = bw / bh if bh > 0 else 1.0
        if 0.3 <= aspect <= 3.0:
            score += 0.5

        if score > 2.0:
            scored.append((info, score))

    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Limit to reasonable number
    max_panels = 16
    return scored[:max_panels]


def _remove_nested(rects: List[QRectF]) -> List[QRectF]:
    """Remove nested rectangles, keeping the outer ones."""
    if len(rects) <= 1:
        return rects

    # Sort by area (largest first)
    rects_sorted = sorted(rects, key=lambda r: r.width() * r.height(), reverse=True)

    kept = []
    for rect in rects_sorted:
        is_nested = False
        for outer in kept:
            inter = rect.intersected(outer)
            if not inter.isEmpty():
                inter_area = inter.width() * inter.height()
                rect_area = rect.width() * rect.height()
                if rect_area > 0 and inter_area / rect_area > 0.85:
                    is_nested = True
                    break
        if not is_nested:
            kept.append(rect)

    return kept


# Type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config import DetectorConfig
