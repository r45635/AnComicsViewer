"""Multi-scale detection with consensus merging.

Runs panel detection at multiple scales and merges results.
Panels agreed upon by multiple scales are kept with higher confidence,
dramatically reducing false positives and catching panels missed at one scale.

Pipeline:
1. Run detection at 0.6x, 1.0x, 1.5x scales
2. Scale all rects back to original dimensions
3. Consensus merge: keep rects agreed by >= 2 scales
4. Score-weighted selection for ambiguous cases
"""

from __future__ import annotations

from typing import List, Tuple, Callable, Optional, Dict
from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSizeF

from .utils import pdebug, iou, merge_rects, HAS_CV2, HAS_NUMPY

if HAS_CV2:
    import cv2
if HAS_NUMPY:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class ScaleResult:
    """Detection result at a specific scale."""
    scale: float
    rects: List[QRectF]
    original_rects: List[QRectF]  # Before scaling back


@dataclass
class ConsensusRect:
    """A rectangle with consensus score across scales."""
    rect: QRectF
    agreement_count: int     # How many scales agree
    scales: List[float]      # Which scales detected it
    confidence: float        # Weighted confidence score


def multiscale_detect(
    detect_fn: Callable,
    gray: NDArray,
    L: NDArray,
    img_bgr: NDArray,
    w: int,
    h: int,
    page_point_size: QSizeF,
    config: "DetectorConfig",
    scales: Optional[List[float]] = None,
    min_agreement: int = 2,
) -> List[QRectF]:
    """Run detection at multiple scales and merge with consensus.

    Args:
        detect_fn: Detection function(gray, L, img_bgr, w, h, page_point_size, config) -> List[QRectF]
        gray: Grayscale image (original scale)
        L: LAB L-channel (original scale)
        img_bgr: BGR image (original scale)
        w, h: Image dimensions
        page_point_size: Page size in points
        config: Detector configuration
        scales: Scale factors to use (default: [0.6, 1.0, 1.5])
        min_agreement: Minimum number of scales that must agree

    Returns:
        Consensus-merged list of panel rectangles
    """
    if not HAS_CV2 or not HAS_NUMPY:
        return detect_fn(gray, L, img_bgr, w, h, page_point_size, config)

    if scales is None:
        scales = getattr(config, 'multiscale_factors', [0.6, 1.0, 1.5])

    pdebug(f"[MultiScale] Running detection at scales: {scales}")

    scale_results: List[ScaleResult] = []

    for scale in scales:
        if abs(scale - 1.0) < 0.01:
            # Original scale - no resize needed
            rects = detect_fn(gray, L, img_bgr, w, h, page_point_size, config)
            scale_results.append(ScaleResult(
                scale=scale,
                rects=rects,
                original_rects=list(rects),
            ))
        else:
            # Resize images
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w < 100 or new_h < 100:
                continue

            gray_s = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
            L_s = cv2.resize(L, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
            img_bgr_s = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

            # Detect at this scale
            rects_at_scale = detect_fn(gray_s, L_s, img_bgr_s, new_w, new_h, page_point_size, config)

            # Scale rects back to original image coordinates (page points are unchanged)
            # Since detection returns page points, no scaling needed - page_point_size is the same
            scale_results.append(ScaleResult(
                scale=scale,
                rects=rects_at_scale,
                original_rects=list(rects_at_scale),
            ))

        pdebug(f"[MultiScale] scale={scale:.1f}: {len(scale_results[-1].rects)} panels")

    if not scale_results:
        return []

    # If only one scale, return directly
    if len(scale_results) == 1:
        return scale_results[0].rects

    # Consensus merge
    consensus = consensus_merge(
        scale_results,
        min_agreement=min_agreement,
        iou_threshold=getattr(config, 'multiscale_iou_threshold', 0.30),
    )

    if consensus:
        pdebug(f"[MultiScale] Consensus: {len(consensus)} panels (min_agreement={min_agreement})")
        return consensus

    # Fallback: if consensus is empty, use the scale with most panels
    best = max(scale_results, key=lambda sr: len(sr.rects))
    pdebug(f"[MultiScale] No consensus, using best scale={best.scale}: {len(best.rects)} panels")
    return best.rects


def consensus_merge(
    scale_results: List[ScaleResult],
    min_agreement: int = 2,
    iou_threshold: float = 0.30,
) -> List[QRectF]:
    """Merge detection results across scales using consensus voting.

    For each rectangle, count how many scales detected a similar rectangle.
    Keep only those agreed upon by >= min_agreement scales.

    Args:
        scale_results: Detection results at different scales
        min_agreement: Minimum number of agreeing scales
        iou_threshold: IoU threshold to consider two rects as matching

    Returns:
        Consensus-approved list of rectangles
    """
    if not scale_results:
        return []

    # Collect all rects with their source scale
    all_rects: List[Tuple[QRectF, float]] = []
    for sr in scale_results:
        for rect in sr.rects:
            all_rects.append((rect, sr.scale))

    if not all_rects:
        return []

    # Build agreement matrix
    n = len(all_rects)
    used = [False] * n
    consensus_rects: List[ConsensusRect] = []

    for i in range(n):
        if used[i]:
            continue

        rect_i, scale_i = all_rects[i]
        matching_indices = [i]
        matching_scales = {scale_i}

        for j in range(i + 1, n):
            if used[j]:
                continue

            rect_j, scale_j = all_rects[j]

            if iou(rect_i, rect_j) >= iou_threshold:
                matching_indices.append(j)
                matching_scales.add(scale_j)

        if len(matching_scales) >= min_agreement:
            # Average the matching rects for better position
            avg_rect = _average_rects([all_rects[k][0] for k in matching_indices])

            # Confidence = agreement ratio * coverage diversity
            agreement = len(matching_scales) / len(scale_results)
            confidence = agreement

            consensus_rects.append(ConsensusRect(
                rect=avg_rect,
                agreement_count=len(matching_scales),
                scales=sorted(matching_scales),
                confidence=confidence,
            ))

            for k in matching_indices:
                used[k] = True

    # Sort by confidence
    consensus_rects.sort(key=lambda cr: cr.confidence, reverse=True)

    result = [cr.rect for cr in consensus_rects]

    # Final merge to remove any remaining overlaps
    result = merge_rects(result, iou_thresh=0.25)

    return result


def _average_rects(rects: List[QRectF]) -> QRectF:
    """Compute the average rectangle from a list of rects."""
    if not rects:
        return QRectF()

    if len(rects) == 1:
        return rects[0]

    avg_left = sum(r.left() for r in rects) / len(rects)
    avg_top = sum(r.top() for r in rects) / len(rects)
    avg_width = sum(r.width() for r in rects) / len(rects)
    avg_height = sum(r.height() for r in rects) / len(rects)

    return QRectF(avg_left, avg_top, avg_width, avg_height)


def score_detection_result(
    rects: List[QRectF],
    page_point_size: QSizeF,
) -> float:
    """Score a detection result for quality.

    Higher score = better detection. Used to compare different routes.

    Args:
        rects: Detected panels
        page_point_size: Page dimensions

    Returns:
        Quality score (0.0 to 1.0)
    """
    if not rects:
        return 0.0

    page_area = page_point_size.width() * page_point_size.height()
    if page_area <= 0:
        return 0.0

    n = len(rects)

    # Panel count score: 3-8 panels is ideal
    if 3 <= n <= 8:
        count_score = 1.0
    elif 2 <= n <= 12:
        count_score = 0.7
    elif n == 1:
        count_score = 0.3
    else:
        count_score = 0.4

    # Coverage score: panels should cover most of the page
    total_area = sum(r.width() * r.height() for r in rects)
    coverage = min(total_area / page_area, 1.0)
    coverage_score = min(coverage / 0.75, 1.0)  # Full score at 75% coverage

    # Overlap penalty: panels shouldn't overlap too much
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            overlap = iou(rects[i], rects[j])
            if overlap > 0.15:
                overlap_penalty += overlap * 0.2

    # Size uniformity: panels should be reasonably similar in size
    if n > 1:
        areas = [r.width() * r.height() for r in rects]
        min_area = min(areas)
        max_area = max(areas)
        ratio = min_area / max_area if max_area > 0 else 0
        uniformity_score = max(ratio, 0.2)  # At least 0.2
    else:
        uniformity_score = 0.5

    final_score = (
        count_score * 0.35 +
        coverage_score * 0.30 +
        uniformity_score * 0.15 +
        max(0, 1.0 - overlap_penalty) * 0.20
    )

    return min(1.0, max(0.0, final_score))


def select_best_result(
    results: Dict[str, List[QRectF]],
    page_point_size: QSizeF,
) -> Tuple[str, List[QRectF]]:
    """Select the best detection result from multiple methods.

    Args:
        results: Dict of {method_name: detected_rects}
        page_point_size: Page dimensions

    Returns:
        (best_method_name, best_rects)
    """
    if not results:
        return ("none", [])

    best_name = ""
    best_score = -1.0
    best_rects: List[QRectF] = []

    for name, rects in results.items():
        score = score_detection_result(rects, page_point_size)
        pdebug(f"[Score] {name}: {len(rects)} panels, score={score:.3f}")

        if score > best_score:
            best_score = score
            best_name = name
            best_rects = rects

    pdebug(f"[Score] Best: {best_name} (score={best_score:.3f})")
    return best_name, best_rects


# Type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config import DetectorConfig
