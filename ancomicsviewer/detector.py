"""Panel detection engine for AnComicsViewer.

Optimized heuristic-based comic panel detection using OpenCV.
Features:
- Multiple detection routes (adaptive threshold, LAB, Canny)
- Recursive gutter-based splitting
- Title row filtering
- Configurable via DetectorConfig dataclass
- Thread-safe with result caching
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading

from PySide6.QtCore import QRectF, QSizeF
from PySide6.QtGui import QImage

from .config import DetectorConfig
from .image_utils import (
    pdebug,
    qimage_to_numpy_rgba,
    rgba_to_grayscale,
    rgba_to_lab_l,
    check_dependencies,
    HAS_CV2,
    HAS_NUMPY,
)

if HAS_CV2:
    import cv2
if HAS_NUMPY:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class DebugInfo:
    """Debug information from detection pass."""
    vertical_splits: List[Tuple[float, float, float, float]]  # (x, y, w, h) in page points
    horizontal_splits: List[Tuple[float, float, float, float]]

    @classmethod
    def empty(cls) -> "DebugInfo":
        return cls(vertical_splits=[], horizontal_splits=[])


class PanelDetector:
    """High-performance comic panel detector.

    Uses OpenCV heuristics with multiple fallback routes:
    1. Adaptive threshold + morphology (primary)
    2. LAB L-channel based detection (fallback)
    3. Canny edge detection (optional fallback)

    Thread-safe with internal locking for shared state.
    """

    # Thread pool for parallel processing
    _executor: Optional[ThreadPoolExecutor] = None
    _executor_lock = threading.Lock()

    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize detector with configuration.

        Args:
            config: Detection parameters. Uses defaults if None.
        """
        self.config = config or DetectorConfig()
        self._last_debug = DebugInfo.empty()
        self._lock = threading.Lock()

    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """Get or create shared thread pool executor."""
        with cls._executor_lock:
            if cls._executor is None:
                cls._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="panel_detect")
            return cls._executor

    @property
    def last_debug(self) -> DebugInfo:
        """Get debug info from last detection pass (thread-safe)."""
        with self._lock:
            return self._last_debug

    def detect_panels(self, qimage: QImage, page_point_size: QSizeF) -> List[QRectF]:
        """Detect comic panels in a rendered page image.

        Args:
            qimage: Rendered page as QImage
            page_point_size: Page dimensions in points (72 DPI)

        Returns:
            List of panel rectangles in page point coordinates, sorted by reading order
        """
        if not HAS_CV2 or not HAS_NUMPY:
            pdebug("OpenCV/NumPy not available -> no detection")
            return []

        try:
            with self._lock:
                self._last_debug = DebugInfo.empty()

            self._log_params()

            # Convert QImage to numpy array
            arr = qimage_to_numpy_rgba(qimage)
            if arr is None:
                pdebug("QImage conversion failed")
                return []

            h, w = arr.shape[:2]
            pdebug(f"Image size: {w}x{h}")

            # Handle very large images by scaling down temporarily for detection
            # but scale results back up to original coordinates
            scale_factor = 1.0
            max_width = 2400  # Maximum working width for detection
            if w > max_width:
                scale_factor = max_width / w
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                pdebug(f"Scaling down from {w}x{h} to {new_w}x{new_h} (factor: {scale_factor:.2f})")
                arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = arr.shape[:2]

            # Prepare grayscale and LAB L channel
            gray = rgba_to_grayscale(arr)
            L = rgba_to_lab_l(arr, apply_clahe=True)

            # Try detection routes in order
            rects = self._try_detection_routes(gray, L, w, h, page_point_size)

            # Filter title rows (top 35% of page)
            if rects and self.config.filter_title_rows:
                rects = self._filter_title_rows(L, rects, w, h, page_point_size)
                pdebug(f"After title-row filter -> {len(rects)} rects")

            # Filter by minimum area
            if rects:
                rects = self._filter_by_area(rects, page_point_size)
                pdebug(f"After area filter -> {len(rects)} rects")

            # Sort by reading order
            rects = self._sort_by_reading_order(rects)
            pdebug(f"Reading order: RTL={self.config.reading_rtl}")

            return rects

        except Exception:
            pdebug(f"detect_panels error:\n{traceback.format_exc()}")
            return []

    def _log_params(self) -> None:
        """Log current detection parameters."""
        c = self.config
        pdebug(
            "params:",
            f"ab={c.adaptive_block} C={c.adaptive_C}",
            f"mk={c.morph_kernel} mi={c.morph_iter}",
            f"min_area={c.min_area_pct:.3f} max_area={c.max_area_pct:.2f}",
            f"min_fill={c.min_fill_ratio:.2f} min_px={c.min_rect_px}",
            f"light_col={c.light_col_rel:.2f} light_row={c.light_row_rel:.2f}",
            f"gutter:min_px={c.min_gutter_px} cov>={c.gutter_cov_min:.2f}",
            f"title:on={c.filter_title_rows} psk={c.proj_smooth_k}"
        )

    def _try_detection_routes(
        self, gray: NDArray, L: NDArray, w: int, h: int, page_point_size: QSizeF
    ) -> List[QRectF]:
        """Simple detection using adaptive threshold (proven to work)."""

        # Use adaptive threshold route only
        mask = self._adaptive_route(gray)
        rects = self._rects_from_mask(mask, w, h, page_point_size)
        pdebug(f"Adaptive route -> {len(rects)} rects")
        
        return rects

    def _adaptive_route(self, gray: NDArray) -> NDArray:
        """Primary detection route using adaptive threshold.

        Uses bilateral filter + adaptive threshold + morphological closing.
        """
        c = self.config
        h, w = gray.shape[:2]
        
        # Adapt block size to image resolution
        # For very large images, use relative block size
        # For standard images (1200px wide): block_size = 99
        # For large images (2400px wide): block_size = 199 
        relative_block = 99 * (w / 1200.0)
        block_size = int(relative_block) | 1  # Make odd
        block_size = max(51, min(201, block_size))  # Clamp between 51 and 201
        
        # Use bilateral filter to preserve edges while smoothing text
        gray_smooth = cv2.bilateralFilter(gray, 9, 75, 75)

        th = cv2.adaptiveThreshold(
            gray_smooth, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, 5
        )

        # Morphological closing to connect borders
        # Also adapt kernel size
        kernel_size = int(c.morph_kernel * (w / 1200.0)) | 1
        kernel_size = max(3, min(15, kernel_size))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=c.morph_iter)

    def _lab_route(self, L: NDArray) -> NDArray:
        """LAB L-channel fallback route."""
        th = cv2.adaptiveThreshold(
            L, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51, 5
        )
        kernel = np.ones((7, 7), np.uint8)
        return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    def _canny_route(self, gray: NDArray) -> NDArray:
        """Canny edge-based fallback route."""
        edges = cv2.Canny(gray, 60, 180)
        kernel = np.ones((5, 5), np.uint8)
        dil = cv2.dilate(edges, kernel, iterations=2)
        return cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kernel, iterations=2)

    def _gutter_based_detection(
        self, 
        gray: NDArray, 
        L: NDArray, 
        w: int, 
        h: int, 
        page_point_size: QSizeF
    ) -> List[QRectF]:
        """Detect panels by finding white gutters (separations) instead of panels.
        
        This inverse approach works better when panels contain lots of text that
        fragments the standard detection.
        """
        # Use high-contrast binarization to find white spaces
        _, binary = cv2.threshold(L, 220, 255, cv2.THRESH_BINARY)
        
        # Dilate to connect nearby white pixels (gutters)
        gutter_kernel = np.ones((3, 3), np.uint8)
        gutters = cv2.dilate(binary, gutter_kernel, iterations=1)
        
        # Invert: black becomes white (panels), white becomes black (gutters)
        panels_mask = cv2.bitwise_not(gutters)
        
        # Clean up with morphology
        clean_kernel = np.ones((5, 5), np.uint8)
        panels_mask = cv2.morphologyEx(panels_mask, cv2.MORPH_CLOSE, clean_kernel, iterations=2)
        panels_mask = cv2.morphologyEx(panels_mask, cv2.MORPH_OPEN, clean_kernel, iterations=1)
        
        # Extract rectangles
        return self._rects_from_mask(panels_mask, w, h, page_point_size)

    def _rects_from_mask(
        self, mask: NDArray, w: int, h: int, page_point_size: QSizeF
    ) -> List[QRectF]:
        """Extract and filter rectangles from binary mask.
        
        Uses relaxed thresholds for initial extraction to catch fragments,
        proper filtering happens later after merging.
        """
        c = self.config

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        page_area_px = float(w * h)
        scale = w / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0

        # Relaxed threshold for initial extraction (half of normal)
        min_area_initial = c.min_area_pct / 2.0  # More permissive
        min_px_dyn = max(c.min_rect_px // 2, int(c.min_rect_frac * min(w, h) / 2))

        rects: List[QRectF] = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)

            # Area filter (relaxed)
            area_px = cw * ch
            area_pct = area_px / page_area_px
            if area_pct < min_area_initial or area_pct > c.max_area_pct:
                continue

            # Fill ratio filter (relaxed)
            contour_area = float(cv2.contourArea(contour))
            if contour_area <= 0:
                continue
            fill = contour_area / float(area_px)
            if fill < (c.min_fill_ratio * 0.8):  # 20% more permissive
                continue

            # Size filter (relaxed)
            if cw < min_px_dyn or ch < min_px_dyn:
                continue

            # Convert to page points
            px = x / scale
            py = y / scale
            pw = cw / scale
            ph = ch / scale
            rects.append(QRectF(px, py, pw, ph))

        return self._merge_rects(rects, iou_thresh=0.25)

    def _merge_rects(self, rects: List[QRectF], iou_thresh: float) -> List[QRectF]:
        """Merge overlapping rectangles using IoU threshold."""
        if not rects:
            return []

        merged: List[QRectF] = []
        for r in rects:
            did_merge = False
            for i, m in enumerate(merged):
                if self._iou(r, m) >= iou_thresh:
                    merged[i] = self._union(r, m)
                    did_merge = True
                    break
            if not did_merge:
                merged.append(r)
        return merged

    @staticmethod
    def _iou(a: QRectF, b: QRectF) -> float:
        """Calculate Intersection over Union."""
        inter = a.intersected(b)
        if inter.isEmpty():
            return 0.0
        ia = inter.width() * inter.height()
        ua = a.width() * a.height() + b.width() * b.height() - ia
        return ia / ua if ua > 0 else 0.0

    @staticmethod
    def _union(a: QRectF, b: QRectF) -> QRectF:
        """Calculate bounding box union."""
        return QRectF(
            min(a.left(), b.left()),
            min(a.top(), b.top()),
            max(a.right(), b.right()) - min(a.left(), b.left()),
            max(a.bottom(), b.bottom()) - min(a.top(), b.top()),
        )

    def _split_rects_by_light(
        self, L_img: NDArray, rects_pts: List[QRectF],
        W: int, H: int, page_point_size: QSizeF
    ) -> List[QRectF]:
        """Split rectangles by detecting bright gutters in L channel."""
        scale = W / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0

        out: List[QRectF] = []
        for rp in rects_pts:
            x0 = max(0, min(int(round(rp.left() * scale)), W - 1))
            y0 = max(0, min(int(round(rp.top() * scale)), H - 1))
            ww = max(1, min(int(round(rp.width() * scale)), W - x0))
            hh = max(1, min(int(round(rp.height() * scale)), H - y0))
            roi = L_img[y0:y0+hh, x0:x0+ww]
            out.extend(self._split_by_light_recursive(roi, x0, y0, scale))

        return out

    def _split_by_light_recursive(
        self, roi: NDArray, x0: int, y0: int, scale: float
    ) -> List[QRectF]:
        """Recursively split ROI by detecting bright gutters."""
        c = self.config
        h, w = roi.shape[:2]

        stack = [(0, 0, w, h)]
        result: List[QRectF] = []

        while stack:
            x, y, ww, hh = stack.pop()

            # Stop conditions
            if ww < c.min_rect_px or hh < c.min_rect_px:
                result.append(QRectF((x0+x)/scale, (y0+y)/scale, ww/scale, hh/scale))
                continue

            if len(result) + len(stack) >= c.max_panels_per_page:
                result.append(QRectF((x0+x)/scale, (y0+y)/scale, ww/scale, hh/scale))
                continue

            sub = roi[y:y+hh, x:x+ww].astype(np.float32) / 255.0

            # Compute smoothed projections
            col_mean = sub.mean(axis=0)
            row_mean = sub.mean(axis=1)
            k = max(9, c.proj_smooth_k | 1)

            if ww >= k:
                col_mean = np.convolve(col_mean, np.ones(k)/k, mode="same")
            if hh >= k:
                row_mean = np.convolve(row_mean, np.ones(k)/k, mode="same")

            # Relative thresholds
            ct = (col_mean.max() - col_mean.min()) * c.light_col_rel + col_mean.mean()
            rt = (row_mean.max() - row_mean.min()) * c.light_row_rel + row_mean.mean()

            # Find candidate runs
            v_runs = self._find_runs(col_mean >= ct, c.min_gutter_px)
            h_runs = self._find_runs(row_mean >= rt, c.min_gutter_px)

            # Filter and find best split
            best = self._find_best_split(sub, v_runs, h_runs, ww, hh)

            if best is None:
                result.append(QRectF((x0+x)/scale, (y0+y)/scale, ww/scale, hh/scale))
            else:
                axis, cut, _ = best
                self._record_debug_split(axis, x0+x, y0+y, cut, ww, hh, scale)

                if axis == 'v':
                    stack.append((x + cut, y, ww - cut, hh))
                    stack.append((x, y, cut, hh))
                else:
                    stack.append((x, y + cut, ww, hh - cut))
                    stack.append((x, y, ww, cut))

        # Final size filter
        return [r for r in result
                if r.width() * scale >= c.min_rect_px and r.height() * scale >= c.min_rect_px]

    def _find_best_split(
        self, sub: NDArray, v_runs: List[Tuple[int, int]],
        h_runs: List[Tuple[int, int]], ww: int, hh: int
    ) -> Optional[Tuple[str, int, int]]:
        """Find the best gutter split from candidate runs."""
        c = self.config

        edge_margin_x = int(round(ww * c.edge_margin_frac))
        edge_margin_y = int(round(hh * c.edge_margin_frac))
        max_gutter_w = max(c.min_gutter_px, int(round(ww * c.max_gutter_px_frac)))
        max_gutter_h = max(c.min_gutter_px, int(round(hh * c.max_gutter_px_frac)))
        min_gutter_w = max(c.min_gutter_px, int(round(ww * c.min_gutter_frac)))
        min_gutter_h = max(c.min_gutter_px, int(round(hh * c.min_gutter_frac)))

        def filter_v(run: Tuple[int, int]) -> Optional[Tuple[int, int]]:
            a, b = run
            span = b - a
            if span < min_gutter_w or span > max_gutter_w:
                return None
            if a <= edge_margin_x or b >= ww - edge_margin_x:
                return None
            cov = sub[:, a:b].mean()
            if cov < c.gutter_cov_min:
                return None
            cut = (a + b) // 2
            if cut < c.min_rect_px or (ww - cut) < c.min_rect_px:
                return None
            return (span, cut)

        def filter_h(run: Tuple[int, int]) -> Optional[Tuple[int, int]]:
            a, b = run
            span = b - a
            if span < min_gutter_h or span > max_gutter_h:
                return None
            if a <= edge_margin_y or b >= hh - edge_margin_y:
                return None
            cov = sub[a:b, :].mean()
            if cov < c.gutter_cov_min:
                return None
            cut = (a + b) // 2
            if cut < c.min_rect_px or (hh - cut) < c.min_rect_px:
                return None
            return (span, cut)

        best: Optional[Tuple[str, int, int]] = None

        if v_runs:
            v_opts = [filter_v(r) for r in v_runs]
            v_opts = [o for o in v_opts if o is not None]
            if v_opts:
                v_span, v_cut = max(v_opts, key=lambda t: t[0])
                best = ('v', v_cut, v_span)

        if h_runs:
            h_opts = [filter_h(r) for r in h_runs]
            h_opts = [o for o in h_opts if o is not None]
            if h_opts:
                h_span, h_cut = max(h_opts, key=lambda t: t[0])
                if best is None or h_span > best[2]:
                    best = ('h', h_cut, h_span)

        return best

    def _record_debug_split(
        self, axis: str, x: int, y: int, cut: int, ww: int, hh: int, scale: float
    ) -> None:
        """Record split line for debug visualization."""
        with self._lock:
            if axis == 'v':
                self._last_debug.vertical_splits.append(
                    ((x + cut) / scale, y / scale, 0.0, hh / scale)
                )
            else:
                self._last_debug.horizontal_splits.append(
                    (x / scale, (y + cut) / scale, ww / scale, 0.0)
                )

    @staticmethod
    def _find_runs(bool_arr: NDArray, min_len: int) -> List[Tuple[int, int]]:
        """Find consecutive True runs of minimum length."""
        runs = []
        start = None

        # Convert to list and add sentinel
        values = list(bool_arr) + [False]

        for i, val in enumerate(values):
            if val and start is None:
                start = i
            elif not val and start is not None:
                if i - start >= min_len:
                    runs.append((start, i))
                start = None

        return runs

    def _filter_title_rows(
        self, L_img: NDArray, rects_pts: List[QRectF],
        W: int, H: int, page_point_size: QSizeF
    ) -> List[QRectF]:
        """Filter out title boxes at top of page.
        
        More selective: only removes small boxes in top 15% of page.
        Real panels are usually taller and don't get filtered.
        """
        if not rects_pts:
            return []
        
        c = self.config
        
        keep = []
        for r in rects_pts:
            # Only check boxes in top 15% of page (not 35%)
            in_top = r.top() < 0.15 * page_point_size.height()
            
            if in_top:
                # Additional checks to distinguish title from real panel:
                # Title boxes are usually:
                # - Short (< 12% of page height)
                # - Wide OR centered
                h_ratio = r.height() / page_point_size.height()
                w_ratio = r.width() / page_point_size.width()
                
                is_short = h_ratio < 0.12
                is_wide = w_ratio > 0.35
                
                # If short AND wide, likely a title
                if is_short and is_wide:
                    if c.debug:
                        pdebug(f"[title-row] Removed title box at y={r.top():.0f} h={r.height():.0f}")
                    continue
            
            keep.append(r)
        
        return keep

    def _filter_by_area(self, rects: List[QRectF], page_point_size: QSizeF) -> List[QRectF]:
        """Filter rectangles by minimum area percentage."""
        c = self.config
        page_area = page_point_size.width() * page_point_size.height()
        if page_area <= 0:
            return rects

        min_area = page_area * c.min_area_pct
        max_area = page_area * c.max_area_pct

        return [r for r in rects
                if min_area <= (r.width() * r.height()) <= max_area]

    def _sort_by_reading_order(self, rects: List[QRectF]) -> List[QRectF]:
        """Sort rectangles by reading order (LTR or RTL)."""
        # First, group by rows (similar Y positions)
        if not rects:
            return []
        
        # Sort by top first
        sorted_by_top = sorted(rects, key=lambda r: r.top())
        
        # Group into rows (panels with similar Y position)
        rows = []
        current_row = [sorted_by_top[0]]
        row_threshold = 20  # pixels tolerance for same row
        
        for rect in sorted_by_top[1:]:
            if abs(rect.top() - current_row[0].top()) < row_threshold:
                current_row.append(rect)
            else:
                rows.append(current_row)
                current_row = [rect]
        rows.append(current_row)
        
        # Sort each row by reading direction
        result = []
        for row in rows:
            if self.config.reading_rtl:
                # RTL: right to left (larger X first)
                row_sorted = sorted(row, key=lambda r: -r.left())
            else:
                # LTR: left to right (smaller X first)
                row_sorted = sorted(row, key=lambda r: r.left())
            result.extend(row_sorted)
        
        return result

    def _remove_contained_and_merge_overlaps(self, rects: List[QRectF]) -> List[QRectF]:
        """Remove rectangles contained in others and merge high overlaps.
        
        This handles cases where:
        1. Small fragments are fully inside a larger panel
        2. Two rectangles overlap significantly (e.g. 50% overlap)
        """
        if len(rects) < 2:
            return rects
        
        # Sort by area (largest first)
        sorted_rects = sorted(rects, key=lambda r: r.width() * r.height(), reverse=True)
        
        result = []
        
        for rect_a in sorted_rects:
            # Check if this rect is contained in any already-kept rect
            is_contained = False
            merge_index = None
            
            for idx, rect_b in enumerate(result):
                # Calculate intersection
                inter = rect_a.intersected(rect_b)
                if inter.isEmpty():
                    continue
                
                inter_area = inter.width() * inter.height()
                area_a = rect_a.width() * rect_a.height()
                area_b = rect_b.width() * rect_b.height()
                
                # Check if rect_a is contained in rect_b (>70% overlap)
                overlap_ratio_a = inter_area / area_a if area_a > 0 else 0
                overlap_ratio_b = inter_area / area_b if area_b > 0 else 0
                
                if overlap_ratio_a > 0.7:
                    is_contained = True
                    pdebug(f"  Removed contained: {rect_a.width():.0f}x{rect_a.height():.0f} ({overlap_ratio_a*100:.0f}% in {rect_b.width():.0f}x{rect_b.height():.0f})")
                    break
                
                # Check if significant overlap but not contained (merge case)
                # Only merge if neither is mostly contained (to avoid merging nested panels)
                smaller_area = min(area_a, area_b)
                if inter_area / smaller_area > 0.5 and overlap_ratio_a <= 0.7 and overlap_ratio_b <= 0.7:
                    # Merge by creating union
                    merge_index = idx
                    break
            
            if is_contained:
                # Skip this rectangle
                continue
            elif merge_index is not None:
                # Merge with existing rectangle
                union = self._union(rect_a, result[merge_index])
                pdebug(f"  Merged overlap: {rect_a.width():.0f}x{rect_a.height():.0f} + {result[merge_index].width():.0f}x{result[merge_index].height():.0f} → {union.width():.0f}x{union.height():.0f}")
                result[merge_index] = union
            else:
                # Keep this rectangle
                result.append(rect_a)
        
        if len(result) < len(rects):
            pdebug(f"  Containment/overlap cleanup: {len(rects)} → {len(result)} rects")
        
        return result

    def _merge_fragments(self, rects: List[QRectF], page_point_size: QSizeF) -> List[QRectF]:
        """Merge small rectangles that are likely fragments of a single panel.
        
        Improved algorithm:
        - Removes rectangles fully contained in others
        - Merges rectangles with high overlap
        - Detects fragments that align vertically or horizontally
        - Merges fragments that share similar X or Y coordinates
        - Handles text boxes that fragment into horizontal strips
        """
        if len(rects) < 2:
            return rects
        
        # Step 1: Remove contained rectangles and merge high overlaps
        rects = self._remove_contained_and_merge_overlaps(rects)
        if len(rects) < 2:
            return rects
        
        page_area = page_point_size.width() * page_point_size.height()
        page_width = page_point_size.width()
        page_height = page_point_size.height()
        
        # More aggressive fragment detection for text-heavy panels
        fragment_threshold = page_area * 0.025  # 2.5% of page
        
        # Identify potential fragments
        fragments = []
        normal_panels = []
        
        for r in rects:
            area = r.width() * r.height()
            if area < fragment_threshold:
                fragments.append(r)
            else:
                normal_panels.append(r)
        
        if len(fragments) < 2:
            return rects
        
        pdebug(f"  Fragment merge: {len(fragments)} fragments, {len(normal_panels)} normal panels")
        
        # Try to merge fragments using clustering
        merged = []
        used = set()
        
        for i, frag in enumerate(fragments):
            if i in used:
                continue
            
            # Build cluster of vertically or horizontally aligned fragments
            cluster = [frag]
            cluster_indices = {i}
            
            # Keep expanding cluster
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(fragments):
                    if j in used or j in cluster_indices:
                        continue
                    
                    # Check alignment with any fragment in cluster
                    for c in cluster:
                        # Horizontal alignment (similar Y, close X)
                        y_overlap = min(c.bottom(), other.bottom()) - max(c.top(), other.top())
                        x_gap = max(c.left(), other.left()) - min(c.right(), other.right())
                        
                        # Vertical alignment (similar X, close Y)
                        x_overlap = min(c.right(), other.right()) - max(c.left(), other.left())
                        y_gap = max(c.top(), other.top()) - min(c.bottom(), other.bottom())
                        
                        # Gap thresholds
                        max_h_gap = page_width * 0.02  # 2% horizontal gap
                        max_v_gap = page_height * 0.03  # 3% vertical gap
                        
                        # Horizontal strip (text lines)
                        if y_overlap > 0 and x_gap < max_h_gap:
                            cluster.append(other)
                            cluster_indices.add(j)
                            changed = True
                            break
                        
                        # Vertical strip
                        elif x_overlap > 0 and y_gap < max_v_gap:
                            cluster.append(other)
                            cluster_indices.add(j)
                            changed = True
                            break
            
            # Merge cluster if it contains multiple fragments
            if len(cluster) >= 2:
                # Create bounding box
                min_x = min(r.left() for r in cluster)
                min_y = min(r.top() for r in cluster)
                max_x = max(r.right() for r in cluster)
                max_y = max(r.bottom() for r in cluster)
                
                merged_rect = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)
                merged_area = merged_rect.width() * merged_rect.height()
                
                # Validate merged rectangle
                if merged_area < page_area * self.config.max_area_pct:
                    merged.append(merged_rect)
                    used.update(cluster_indices)
                    pdebug(f"  Merged {len(cluster)} fragments → {merged_rect.width():.0f}x{merged_rect.height():.0f}pt")
                else:
                    # Too large, keep fragments
                    merged.extend(cluster)
                    used.update(cluster_indices)
            else:
                # Keep single fragment
                merged.append(frag)
                used.add(i)
        
        # Combine with normal panels
        result = normal_panels + merged
        
        if len(merged) < len(fragments):
            pdebug(f"  Fragment reduction: {len(fragments)} → {len(merged)}")
        
        return result
