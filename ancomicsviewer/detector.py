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

            # Prepare grayscale and LAB L channel
            gray = rgba_to_grayscale(arr)
            L = rgba_to_lab_l(arr, apply_clahe=True)

            # Try detection routes in order
            rects = self._try_detection_routes(gray, L, w, h, page_point_size)

            # Post-processing: split by light and filter title rows
            if rects:
                rects = self._split_rects_by_light(L, rects, w, h, page_point_size)
                pdebug(f"After light split -> {len(rects)} rects")

            # Filter by minimum area (post-split cleanup)
            if rects:
                rects = self._filter_by_area(rects, page_point_size)
                pdebug(f"After area filter -> {len(rects)} rects")

            if rects and self.config.filter_title_rows:
                rects = self._filter_title_rows(L, rects, w, h, page_point_size)
                pdebug(f"After title-row filter -> {len(rects)} rects")

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
        """Try detection routes in order until one succeeds."""

        # Calculate scale factor for DPI-adaptive morphology
        # Reference: 1500px width = scale 1.0 (roughly 150 DPI on A4)
        reference_width = 1500.0
        scale_factor = w / reference_width

        # Route 1: Adaptive threshold
        mask = self._adaptive_route(gray, scale_factor)
        rects = self._rects_from_mask(mask, w, h, page_point_size)
        pdebug(f"Adaptive route -> {len(rects)} rects")
        if rects:
            return rects

        # Route 2: LAB L-channel
        mask = self._lab_route(L)
        rects = self._rects_from_mask(mask, w, h, page_point_size)
        pdebug(f"LAB route -> {len(rects)} rects")
        if rects:
            return rects

        # Route 3: Canny (if enabled)
        if self.config.use_canny_fallback:
            pdebug("Trying Canny fallback...")
            mask = self._canny_route(gray)
            rects = self._rects_from_mask(mask, w, h, page_point_size)
            pdebug(f"Canny route -> {len(rects)} rects")

        return rects

    def _adaptive_route(self, gray: NDArray, scale_factor: float = 1.0) -> NDArray:
        """Primary detection route using adaptive threshold.

        Args:
            gray: Grayscale image
            scale_factor: Scale factor for DPI-adaptive morphology (1.0 = 150 DPI reference)
        """
        c = self.config
        k = c.adaptive_block | 1  # Ensure odd
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        th = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            k, c.adaptive_C
        )

        # Adaptive morphology kernel based on image resolution
        if c.morph_scale_with_dpi:
            # Scale kernel size with resolution (smaller kernel for lower res to preserve gutters)
            scaled_kernel = max(3, int(c.morph_kernel * scale_factor)) | 1  # Ensure odd
            scaled_iter = max(1, int(c.morph_iter * scale_factor))
        else:
            scaled_kernel = c.morph_kernel
            scaled_iter = c.morph_iter

        kernel = np.ones((scaled_kernel, scaled_kernel), np.uint8)
        return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=scaled_iter)

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

    def _rects_from_mask(
        self, mask: NDArray, w: int, h: int, page_point_size: QSizeF
    ) -> List[QRectF]:
        """Extract and filter rectangles from binary mask."""
        c = self.config

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        page_area_px = float(w * h)
        scale = w / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0

        # Dynamic minimum size (DPI-stable)
        min_px_dyn = max(c.min_rect_px, int(c.min_rect_frac * min(w, h)))

        rects: List[QRectF] = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)

            # Area filters
            area_px = cw * ch
            area_pct = area_px / page_area_px
            if not (c.min_area_pct <= area_pct <= c.max_area_pct):
                continue

            # Fill ratio filter
            contour_area = float(cv2.contourArea(contour))
            if contour_area <= 0:
                continue
            fill = contour_area / float(area_px)
            if fill < c.min_fill_ratio:
                continue

            # Size filter
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
        """Remove title rows (top, small, bright, multi-box patterns)."""
        c = self.config
        scale = W / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0

        # Convert to pixel coordinates and sort by Y
        items = [
            (int(r.left()*scale), int(r.top()*scale),
             int(r.width()*scale), int(r.height()*scale), r)
            for r in rects_pts
        ]
        items.sort(key=lambda t: t[1])

        # Group by vertical overlap
        rows: List[List[int, int, List]] = []  # [y0, y1, items]
        for x, y, w, h, r in items:
            placed = False
            for row in rows:
                ry0, ry1, lst = row
                if not (y + h < ry0 or y > ry1):  # Overlap
                    row[0] = min(ry0, y)
                    row[1] = max(ry1, y + h)
                    lst.append((x, y, w, h, r))
                    placed = True
                    break
            if not placed:
                rows.append([y, y + h, [(x, y, w, h, r)]])

        keep = []
        page_h = float(H)

        for ry0, ry1, lst in rows:
            y_center_frac = ((ry0 + ry1) / 2.0) / page_h
            median_h_frac = np.median([h for _, _, _, h, _ in lst]) / page_h
            median_w_frac = np.median([w for _, _, w, _, _ in lst]) / float(W)
            count = len(lst)

            # Mean luminosity of row band
            band = L_img[max(0, ry0):min(H, ry1), :]
            meanL = float(band.mean()) / 255.0 if band.size else 0.0

            # Two title patterns
            many_small = (
                count >= c.title_row_min_boxes and
                median_w_frac <= c.title_row_median_w_frac_max
            )
            few_big = (
                count >= c.title_row_big_min_boxes and
                median_w_frac >= c.title_row_big_w_min_frac
            )

            drop = (
                y_center_frac < c.title_row_top_frac and
                median_h_frac < c.title_row_max_h_frac and
                meanL >= c.title_row_min_meanL and
                (many_small or few_big)
            )

            if c.debug:
                pdebug(
                    f"[title-row] y={y_center_frac:.2f} h={median_h_frac:.3f} "
                    f"n={count} medW={median_w_frac:.2f} L={meanL:.2f} -> "
                    f"{'DROP' if drop else 'KEEP'}"
                )

            if not drop:
                keep.extend(item[4] for item in lst)

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
