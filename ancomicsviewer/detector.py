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
        """Try detection routes: adaptive first, then gutter-based, then hybrid."""

        # Primary route: adaptive threshold (works for Tintin-style with black borders)
        mask = self._adaptive_route(gray)
        rects = self._rects_from_mask(mask, w, h, page_point_size)
        pdebug(f"Adaptive route -> {len(rects)} rects")
        
        # If adaptive found nothing or too few, try gutter-based detection
        # (works for color-transition layouts like Grémillet)
        if len(rects) < 3:
            pdebug("Adaptive found few panels, trying gutter-based detection...")
            gutter_rects = self._gutter_based_detection(gray, L, w, h, page_point_size)
            pdebug(f"Gutter-based route -> {len(gutter_rects)} rects")
            if len(gutter_rects) > len(rects):
                rects = gutter_rects
        elif 3 <= len(rects) < 6:
            # Moderate success - try to complement with gutter-based
            # This helps pages like Grémillet p6 and p8
            gutter_rects = self._gutter_based_detection(gray, L, w, h, page_point_size)
            pdebug(f"Gutter-based route (hybrid) -> {len(gutter_rects)} rects")
            
            # Merge and deduplicate
            if len(gutter_rects) > len(rects):
                # Replace with gutter-based if significantly better
                if len(gutter_rects) - len(rects) >= 2:
                    rects = gutter_rects
                else:
                    # Merge both, remove duplicates
                    rects = self._merge_overlapping_rects(rects + gutter_rects)
        
        return rects

    def _merge_overlapping_rects(self, rects: List[QRectF]) -> List[QRectF]:
        """Merge rectangles that overlap significantly."""
        if not rects:
            return []
        
        # Sort by area descending
        rects_sorted = sorted(rects, key=lambda r: r.width() * r.height(), reverse=True)
        
        merged = []
        used = set()
        
        for i, rect in enumerate(rects_sorted):
            if i in used:
                continue
            
            merged.append(rect)
            
            # Check for overlaps with remaining rects
            for j in range(i + 1, len(rects_sorted)):
                if j in used:
                    continue
                
                other = rects_sorted[j]
                
                # Calculate overlap ratio
                inter_rect = rect.intersected(other)
                if not inter_rect.isEmpty():
                    overlap = inter_rect.width() * inter_rect.height()
                    area_min = min(rect.width() * rect.height(), other.width() * other.height())
                    overlap_ratio = overlap / area_min if area_min > 0 else 0
                    
                    # If significant overlap (>40%), consider it a duplicate
                    if overlap_ratio > 0.4:
                        used.add(j)
        
        return merged

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
        """Detect panels by finding gutters (white/light separations between panels).
        
        Better for layouts where panels are separated by color transitions, not black lines.
        Analyzes horizontal and vertical luminosity projections to find separation lines.
        """
        c = self.config
        
        # Use luminosity to find bright separations (gutters)
        # In color layouts, gutters are often white or very light
        _, binary = cv2.threshold(L, 200, 255, cv2.THRESH_BINARY)
        
        # Find horizontal and vertical projections of white pixels
        h_proj = np.sum(binary, axis=1)  # Sum across columns (horizontal lines)
        v_proj = np.sum(binary, axis=0)  # Sum across rows (vertical lines)
        
        # Normalize projections
        h_proj_norm = h_proj / w if w > 0 else h_proj
        v_proj_norm = v_proj / h if h > 0 else v_proj
        
        # Find valleys (gutters) - significant dips in luminosity
        # Use more aggressive thresholding
        from scipy import signal
        try:
            h_smooth = signal.savgol_filter(h_proj_norm, window_length=min(51, len(h_proj_norm)//2 | 1), polyorder=3)
            v_smooth = signal.savgol_filter(v_proj_norm, window_length=min(51, len(v_proj_norm)//2 | 1), polyorder=3)
        except:
            h_smooth = h_proj_norm
            v_smooth = v_proj_norm
        
        # Find extrema (peaks of white space)
        from scipy.signal import find_peaks
        try:
            # Look for peaks in the smoothed projections (high white pixel count = gutters)
            # Use high percentile to find major gutters (85 gives good balance)
            h_threshold = np.percentile(h_smooth, 85)
            v_threshold = np.percentile(v_smooth, 85)
            h_peaks, _ = find_peaks(h_smooth, height=h_threshold, distance=40, prominence=0.015)
            v_peaks, _ = find_peaks(v_smooth, height=v_threshold, distance=40, prominence=0.015)
        except:
            h_peaks = np.where(h_smooth > np.percentile(h_smooth, 85))[0]
            v_peaks = np.where(v_smooth > np.percentile(v_smooth, 85))[0]
        
        # Limit to max 10 horizontal and 10 vertical gutters
        if len(h_peaks) > 10:
            peak_heights = h_smooth[h_peaks]
            top_indices = np.argsort(peak_heights)[-10:]
            h_peaks = h_peaks[np.sort(top_indices)]
        if len(v_peaks) > 10:
            peak_heights = v_smooth[v_peaks]
            top_indices = np.argsort(peak_heights)[-10:]
            v_peaks = v_peaks[np.sort(top_indices)]
            v_peaks = np.where(v_smooth > np.percentile(v_smooth, 60))[0]
        
        # Group peaks into gutter regions
        h_lines = self._group_gutter_lines(h_peaks)
        v_lines = self._group_gutter_lines(v_peaks)
        
        pdebug(f"Found {len(h_lines)} horizontal gutters, {len(v_lines)} vertical gutters")
        
        # Generate panels from gutter intersections
        if h_lines and v_lines:
            rects = self._panels_from_gutters(h_lines, v_lines, w, h, page_point_size)
        else:
            rects = []

        # If very bright page and too few panels, run a contrast-focused local pass
        mean_L = float(np.mean(L)) if L is not None else 0.0
        if len(rects) < 6 and mean_L > 180.0:
            pdebug(f"Bright page (L_mean={mean_L:.1f}) with few gutters -> contrast pass")
            contrast_mask = self._contrast_mask(L)
            contrast_rects = self._rects_from_mask(contrast_mask, w, h, page_point_size)
            pdebug(f"Contrast mask generated {len(contrast_rects)} rects from mask")
            if contrast_rects:
                rects = self._merge_overlapping_rects(rects + contrast_rects)
                pdebug(f"After merge: {len(rects)} rects")

            # Targeted bottom-band scan to recover a missing last-row panel
            band_rect = self._targeted_bottom_band(L, rects, w, h, page_point_size)
            if band_rect is not None:
                rects = self._merge_overlapping_rects(rects + [band_rect])
                pdebug(f"Bottom-band scan recovered 1 rect -> total {len(rects)} rects")
            else:
                pdebug("Bottom-band scan found no new panel")
        
        return rects

    def _contrast_mask(self, L: NDArray) -> NDArray:
        """Enhance contrast and binarize for bright pages without dark borders."""
        # CLAHE to boost local contrast - more aggressive for bright pages
        clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
        L_eq = clahe.apply(L)

        # Adaptive threshold: find darker regions (panels) against bright background
        # Using THRESH_BINARY (not INV) so panels become white in mask
        th = cv2.adaptiveThreshold(
            L_eq,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,  # NOT inverted - panels are darker
            51,
            -8,  # More aggressive (was -5)
        )

        # Slightly larger close to reconnect faint borders
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _targeted_bottom_band(
        self, L: NDArray, rects: List[QRectF], w: int, h: int, page_point_size: QSizeF
    ) -> Optional[QRectF]:
        """Scan the lower band to recover a missing bottom-row panel on bright pages."""
        # Trigger only if very few panels (likely missing one) and page is tall enough
        if h < 500 or len(rects) >= 4:
            if self.config.debug:
                pdebug(f"[band] skip: h={h} rects={len(rects)}")
            return None

        scale_x = w / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0
        scale_y = h / float(page_point_size.height()) if page_point_size.height() > 0 else 1.0

        # Find the lowest point of existing rects
        # Scan from there down, not from a fixed percentage
        if rects:
            max_y = max(int(r.bottom() * scale_y) for r in rects)
        else:
            max_y = int(0.4 * h)  # Default to middle if no rects

        # Use a band starting ~20px below the last rect
        band_start = min(max_y + 20, int(0.7 * h))
        band = L[band_start:, :]

        if self.config.debug:
            pdebug(f"[band] max_y from rects={max_y}, band_start={band_start}")

        # Very permissive band coverage check: only skip if nearly completely covered
        covered = 0.0
        band_area = w * (h - band_start)
        for r in rects:
            rx0 = int(r.left() * scale_x)
            ry0 = int(r.top() * scale_y)
            rx1 = int((r.left() + r.width()) * scale_x)
            ry1 = int((r.top() + r.height()) * scale_y)
            inter_top = max(band_start, ry0)
            inter_bottom = min(h, ry1)
            if inter_bottom > inter_top:
                inter_h = inter_bottom - inter_top
                inter_w = max(0, rx1 - rx0)
                inter_area = inter_h * inter_w
                covered += inter_area
        coverage = covered / band_area if band_area > 0 else 0.0
        if self.config.debug:
            pdebug(f"[band] coverage={coverage:.2f} band_area={band_area}")
        # Only skip if COMPLETELY covered
        if coverage > 0.95:
            if self.config.debug:
                pdebug(f"[band] skip: band completely covered")
            return None

        # Enhance contrast locally with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        band_eq = clahe.apply(band)

        # Adaptive threshold tuned for light-on-white panel edges
        th = cv2.adaptiveThreshold(
            band_eq,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            -5,
        )
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if self.config.debug:
            pdebug(f"[band] found {len(contours)} contours after threshold")
        if not contours:
            return None

        # Keep candidates that are NOT the entire band (size filter)
        page_area = float(w * h)
        band_w = w
        band_h = h - band_start
        max_area_for_panel = 0.35 * page_area  # Don't accept entire band
        candidates = []
        for cnt in contours:
            # Robust rotated bounding box, clamped to band bounds
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            xs, ys = box[:, 0], box[:, 1]
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())

            # Clamp to band image bounds
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(band_w, x1)
            y1 = min(band_h, y1)
            cw = max(1, x1 - x0)
            ch = max(1, y1 - y0)
            area = cw * ch

            # Check for entire-band pattern more carefully
            is_full_width = cw >= band_w * 0.90
            is_large_height = ch >= band_h * 0.80  # Slightly relaxed
            if is_full_width and is_large_height:
                if self.config.debug:
                    pdebug(f"[band] skip contour (entire band): {cw}x{ch} vs band {band_w}x{band_h}")
                continue
            # Be more permissive on area, but still reasonable
            if area < 0.008 * page_area or area > max_area_for_panel:
                continue
            # Convert y back to full image coordinates
            y_full = y0 + band_start
            candidates.append((area, x0, y_full, cw, ch))

        if self.config.debug:
            pdebug(f"[band] candidates after filter: {len(candidates)}")
        if not candidates:
            return None

        # Select the largest candidate
        candidates.sort(reverse=True)
        top_area, x, y_full, cw, ch = candidates[0]
        if self.config.debug:
            pdebug(f"[band] selected: area={top_area/page_area:.3f} pos=({x},{y_full}) size={cw}x{ch}")

        # Convert to page points
        rect = QRectF(x / scale_x, y_full / scale_y, cw / scale_x, ch / scale_y)

        # Reject if heavily overlapping an existing rect (duplicate)
        for r in rects:
            inter = r.intersected(rect)
            if not inter.isEmpty():
                overlap = inter.width() * inter.height()
                min_area = min(r.width() * r.height(), rect.width() * rect.height())
                if min_area > 0 and overlap / min_area > 0.3:  # More permissive
                    if self.config.debug:
                        pdebug(f"[band] rejected: overlap={overlap/min_area:.2f} with existing rect")
                    return None

        if self.config.debug:
            pdebug(f"[band] returning candidate rect")
        return rect

    def _group_gutter_lines(self, gutter_pixels: NDArray) -> List[Tuple[int, int]]:
        """Group consecutive gutter pixels into regions."""
        if len(gutter_pixels) == 0:
            return []
        
        lines = []
        start = gutter_pixels[0]
        prev = start
        
        for pixel in gutter_pixels[1:]:
            if pixel > prev + 1:
                # Gap found, save region
                lines.append((start, prev))
                start = pixel
            prev = pixel
        
        lines.append((start, prev))
        return lines

    def _panels_from_gutters(
        self, 
        h_lines: List[Tuple[int, int]], 
        v_lines: List[Tuple[int, int]], 
        w: int, 
        h: int, 
        page_point_size: QSizeF
    ) -> List[QRectF]:
        """Generate panel rectangles from gutter positions."""
        rects = []
        scale = w / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0
        
        # Add page boundaries as "gutters"
        h_boundaries = [(0, 0)] + h_lines + [(h-1, h-1)]
        v_boundaries = [(0, 0)] + v_lines + [(w-1, w-1)]
        
        # Extract panels from gutter grid
        # Relaxed minimum to allow smaller panels (e.g., third column on Grémillet p6)
        # Use 8% instead of 12% to catch more panels while staying reasonable
        min_panel_w = max(self.config.min_rect_px, int(0.08 * w))
        min_panel_h = max(self.config.min_rect_px, int(0.08 * h))
        
        if self.config.debug:
            pdebug(f"[gutters] h_boundaries={len(h_boundaries)} v_boundaries={len(v_boundaries)}")
            pdebug(f"[gutters] min_panel_w={min_panel_w} min_panel_h={min_panel_h}")
        
        for i in range(len(h_boundaries) - 1):
            y_start = h_boundaries[i][1] + 1
            y_end = h_boundaries[i+1][0]
            h_cell = y_end - y_start
            if h_cell < min_panel_h:
                if self.config.debug and i < 3:
                    pdebug(f"[gutters] skip h-cell {i}: height {h_cell} < {min_panel_h}")
                continue
                
            for j in range(len(v_boundaries) - 1):
                x_start = v_boundaries[j][1] + 1
                x_end = v_boundaries[j+1][0]
                w_cell = x_end - x_start
                if w_cell < min_panel_w:
                    if self.config.debug and j < 3:
                        pdebug(f"[gutters] skip v-cell {j}: width {w_cell} < {min_panel_w}")
                    continue
                
                # Create panel rectangle
                rect_w = x_end - x_start
                rect_h = y_end - y_start
                
                # Convert to page point coordinates
                x_pts = x_start / scale
                y_pts = y_start / scale
                w_pts = rect_w / scale
                h_pts = rect_h / scale
                
                rects.append(QRectF(x_pts, y_pts, w_pts, h_pts))
        
        if self.config.debug:
            pdebug(f"[gutters] generated {len(rects)} panels from gutter grid")
        
        return rects

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
            # Rotated bbox then clamp to image bounds to avoid negative/overflow
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            xs, ys = box[:, 0], box[:, 1]
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())

            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(w, x1)
            y1 = min(h, y1)

            cw = max(1, x1 - x0)
            ch = max(1, y1 - y0)

            # Area filter (relaxed)
            area_px = cw * ch
            area_pct = area_px / page_area_px
            if area_pct < min_area_initial or area_pct > c.max_area_pct:
                continue

            # Fill ratio filter (relaxed for bright pages)
            contour_area = float(cv2.contourArea(contour))
            if contour_area <= 0:
                continue
            fill = contour_area / float(area_px)
            if fill < (c.min_fill_ratio * 0.8):  # 20% more permissive (standard)
                continue

            # Size filter (relaxed)
            if cw < min_px_dyn or ch < min_px_dyn:
                continue

            # Convert to page points
            px = x0 / scale
            py = y0 / scale
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
