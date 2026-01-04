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
class PanelRegion:
    """Represents a detected panel region with multiple representations.
    
    Used in freeform detection for complex layouts (parallelograms, tinted backgrounds, etc.)
    """
    contour: NDArray          # Original contour (Nx1x2)
    poly: NDArray             # Simplified polygon approximation
    bbox: Tuple[int, int, int, int]  # Axis-aligned bounding box (x, y, w, h)
    obb: NDArray              # Oriented bounding box (4 points from minAreaRect)
    area: float               # Contour area
    fill_ratio: float         # area / (bbox_w * bbox_h)
    touches_border: bool      # Whether region touches image border
    centroid: Tuple[float, float]  # (cx, cy)
    
    def to_qrectf(self, scale: float) -> QRectF:
        """Convert bbox to QRectF in page points."""
        x, y, w, h = self.bbox
        return QRectF(x / scale, y / scale, w / scale, h / scale)


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
            
            # Convert to BGR for Lab-based filtering
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            
            # Estimate background color in Lab space
            bg_lab = _estimate_bg_lab(img_bgr, border_pct=0.04)
            pdebug(f"[Panels] bg_lab=({bg_lab[0]:.1f}, {bg_lab[1]:.1f}, {bg_lab[2]:.1f})")

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
            
            # Filter out empty rectangles (no content)
            if rects:
                rects = self._filter_empty_rects(rects, gray, w, h, page_point_size)
                pdebug(f"After empty filter -> {len(rects)} rects")
            
            # Lab-based empty filtering (for tinted/watercolor pages)
            if rects:
                initial_count = len(rects)
                filtered_rects = []
                
                # Calculate median dimensions for thin panel detection
                if len(rects) > 1:
                    widths = [r.width() for r in rects]
                    heights = [r.height() for r in rects]
                    median_w = np.median(widths)
                    median_h = np.median(heights)
                else:
                    median_w = median_h = 0
                
                for rect in rects:
                    # Check non-background ratio
                    x = int(rect.left() * (w / page_point_size.width()))
                    y = int(rect.top() * (h / page_point_size.height()))
                    rw = int(rect.width() * (w / page_point_size.width()))
                    rh = int(rect.height() * (h / page_point_size.height()))
                    
                    # Clamp to bounds
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    rw = max(1, min(rw, w - x))
                    rh = max(1, min(rh, h - y))
                    
                    roi_bgr = img_bgr[y:y+rh, x:x+rw]
                    
                    if roi_bgr.size == 0:
                        continue
                    
                    non_bg = _non_bg_ratio(roi_bgr, bg_lab, delta=self.config.bg_delta)
                    
                    # Filter if too empty (mostly background)
                    if non_bg < self.config.min_non_bg_ratio:
                        pdebug(f"[Lab] Dropped mostly-bg panel at ({rect.left():.0f},{rect.top():.0f}): "
                              f"non_bg={non_bg:.3f} < {self.config.min_non_bg_ratio:.3f}")
                        continue
                    
                    # Filter if too thin (likely gutters misdetected as panels)
                    if median_w > 0 and median_h > 0:
                        if (rect.height() < self.config.min_dim_ratio * median_h or
                            rect.width() < self.config.min_dim_ratio * median_w):
                            pdebug(f"[Lab] Dropped thin panel at ({rect.left():.0f},{rect.top():.0f}): "
                                  f"{rect.width():.0f}x{rect.height():.0f}pt vs median "
                                  f"{median_w:.0f}x{median_h:.0f}pt (min_ratio={self.config.min_dim_ratio:.2f})")
                            continue
                    
                    filtered_rects.append(rect)
                
                rects = filtered_rects
                if len(rects) < initial_count:
                    pdebug(f"[Lab] Lab-based filter: {initial_count} -> {len(rects)} panels")
            
            # Suppress nested rectangles (small rects inside larger ones)
            if rects:
                # Save debug image before nested suppression if debug mode
                if self.config.debug:
                    self._save_debug_panels(img_bgr, rects, w, h, page_point_size, "before")
                
                rects = self._suppress_nested_rects(rects, img_bgr, bg_lab, w, h, page_point_size,
                                                    delta=self.config.bg_delta)
                pdebug(f"After nested suppression -> {len(rects)} rects")
                
                # Save debug image after nested suppression if debug mode
                if self.config.debug:
                    self._save_debug_panels(img_bgr, rects, w, h, page_point_size, "after")

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
        """Try detection routes: adaptive first, then gutter-based, then hybrid, finally freeform fallback."""

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
            # This helps pages like Grémillet p6 and p8, but be careful
            # not to replace good adaptive rects with gutter grid that has many empty cells
            gutter_rects = self._gutter_based_detection(gray, L, w, h, page_point_size)
            pdebug(f"Gutter-based route (hybrid) -> {len(gutter_rects)} rects")
            
            # Only use gutter-based if it finds significantly more meaningful panels
            # Check if gutter grid would produce a large number of small/empty cells
            if len(gutter_rects) >= 10 and len(gutter_rects) > len(rects) * 3:
                # Too many rects for a few content areas - likely creating grid with many empty cells
                pdebug(f"[Gutter] Too many rects ({len(gutter_rects)}) for {len(rects)} adaptive panels - likely marginal cells, skipping")
            elif len(gutter_rects) > len(rects):
                # Use gutter-based if it found more meaningful panels
                if len(gutter_rects) - len(rects) >= 2:
                    rects = gutter_rects
                else:
                    # Merge both, remove duplicates
                    rects = self._merge_overlapping_rects(rects + gutter_rects)
        
        # Check if freeform fallback is needed
        if self.config.use_freeform_fallback:
            page_area = page_point_size.width() * page_point_size.height()
            needs_freeform = False
            reason = ""
            
            if len(rects) < 2:
                needs_freeform = True
                reason = "too few panels detected"
            elif len(rects) > 0:
                max_rect_area = max(r.width() * r.height() for r in rects)
                max_ratio = max_rect_area / page_area
                if max_ratio > 0.50:  # Lowered from 0.60 to catch more cases
                    needs_freeform = True
                    reason = f"single large panel covering {max_ratio*100:.1f}% of page"
            
            if needs_freeform:
                pdebug(f"[Freeform] Triggering fallback: {reason}")
                
                # Convert grayscale to BGR for freeform processing
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                freeform_rects = self._freeform_detection(gray_bgr, w, h, page_point_size)
                
                if len(freeform_rects) > len(rects):
                    pdebug(f"[Freeform] Replacing {len(rects)} rects with {len(freeform_rects)} freeform rects")
                    rects = freeform_rects
                else:
                    pdebug(f"[Freeform] Keeping original {len(rects)} rects (freeform found {len(freeform_rects)})")
        
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

    def _freeform_detection(self, img_bgr: NDArray, w: int, h: int, 
                           page_point_size: QSizeF) -> List[QRectF]:
        """Freeform panel detection using watershed segmentation.
        
        Works on complex layouts with parallelograms, tinted backgrounds, etc.
        
        Args:
            img_bgr: Image in BGR format
            w, h: Image dimensions
            page_point_size: Page dimensions in points
            
        Returns:
            List of panel rectangles
        """
        # 1. Estimate background color
        bg_lab = estimate_background_color_lab(img_bgr)
        
        # 2. Create background mask
        debug_paths = {}
        if self.config.debug:
            import os
            debug_dir = "debug_output"
            os.makedirs(debug_dir, exist_ok=True)
            debug_paths = {
                'bg_mask': os.path.join(debug_dir, 'freeform_bg_mask.png'),
                'mask_fg': os.path.join(debug_dir, 'freeform_mask_fg.png'),
                'sure_fg': os.path.join(debug_dir, 'freeform_sure_fg.png'),
                'markers': os.path.join(debug_dir, 'freeform_markers.png'),
            }
        
        mask_bg = make_background_mask(
            img_bgr, bg_lab, 
            delta=self.config.bg_delta,
            debug_path=debug_paths.get('bg_mask')
        )
        
        # 3. Watershed segmentation
        markers = segment_panels_watershed(
            img_bgr, mask_bg,
            sure_fg_ratio=self.config.sure_fg_ratio,
            debug_paths={k: v for k, v in debug_paths.items() if k != 'bg_mask'}
        )
        
        # 4. Extract regions
        regions = extract_panel_regions(
            markers, (h, w),
            img_bgr, mask_bg,
            min_area_ratio=self.config.min_area_ratio_freeform,
            max_area_ratio=self.config.max_area_pct,
            min_fill_ratio=self.config.min_fill_ratio_freeform,
            min_content_ratio=0.05,  # At least 5% of region must be non-background
            approx_eps_ratio=self.config.approx_eps_ratio
        )
        
        pdebug(f"[Freeform] Extracted {len(regions)} regions before merge")
        
        # 5. Merge overlapping regions
        regions = merge_overlapping_regions(regions, iou_thr=self.config.iou_merge_thr)
        
        pdebug(f"[Freeform] After merge: {len(regions)} regions")
        
        # 6. Sort in reading order
        regions = sort_reading_order(regions, rtl=self.config.reading_rtl)
        
        # 7. Convert to QRectF
        scale = (w / page_point_size.width())  # pixels per point
        rects = [region.to_qrectf(scale) for region in regions]
        
        # 8. Save debug visualization if enabled
        if self.config.debug and debug_paths:
            self._save_freeform_debug(img_bgr, regions, debug_paths)
        
        return rects
    
    def _save_freeform_debug(self, img_bgr: NDArray, regions: List[PanelRegion], 
                            debug_paths: dict) -> None:
        """Save debug visualization of detected regions."""
        import os
        debug_dir = "debug_output"
        
        # Draw contours, bboxes, and OBBs
        debug_img = img_bgr.copy()
        
        for i, region in enumerate(regions):
            # Draw contour in green
            cv2.drawContours(debug_img, [region.contour], -1, (0, 255, 0), 2)
            
            # Draw bbox in blue
            x, y, bw, bh = region.bbox
            cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            
            # Draw OBB in red
            cv2.drawContours(debug_img, [region.obb], -1, (0, 0, 255), 2)
            
            # Draw index
            cx, cy = region.centroid
            cv2.putText(debug_img, str(i + 1), (int(cx) - 10, int(cy) + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        out_path = os.path.join(debug_dir, 'freeform_regions_contours.png')
        cv2.imwrite(out_path, debug_img)
        pdebug(f"[Freeform] Saved regions visualization: {out_path}")


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

    def _make_gutter_mask(self, gray: NDArray, L: NDArray) -> NDArray:
        """Build a robust gutter mask using brightness + gradient uniformity.
        
        Identifies gutters as pixels that are BOTH very bright AND locally uniform
        (low gradient). This avoids picking up bright regions inside panels.
        
        Args:
            gray: Grayscale image
            L: LAB L-channel
            
        Returns:
            uint8 gutter mask (0=not gutter, 255=gutter)
        """
        h, w = L.shape
        
        # Adaptive brightness threshold: use high percentile (gutters are very bright)
        # Default: 94th percentile of L values
        L_percentile = getattr(self.config, 'gutter_bright_percentile', 94)
        L_high = np.percentile(L, L_percentile)
        pdebug(f"[gutter_mask] L_high (p{L_percentile})={L_high:.1f}")
        
        # Compute gradient magnitude (Sobel)
        # Gutters are uniform (low gradient), highlights/text have high gradient
        grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.abs(grad_x) + np.abs(grad_y)
        
        # Adaptive gradient threshold: use low percentile (gutters have low edges)
        # Default: 50th percentile (median) or lower
        grad_percentile = getattr(self.config, 'gutter_grad_percentile', 50)
        grad_low = np.percentile(grad_mag, grad_percentile)
        pdebug(f"[gutter_mask] grad_low (p{grad_percentile})={grad_low:.1f}")
        
        # Combine: both bright AND uniform
        bright_mask = (L >= L_high).astype(np.uint8) * 255
        uniform_mask = (grad_mag <= grad_low).astype(np.uint8) * 255
        gutter_mask = cv2.bitwise_and(bright_mask, uniform_mask)
        
        # Morphological closing to connect gutter regions
        # Use elongated kernels to preserve gutter shape
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
        
        gutter_h = cv2.morphologyEx(gutter_mask, cv2.MORPH_CLOSE, h_kernel, iterations=1)
        gutter_v = cv2.morphologyEx(gutter_mask, cv2.MORPH_CLOSE, v_kernel, iterations=1)
        gutter_mask = cv2.bitwise_or(gutter_h, gutter_v)
        
        pdebug(f"[gutter_mask] Created gutter mask from brightness + gradient")
        return gutter_mask

    def _validate_gutter_lines(self, gutter_mask: NDArray, 
                               h_lines: List[Tuple[int, int]], 
                               v_lines: List[Tuple[int, int]]) -> tuple:
        """Validate gutter candidates based on coverage and thickness.
        
        Keep only gutters that have high coverage across their extent and
        sufficient thickness.
        
        Args:
            gutter_mask: Binary gutter mask (255=gutter)
            h_lines: List of (y_start, y_end) horizontal gutter candidates
            v_lines: List of (x_start, x_end) vertical gutter candidates
            
        Returns:
            (validated_h_lines, validated_v_lines)
        """
        c = self.config
        min_coverage = c.gutter_cov_min  # Default: 0.85
        min_thickness = c.min_gutter_px  # Default: 5 pixels
        
        h, w = gutter_mask.shape
        validated_h = []
        validated_v = []
        
        # Validate horizontal gutters
        for y_start, y_end in h_lines:
            thickness = y_end - y_start + 1
            
            if thickness < min_thickness:
                pdebug(f"[validate_gutters] Rejected h-gutter y={y_start}..{y_end}: thickness={thickness} < {min_thickness}")
                continue
            
            # Extract band and compute coverage
            band = gutter_mask[y_start:y_end+1, :]
            if band.size == 0:
                continue
            
            coverage = np.mean(band == 255)
            
            if coverage >= min_coverage:
                validated_h.append((y_start, y_end))
                pdebug(f"[validate_gutters] Kept h-gutter y={y_start}..{y_end}: coverage={coverage:.3f}")
            else:
                pdebug(f"[validate_gutters] Rejected h-gutter y={y_start}..{y_end}: coverage={coverage:.3f} < {min_coverage}")
        
        # Validate vertical gutters
        for x_start, x_end in v_lines:
            thickness = x_end - x_start + 1
            
            if thickness < min_thickness:
                pdebug(f"[validate_gutters] Rejected v-gutter x={x_start}..{x_end}: thickness={thickness} < {min_thickness}")
                continue
            
            # Extract band and compute coverage
            band = gutter_mask[:, x_start:x_end+1]
            if band.size == 0:
                continue
            
            coverage = np.mean(band == 255)
            
            if coverage >= min_coverage:
                validated_v.append((x_start, x_end))
                pdebug(f"[validate_gutters] Kept v-gutter x={x_start}..{x_end}: coverage={coverage:.3f}")
            else:
                pdebug(f"[validate_gutters] Rejected v-gutter x={x_start}..{x_end}: coverage={coverage:.3f} < {min_coverage}")
        
        return validated_h, validated_v

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
        Uses adaptive gutter mask (brightness + low gradient) for robustness.
        """
        c = self.config
        
        # Build robust gutter mask: bright + uniform (low gradient)
        gutter_mask = self._make_gutter_mask(gray, L)
        
        # Save debug image of gutter mask if debug mode
        if self.config.debug:
            self._save_debug_image(gutter_mask, "dbg_gutter_mask.png")
        
        # Find horizontal and vertical projections of gutter pixels
        h_proj = np.sum(gutter_mask, axis=1)  # Sum across columns (horizontal lines)
        v_proj = np.sum(gutter_mask, axis=0)  # Sum across rows (vertical lines)
        
        # Normalize projections
        h_proj_norm = h_proj / w if w > 0 else h_proj
        v_proj_norm = v_proj / h if h > 0 else v_proj
        
        # Smooth projections to find peaks
        from scipy import signal
        try:
            h_smooth = signal.savgol_filter(h_proj_norm, window_length=min(51, len(h_proj_norm)//2 | 1), polyorder=3)
            v_smooth = signal.savgol_filter(v_proj_norm, window_length=min(51, len(v_proj_norm)//2 | 1), polyorder=3)
        except:
            h_smooth = h_proj_norm
            v_smooth = v_proj_norm
        
        # Find peaks in projections (gutter positions)
        from scipy.signal import find_peaks
        
        # Scale distance and prominence based on image dimensions
        h_distance = int(max(30, 0.015 * h))
        v_distance = int(max(30, 0.015 * w))
        prominence = 0.015
        
        try:
            h_threshold = np.percentile(h_smooth, 85)
            v_threshold = np.percentile(v_smooth, 85)
            h_peaks, _ = find_peaks(h_smooth, height=h_threshold, distance=h_distance, prominence=prominence)
            v_peaks, _ = find_peaks(v_smooth, height=v_threshold, distance=v_distance, prominence=prominence)
        except:
            h_peaks = np.where(h_smooth > np.percentile(h_smooth, 85))[0]
            v_peaks = np.where(v_smooth > np.percentile(v_smooth, 85))[0]
        
        pdebug(f"[gutters] h_peaks raw={len(h_peaks)} v_peaks raw={len(v_peaks)}")
        
        # Limit to max 10 horizontal and 10 vertical gutters (keep highest energy)
        if len(h_peaks) > 10:
            peak_heights = h_smooth[h_peaks]
            top_indices = np.argsort(peak_heights)[-10:]
            h_peaks = h_peaks[np.sort(top_indices)]
            pdebug(f"[gutters] h_peaks reduced to {len(h_peaks)} using top-energy selection")
        
        if len(v_peaks) > 10:
            peak_heights = v_smooth[v_peaks]
            top_indices = np.argsort(peak_heights)[-10:]
            v_peaks = v_peaks[np.sort(top_indices)]
            pdebug(f"[gutters] v_peaks reduced to {len(v_peaks)} using top-energy selection")
        
        # Group peaks into gutter regions (start/end pairs)
        h_lines = self._group_gutter_lines(h_peaks)
        v_lines = self._group_gutter_lines(v_peaks)
        
        pdebug(f"[gutters] h_lines raw={len(h_lines)} v_lines raw={len(v_lines)}")
        
        # ✅ Validate gutters: keep only those with high coverage and sufficient thickness
        h_lines, v_lines = self._validate_gutter_lines(gutter_mask, h_lines, v_lines)
        
        pdebug(f"[gutters] h_lines valid={len(h_lines)} v_lines valid={len(v_lines)}")
        pdebug(f"Found {len(h_lines)} horizontal gutters, {len(v_lines)} vertical gutters")
        
        # Save debug image with validated gutter lines if debug mode
        if self.config.debug:
            self._save_debug_gutters(gutter_mask, h_lines, v_lines, w, h, "dbg_hv_gutters.png")
        
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
        """Group consecutive gutter pixels into regions, expanding isolated peaks.
        
        If peaks are sparse (not consecutive), expand them into bands to ensure
        minimum thickness.
        """
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
        
        # Expand thin lines to ensure minimum 3-pixel thickness for detection
        expanded_lines = []
        for y_start, y_end in lines:
            thickness = y_end - y_start + 1
            if thickness < 3:
                # Expand symmetrically
                expand = (3 - thickness + 1) // 2
                y_start = max(0, y_start - expand)
                y_end = min(y_end + expand, 4096)  # Reasonable max
            expanded_lines.append((y_start, y_end))
        
        return expanded_lines

    def _validate_h_gutters(self, binary_bright: NDArray, h_lines: List[Tuple[int, int]], 
                           min_cov: float = 0.85, band: int = 6) -> List[Tuple[int, int]]:
        """Validate horizontal gutter candidates by checking continuity across width.
        
        Args:
            binary_bright: Binary mask of bright pixels (255 = bright, 0 = dark)
            h_lines: List of (y0, y1) horizontal gutter candidates
            min_cov: Minimum coverage ratio (bright pixels / total pixels)
            band: Half-height of the band to check around gutter center
            
        Returns:
            List of validated (y0, y1) tuples
        """
        if not h_lines:
            return []
        
        h, w = binary_bright.shape
        validated = []
        
        for y0, y1 in h_lines:
            y_center = (y0 + y1) // 2
            y_min = max(0, y_center - band)
            y_max = min(h, y_center + band)
            
            # Extract horizontal band
            band_region = binary_bright[y_min:y_max, :]
            
            if band_region.size == 0:
                continue
            
            # Calculate coverage of bright pixels
            bright_pixels = np.sum(band_region == 255)
            total_pixels = band_region.size
            coverage = bright_pixels / total_pixels if total_pixels > 0 else 0
            
            if coverage >= min_cov:
                validated.append((y0, y1))
            else:
                pdebug(f"[gutters] Rejected h-gutter at y={y_center}: coverage={coverage:.2f} < {min_cov}")
        
        return validated

    def _validate_v_gutters(self, binary_bright: NDArray, v_lines: List[Tuple[int, int]], 
                           min_cov: float = 0.85, band: int = 6) -> List[Tuple[int, int]]:
        """Validate vertical gutter candidates by checking continuity across height.
        
        Args:
            binary_bright: Binary mask of bright pixels (255 = bright, 0 = dark)
            v_lines: List of (x0, x1) vertical gutter candidates
            min_cov: Minimum coverage ratio (bright pixels / total pixels)
            band: Half-width of the band to check around gutter center
            
        Returns:
            List of validated (x0, x1) tuples
        """
        if not v_lines:
            return []
        
        h, w = binary_bright.shape
        validated = []
        
        for x0, x1 in v_lines:
            x_center = (x0 + x1) // 2
            x_min = max(0, x_center - band)
            x_max = min(w, x_center + band)
            
            # Extract vertical band
            band_region = binary_bright[:, x_min:x_max]
            
            if band_region.size == 0:
                continue
            
            # Calculate coverage of bright pixels
            bright_pixels = np.sum(band_region == 255)
            total_pixels = band_region.size
            coverage = bright_pixels / total_pixels if total_pixels > 0 else 0
            
            if coverage >= min_cov:
                validated.append((x0, x1))
            else:
                pdebug(f"[gutters] Rejected v-gutter at x={x_center}: coverage={coverage:.2f} < {min_cov}")
        
        return validated

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
        # Use 2.5% instead of 12% to catch more panels while staying reasonable
        min_panel_w = max(self.config.min_rect_px, int(0.025 * w))
        min_panel_h = max(self.config.min_rect_px, int(0.025 * h))
        
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

    def _filter_empty_rects(self, rects: List[QRectF], gray: NDArray, 
                           w: int, h: int, page_point_size: QSizeF,
                           min_content_ratio: float = 0.03) -> List[QRectF]:
        """Filter out rectangles that contain mostly empty background.
        
        Args:
            rects: Panel rectangles in page points
            gray: Grayscale image
            w, h: Image dimensions in pixels
            page_point_size: Page dimensions in points
            min_content_ratio: Minimum ratio of dark pixels (content) required
            
        Returns:
            Filtered list of rectangles with actual content
        """
        if not rects or gray is None:
            return rects
        
        scale = w / page_point_size.width() if page_point_size.width() > 0 else 1.0
        
        # Create binary mask: dark pixels (content) vs bright pixels (background)
        # Threshold at 200 to catch most background/gutter areas
        _, content_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        filtered = []
        for rect in rects:
            # Convert rect to pixel coordinates
            x = int(rect.left() * scale)
            y = int(rect.top() * scale)
            rw = int(rect.width() * scale)
            rh = int(rect.height() * scale)
            
            # Clamp to image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            rw = max(1, min(rw, w - x))
            rh = max(1, min(rh, h - y))
            
            # Extract region from content mask
            region_mask = content_mask[y:y+rh, x:x+rw]
            
            if region_mask.size == 0:
                continue
            
            # Calculate ratio of content pixels (dark = 255 in inverted mask)
            content_pixels = np.sum(region_mask == 255)
            total_pixels = region_mask.size
            content_ratio = content_pixels / total_pixels
            
            if content_ratio >= min_content_ratio:
                filtered.append(rect)
            else:
                pdebug(f"[Filter] Rejected empty rect at ({rect.left():.0f},{rect.top():.0f}) "
                      f"{rect.width():.0f}x{rect.height():.0f}pt: content={content_ratio:.3f} < {min_content_ratio}")
        
        if len(filtered) < len(rects):
            pdebug(f"[Filter] Removed {len(rects) - len(filtered)} empty rectangles")
        
        return filtered

    def _suppress_nested_rects(self, rects: List[QRectF], img_bgr: NDArray, 
                               bg_lab: NDArray, w: int, h: int, page_point_size: QSizeF,
                               delta: float = 12.0,
                               contain_thr: float = 0.90, 
                               area_ratio_thr: float = 0.25,
                               empty_ratio_thr: float = 0.10) -> List[QRectF]:
        """Handle nested rectangles by merging them (taking union of nested panels).
        
        When a small panel is completely inside a larger panel, merge them into one
        taking the reading order of the inner (weaker) panel.
        
        Args:
            rects: Panel rectangles in page points
            img_bgr: Image in BGR format (for content checking)
            bg_lab: Background color in Lab
            w, h: Image dimensions in pixels
            page_point_size: Page dimensions in points
            delta: Lab distance threshold for background detection
            contain_thr: Containment threshold (intersection/small_area)
            area_ratio_thr: Maximum area ratio (small/big) for nested detection
            empty_ratio_thr: Maximum non-bg ratio for considering rect empty
            
        Returns:
            List with nested rectangles merged
        """
        if len(rects) <= 1 or img_bgr is None:
            return rects
        
        scale = w / page_point_size.width() if page_point_size.width() > 0 else 1.0
        
        # Convert BGR to avoid repeated conversions
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        
        merged_indices = {}  # Maps old index to merged rectangle + new index
        to_remove = set()
        
        # Check all pairs
        for i, small in enumerate(rects):
            if i in to_remove:
                continue
            
            small_area = small.width() * small.height()
            
            for j, big in enumerate(rects):
                if i == j or j in to_remove:
                    continue
                
                big_area = big.width() * big.height()
                
                # Skip if not nested (small must be smaller)
                if small_area >= big_area:
                    continue
                
                # Check area ratio
                area_ratio = small_area / big_area if big_area > 0 else 0
                if area_ratio > area_ratio_thr:
                    continue
                
                # Calculate intersection
                inter = small.intersected(big)
                if inter.isEmpty():
                    continue
                
                inter_area = inter.width() * inter.height()
                containment = inter_area / small_area if small_area > 0 else 0
                
                # If highly contained
                if containment >= contain_thr:
                    # Merge nested panels: small panel inside big
                    # Take union to capture all content
                    merged_rect = small.united(big)
                    merged_indices[j] = merged_rect
                    to_remove.add(i)
                    
                    pdebug(f"[Nested] Merged nested rect at ({small.left():.0f},{small.top():.0f}) "
                          f"{small.width():.0f}x{small.height():.0f}pt "
                          f"into ({big.left():.0f},{big.top():.0f}) {big.width():.0f}x{big.height():.0f}pt -> "
                          f"union ({merged_rect.left():.0f},{merged_rect.top():.0f}) "
                          f"{merged_rect.width():.0f}x{merged_rect.height():.0f}pt")
                    break
        
        # Build result: keep non-removed rects, apply merged versions where applicable
        result = []
        for i, r in enumerate(rects):
            if i in to_remove:
                continue
            if i in merged_indices:
                result.append(merged_indices[i])
            else:
                result.append(r)
        
        if len(to_remove) > 0:
            pdebug(f"[Panels] nested-merge: merged {len(to_remove)} rects")
        
        return result

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
    
    def _save_debug_image(self, img: NDArray, filename: str) -> None:
        """Save debug image to debug_output/ directory.
        
        Args:
            img: Image array (grayscale or color)
            filename: Output filename
        """
        import os
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        out_path = os.path.join(debug_dir, filename)
        cv2.imwrite(out_path, img)
        pdebug(f"[Debug] Saved {out_path}")
    
    def _save_debug_gutters(self, gutter_mask: NDArray, h_lines: List[Tuple[int, int]], 
                           v_lines: List[Tuple[int, int]], w: int, h: int, 
                           filename: str) -> None:
        """Save debug visualization of gutter lines overlaid on gutter mask.
        
        Args:
            gutter_mask: Binary gutter mask (255=gutter, 0=background)
            h_lines: List of (start, end) tuples for horizontal gutters
            v_lines: List of (start, end) tuples for vertical gutters
            w, h: Image dimensions
            filename: Output filename
        """
        import os
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Downscale for visualization
        max_dim = 1200
        scale = min(1.0, max_dim / max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Convert mask to BGR for visualization
        mask_vis = cv2.resize(gutter_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        
        # Draw horizontal gutters in red (thick lines)
        for y_start, y_end in h_lines:
            y_start_scaled = int(y_start * scale)
            y_end_scaled = int(y_end * scale)
            y_mid = (y_start_scaled + y_end_scaled) // 2
            cv2.line(img_vis, (0, y_mid), (new_w, y_mid), (0, 0, 255), 3)
        
        # Draw vertical gutters in blue (thick lines)
        for x_start, x_end in v_lines:
            x_start_scaled = int(x_start * scale)
            x_end_scaled = int(x_end * scale)
            x_mid = (x_start_scaled + x_end_scaled) // 2
            cv2.line(img_vis, (x_mid, 0), (x_mid, new_h), (255, 0, 0), 3)
        
        out_path = os.path.join(debug_dir, filename)
        cv2.imwrite(out_path, img_vis)
        pdebug(f"[Debug] Saved {out_path}")
    
    def _save_debug_panels(self, img_bgr: NDArray, rects: List[QRectF], 
                          w: int, h: int, page_point_size: QSizeF, 
                          stage: str) -> None:
        """Save debug visualization of panel rectangles.
        
        Args:
            img_bgr: Original BGR image
            rects: Panel rectangles in page points
            w, h: Image dimensions in pixels
            page_point_size: Page dimensions in points
            stage: "before" or "after" (nested suppression)
        """
        import os
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Downscale for visualization
        max_dim = 1200
        scale = min(1.0, max_dim / max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        img_vis = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Scale factor from page points to pixels
        pt_to_px = (w / page_point_size.width()) if page_point_size.width() > 0 else 1.0
        
        # Draw rectangles
        for i, rect in enumerate(rects):
            x = int(rect.left() * pt_to_px * scale)
            y = int(rect.top() * pt_to_px * scale)
            rw = int(rect.width() * pt_to_px * scale)
            rh = int(rect.height() * pt_to_px * scale)
            
            # Draw rectangle in green
            cv2.rectangle(img_vis, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
            
            # Draw panel number
            cv2.putText(img_vis, str(i + 1), (x + 5, y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        filename = f"dbg_panels_{stage}.png"
        out_path = os.path.join(debug_dir, filename)
        cv2.imwrite(out_path, img_vis)
        pdebug(f"[Debug] Saved {out_path}")


# ========== Freeform Detection Functions (for complex layouts) ==========

def _estimate_bg_lab(img_bgr: NDArray, border_pct: float = 0.04) -> NDArray:
    """Estimate background color from image borders in Lab color space.
    
    Args:
        img_bgr: Input image in BGR format
        border_pct: Percentage of image dimensions to use for border sampling
        
    Returns:
        Array of shape (3,) containing (L, a, b) median values
    """
    h, w = img_bgr.shape[:2]
    border_size = max(int(min(h, w) * border_pct), 5)
    
    # Sample borders: top, bottom, left, right
    samples = []
    samples.append(img_bgr[0:border_size, :])  # Top
    samples.append(img_bgr[h-border_size:h, :])  # Bottom
    samples.append(img_bgr[:, 0:border_size])  # Left
    samples.append(img_bgr[:, w-border_size:w])  # Right
    
    # Concatenate all border pixels
    border_pixels = np.vstack([s.reshape(-1, 3) for s in samples])
    
    # Convert to Lab
    border_lab = cv2.cvtColor(border_pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2Lab)
    border_lab = border_lab.reshape(-1, 3).astype(np.float32)
    
    # Compute median (robust to outliers)
    bg_lab = np.median(border_lab, axis=0)
    
    return bg_lab


def _non_bg_ratio(img_bgr_roi: NDArray, bg_lab: NDArray, delta: float = 12.0) -> float:
    """Calculate ratio of non-background pixels in a ROI.
    
    Args:
        img_bgr_roi: ROI in BGR format
        bg_lab: Background color in Lab (shape (3,))
        delta: Maximum Lab distance to be considered background
        
    Returns:
        Ratio of pixels that are NOT background (0.0 to 1.0)
    """
    if img_bgr_roi.size == 0:
        return 0.0
    
    # Convert ROI to Lab
    roi_lab = cv2.cvtColor(img_bgr_roi, cv2.COLOR_BGR2Lab).astype(np.float32)
    
    # Compute Euclidean distance to background color
    dist = np.linalg.norm(roi_lab - bg_lab, axis=2)
    
    # Non-background pixels are those far from bg color
    non_bg = dist > delta
    
    # Return ratio
    ratio = np.mean(non_bg)
    
    return float(ratio)


def estimate_background_color_lab(img_bgr: NDArray, sample_pct: float = 0.03) -> Tuple[float, float, float]:
    """Estimate background color by sampling image borders in Lab color space.
    
    Args:
        img_bgr: Input image in BGR format
        sample_pct: Percentage of image dimensions to use for border sampling
        
    Returns:
        Tuple of (L, a, b) median values
    """
    h, w = img_bgr.shape[:2]
    border_size = max(int(min(h, w) * sample_pct), 5)
    
    # Sample borders: top, bottom, left, right
    samples = []
    samples.append(img_bgr[0:border_size, :])  # Top
    samples.append(img_bgr[h-border_size:h, :])  # Bottom
    samples.append(img_bgr[:, 0:border_size])  # Left
    samples.append(img_bgr[:, w-border_size:w])  # Right
    
    # Concatenate all border pixels
    border_pixels = np.vstack([s.reshape(-1, 3) for s in samples])
    
    # Convert to Lab
    border_lab = cv2.cvtColor(border_pixels.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2Lab)
    border_lab = border_lab.reshape(-1, 3).astype(np.float32)
    
    # Compute median (robust to outliers like text/drawings on edges)
    L_med = np.median(border_lab[:, 0])
    a_med = np.median(border_lab[:, 1])
    b_med = np.median(border_lab[:, 2])
    
    pdebug(f"[Freeform] Background Lab: L={L_med:.1f}, a={a_med:.1f}, b={b_med:.1f}")
    return (L_med, a_med, b_med)


def make_background_mask(img_bgr: NDArray, bg_lab: Tuple[float, float, float], 
                         delta: float = 12.0, debug_path: Optional[str] = None) -> NDArray:
    """Create binary mask of background pixels based on Lab distance.
    
    Args:
        img_bgr: Input image in BGR format
        bg_lab: Background color in Lab (L, a, b)
        delta: Maximum Lab distance to be considered background
        debug_path: Optional path to save debug mask image
        
    Returns:
        Binary mask (uint8, 255 = background, 0 = foreground)
    """
    # Convert image to Lab
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    
    # Compute Euclidean distance to background color
    bg_array = np.array(bg_lab, dtype=np.float32)
    dist = np.linalg.norm(img_lab - bg_array, axis=2)
    
    # Threshold: pixels close to background
    mask_bg = (dist < delta).astype(np.uint8) * 255
    
    # Morphology to clean up noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_OPEN, kernel_open)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_bg = cv2.morphologyEx(mask_bg, cv2.MORPH_CLOSE, kernel_close)
    
    bg_pct = (np.sum(mask_bg == 255) / mask_bg.size) * 100
    pdebug(f"[Freeform] Background mask: {bg_pct:.1f}% of image, delta={delta}")
    
    if debug_path:
        cv2.imwrite(debug_path, mask_bg)
        pdebug(f"[Freeform] Saved background mask: {debug_path}")
    
    return mask_bg


def has_content(img_bgr: NDArray, region: PanelRegion, mask_bg: NDArray, 
                min_content_ratio: float = 0.05) -> bool:
    """Check if a region contains actual content (not just empty background).
    
    Args:
        img_bgr: Input image in BGR format
        region: Panel region to check
        mask_bg: Background mask (255 = background, 0 = foreground)
        min_content_ratio: Minimum ratio of non-background pixels required
        
    Returns:
        True if region has sufficient content, False if mostly empty
    """
    x, y, w, h = region.bbox
    
    # Extract region from background mask
    region_mask_bg = mask_bg[y:y+h, x:x+w]
    
    if region_mask_bg.size == 0:
        return False
    
    # Count non-background pixels (foreground = content)
    foreground_pixels = np.sum(region_mask_bg == 0)
    total_pixels = region_mask_bg.size
    
    content_ratio = foreground_pixels / total_pixels
    
    # Region must have at least min_content_ratio of non-background pixels
    has_enough_content = content_ratio >= min_content_ratio
    
    if not has_enough_content:
        pdebug(f"[Freeform] Region at ({x},{y}) {w}x{h} rejected: content_ratio={content_ratio:.3f} < {min_content_ratio}")
    
    return has_enough_content


def segment_panels_watershed(img_bgr: NDArray, mask_bg: NDArray, 
                             sure_fg_ratio: float = 0.45,
                             debug_paths: Optional[dict] = None) -> NDArray:
    """Segment panels using watershed algorithm.
    
    Args:
        img_bgr: Input image in BGR format
        mask_bg: Background mask (255 = background)
        sure_fg_ratio: Ratio of max distance for sure foreground
        debug_paths: Optional dict with keys 'mask_fg', 'sure_fg', 'markers'
        
    Returns:
        Labeled regions (int32, 0 = background, 1+ = regions)
    """
    # Foreground = NOT background
    mask_fg = cv2.bitwise_not(mask_bg)
    
    # Clean foreground mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    if debug_paths and 'mask_fg' in debug_paths:
        cv2.imwrite(debug_paths['mask_fg'], mask_fg)
        pdebug(f"[Freeform] Saved foreground mask: {debug_paths['mask_fg']}")
    
    # Sure background: dilate background mask
    sure_bg = cv2.dilate(mask_bg, kernel, iterations=2)
    
    # Sure foreground: distance transform + threshold
    dist = cv2.distanceTransform(mask_fg, cv2.DIST_L2, 5)
    dist_max = dist.max()
    threshold_val = sure_fg_ratio * dist_max
    _, sure_fg = cv2.threshold(dist, threshold_val, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    
    pdebug(f"[Freeform] Distance transform max={dist_max:.1f}, threshold={threshold_val:.1f}")
    
    if debug_paths and 'sure_fg' in debug_paths:
        cv2.imwrite(debug_paths['sure_fg'], sure_fg)
        pdebug(f"[Freeform] Saved sure foreground: {debug_paths['sure_fg']}")
    
    # Unknown region
    sure_bg_bin = (sure_bg == 255).astype(np.uint8)
    sure_fg_bin = (sure_fg == 255).astype(np.uint8)
    unknown = cv2.subtract(sure_bg_bin, sure_fg_bin)
    
    # Create markers
    n_labels, markers = cv2.connectedComponents(sure_fg)
    pdebug(f"[Freeform] Connected components found: {n_labels - 1}")
    
    # Add 1 to all labels (so background is not 0)
    markers = markers + 1
    
    # Mark unknown regions as 0
    markers[unknown == 1] = 0
    
    if debug_paths and 'markers' in debug_paths:
        # Normalize markers for visualization
        markers_vis = ((markers.astype(np.float32) / markers.max()) * 255).astype(np.uint8)
        cv2.imwrite(debug_paths['markers'], markers_vis)
        pdebug(f"[Freeform] Saved markers: {debug_paths['markers']}")
    
    # Apply watershed
    markers = cv2.watershed(img_bgr, markers)
    
    # Count final regions (excluding background and borders)
    unique_labels = np.unique(markers)
    valid_labels = unique_labels[(unique_labels > 1)]
    pdebug(f"[Freeform] Watershed produced {len(valid_labels)} regions")
    
    return markers


def extract_panel_regions(markers: NDArray, img_shape: Tuple[int, int],
                          img_bgr: NDArray, mask_bg: NDArray,
                          min_area_ratio: float = 0.01,
                          max_area_ratio: float = 0.95,
                          min_fill_ratio: float = 0.25,
                          min_content_ratio: float = 0.05,
                          approx_eps_ratio: float = 0.01) -> List[PanelRegion]:
    """Extract and filter panel regions from watershed markers.
    
    Args:
        markers: Watershed output (labels)
        img_shape: Image dimensions (h, w)
        img_bgr: Original image in BGR format (for content checking)
        mask_bg: Background mask (for content checking)
        min_area_ratio: Minimum region area as fraction of image
        max_area_ratio: Maximum region area as fraction of image
        min_fill_ratio: Minimum fill ratio (area / bbox_area)
        min_content_ratio: Minimum ratio of non-background pixels
        approx_eps_ratio: Epsilon for polygon approximation
        
    Returns:
        List of filtered PanelRegion objects
    """
    h, w = img_shape
    img_area = h * w
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio
    
    regions = []
    unique_labels = np.unique(markers)
    
    for label in unique_labels:
        if label <= 1:  # Skip background (1) and borders (-1)
            continue
        
        # Create mask for this region
        region_mask = (markers == label).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Take largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Compute bbox
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox = (x, y, bw, bh)
        bbox_area = bw * bh
        
        if bbox_area == 0:
            continue
        
        fill_ratio = area / bbox_area
        
        # Filter by fill ratio
        if fill_ratio < min_fill_ratio:
            continue
        
        # Check if touches border
        touches_border = (x == 0 or y == 0 or x + bw >= w or y + bh >= h)
        
        # Compute OBB (oriented bounding box)
        rect = cv2.minAreaRect(contour)
        obb = cv2.boxPoints(rect)
        obb = np.intp(obb)
        
        # Polygon approximation
        perimeter = cv2.arcLength(contour, True)
        epsilon = approx_eps_ratio * perimeter
        poly = cv2.approxPolyDP(contour, epsilon, True)
        
        # Compute centroid
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
        
        # Filter empty regions (must have content)
        if not has_content(img_bgr, region, mask_bg, min_content_ratio):
            continue
        
        regions.append(region)
    
    pdebug(f"[Freeform] Extracted {len(regions)} regions after filtering")
    return regions


def merge_overlapping_regions(regions: List[PanelRegion], iou_thr: float = 0.20) -> List[PanelRegion]:
    """Merge regions with significant overlap.
    
    Args:
        regions: List of PanelRegion objects
        iou_thr: IoU threshold for merging
        
    Returns:
        List of merged regions
    """
    if len(regions) <= 1:
        return regions
    
    merged = []
    used = set()
    
    for i, reg in enumerate(regions):
        if i in used:
            continue
        
        # Try to merge with other regions
        to_merge = [reg]
        to_merge_indices = {i}
        
        for j, other in enumerate(regions):
            if j <= i or j in used:
                continue
            
            # Compute IoU on bboxes
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
            # Merge contours
            all_points = np.vstack([r.contour for r in to_merge])
            hull = cv2.convexHull(all_points)
            
            # Recompute properties
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
                touches_border=False,  # Re-check if needed
                centroid=(cx, cy)
            )
            
            merged.append(merged_reg)
            used.update(to_merge_indices)
            
            pdebug(f"[Freeform] Merged {len(to_merge)} regions")
        else:
            merged.append(reg)
            used.add(i)
    
    return merged


def sort_reading_order(regions: List[PanelRegion], rtl: bool = False) -> List[PanelRegion]:
    """Sort regions in reading order (top to bottom, left to right or right to left).
    
    Args:
        regions: List of PanelRegion objects
        rtl: Right-to-left reading order
        
    Returns:
        Sorted list of regions
    """
    if not regions:
        return regions
    
    # Sort by centroid Y first
    sorted_by_y = sorted(regions, key=lambda r: r.centroid[1])
    
    # Compute median height for grouping into rows
    heights = [r.bbox[3] for r in regions]
    median_height = np.median(heights) if heights else 100
    
    # Group into rows
    rows = []
    current_row = []
    current_y = None
    
    for reg in sorted_by_y:
        cy = reg.centroid[1]
        
        if current_y is None:
            current_y = cy
            current_row = [reg]
        elif abs(cy - current_y) < 0.5 * median_height:
            # Same row
            current_row.append(reg)
        else:
            # New row
            rows.append(current_row)
            current_row = [reg]
            current_y = cy
    
    if current_row:
        rows.append(current_row)
    
    # Sort each row by X (left to right or right to left)
    result = []
    for row in rows:
        row_sorted = sorted(row, key=lambda r: r.centroid[0], reverse=rtl)
        result.extend(row_sorted)
    
    pdebug(f"[Freeform] Sorted {len(regions)} regions into {len(rows)} rows")
    return result
