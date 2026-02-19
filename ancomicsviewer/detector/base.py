"""Core PanelDetector class with main detection flow.

This is the main entry point for panel detection, coordinating:
- Adaptive threshold detection
- Gutter-based detection
- Line segment detection (LSD)
- Contour hierarchy analysis
- Freeform/watershed detection
- Multi-scale consensus merging
- K-means background clustering
- Template layout matching
- Post-processing filters
"""

from __future__ import annotations

import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
import threading
import os
from datetime import datetime

from PySide6.QtCore import QRectF, QSizeF
from PySide6.QtGui import QImage

from ..config import DetectorConfig
from ..image_utils import (
    qimage_to_numpy_rgba,
    rgba_to_grayscale,
    rgba_to_lab_l,
    HAS_CV2,
    HAS_NUMPY,
)

from .utils import DebugInfo, pdebug, estimate_bg_lab, non_bg_ratio, merge_rects
from .classifier import classify_page_style
from .adaptive import adaptive_threshold_route, rects_from_mask
from .gutter import gutter_based_detection
from .freeform import freeform_detection
from .line_detection import line_based_detection
from .contour_hierarchy import hierarchy_based_detection
from .multiscale import multiscale_detect, score_detection_result, select_best_result
from .clustering import make_kmeans_background_mask, get_dominant_bg_lab
from .templates import refine_with_template
from .filters import (
    filter_title_rows,
    filter_by_area,
    filter_empty_rects,
    filter_by_lab_content,
    suppress_nested_rects,
    remove_header_footer_strips,
    sort_by_reading_order,
    merge_overlapping_rects,
)

if HAS_CV2:
    import cv2
if HAS_NUMPY:
    import numpy as np
    from numpy.typing import NDArray


class PanelDetector:
    """High-performance comic panel detector.

    Uses OpenCV heuristics with multiple detection routes:
    1. Adaptive threshold + morphology (primary)
    2. Gutter-based detection (for white separations)
    3. Freeform/watershed (for complex layouts)

    Thread-safe with internal locking for shared state.
    """

    # Shared thread pool
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
        self._debug_dir: Optional[str] = None
        self._decision_context: Dict[str, Any] = {}

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

    def detect_panels(
        self,
        qimage: QImage,
        page_point_size: QSizeF,
        page_num: Optional[int] = None,
        pdf_path: Optional[str] = None,
        dpi: float = 150.0,
    ) -> List[QRectF]:
        """Detect comic panels in a rendered page image.

        Args:
            qimage: Rendered page as QImage
            page_point_size: Page dimensions in points (72 DPI)
            page_num: Page number (0-indexed) for debug output
            pdf_path: PDF file path for debug output
            dpi: DPI used for rendering

        Returns:
            List of panel rectangles in page point coordinates
        """
        if not HAS_CV2 or not HAS_NUMPY:
            pdebug("OpenCV/NumPy not available -> no detection")
            return []

        try:
            with self._lock:
                self._last_debug = DebugInfo.empty()
                if self.config.debug and page_num is not None:
                    self._debug_dir = self._setup_debug_directory(page_num)
                else:
                    self._debug_dir = None

            self._log_params()

            # Convert QImage to numpy
            arr = qimage_to_numpy_rgba(qimage)
            if arr is None:
                pdebug("QImage conversion failed")
                return []

            h, w = arr.shape[:2]
            pdebug(f"Image size: {w}x{h}")

            # Scale down large images
            scale_factor = 1.0
            max_width = 2400
            if w > max_width:
                scale_factor = max_width / w
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                pdebug(f"Scaling: {w}x{h} -> {new_w}x{new_h}")
                arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = arr.shape[:2]

            # Prepare channels
            gray = rgba_to_grayscale(arr)
            L = rgba_to_lab_l(arr, apply_clahe=True)
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

            # Estimate background (use k-means if enabled for better accuracy)
            if self.config.use_kmeans_bg:
                bg_lab = get_dominant_bg_lab(img_bgr, k=self.config.kmeans_k)
                pdebug(f"bg_lab(kmeans)=({bg_lab[0]:.1f}, {bg_lab[1]:.1f}, {bg_lab[2]:.1f})")
            else:
                bg_lab = estimate_bg_lab(img_bgr)
                pdebug(f"bg_lab=({bg_lab[0]:.1f}, {bg_lab[1]:.1f}, {bg_lab[2]:.1f})")

            # Run detection pipeline
            rects = self._detection_pipeline(
                gray, L, img_bgr, bg_lab, w, h, page_point_size
            )

            # Post-processing
            rects = self._post_process(rects, gray, L, img_bgr, bg_lab, w, h, page_point_size)

            # Sort by reading order
            rects = sort_by_reading_order(rects, rtl=self.config.reading_rtl)
            pdebug(f"Final: {len(rects)} panels, RTL={self.config.reading_rtl}")

            # Export debug
            if self.config.debug and self._debug_dir:
                self._export_debug_info(page_num, pdf_path, page_point_size, dpi, rects)

            return rects

        except Exception:
            pdebug(f"detect_panels error:\n{traceback.format_exc()}")
            return []

    def _detection_pipeline(
        self,
        gray: NDArray,
        L: NDArray,
        img_bgr: NDArray,
        bg_lab: NDArray,
        w: int,
        h: int,
        page_point_size: QSizeF,
    ) -> List[QRectF]:
        """Main detection pipeline using configured mode.

        Enhanced pipeline with multi-route detection:
        1. Adaptive threshold (original)
        2. Line segment detection (LSD + gradient borders)
        3. Gutter-based detection
        4. Contour hierarchy analysis
        5. Freeform/watershed
        6. Multi-scale consensus (wraps the above)
        7. Template matching refinement

        Returns:
            List of detected panel rectangles
        """
        panel_mode = self.config.panel_mode

        # === Phase 1: Multi-route detection ===

        # If multi-scale is enabled, wrap the core detection
        if self.config.use_multiscale:
            rects = multiscale_detect(
                detect_fn=self._core_detection,
                gray=gray, L=L, img_bgr=img_bgr,
                w=w, h=h,
                page_point_size=page_point_size,
                config=self.config,
                scales=self.config.multiscale_factors,
                min_agreement=self.config.multiscale_min_agreement,
            )
            pdebug(f"[Pipeline] Multi-scale result: {len(rects)} panels")
        else:
            rects = self._core_detection(gray, L, img_bgr, w, h, page_point_size, self.config)

        # === Phase 2: Template refinement ===
        if self.config.use_template_matching and rects is not None:
            rects = refine_with_template(
                rects, gray, page_point_size,
                config=self.config,
                min_template_score=self.config.template_min_score,
                detection_score_threshold=self.config.template_detection_threshold,
            )

        return rects

    def _core_detection(
        self,
        gray: NDArray,
        L: NDArray,
        img_bgr: NDArray,
        w: int,
        h: int,
        page_point_size: QSizeF,
        config: "DetectorConfig",
    ) -> List[QRectF]:
        """Core detection at a single scale - runs all routes and selects best.

        This method is called by multiscale_detect at each scale factor.
        """
        panel_mode = config.panel_mode

        # Route 1: Adaptive threshold (fast, primary)
        mask = adaptive_threshold_route(gray, config)
        adaptive_rects = rects_from_mask(mask, w, h, page_point_size, config)
        pdebug(f"Adaptive: {len(adaptive_rects)} rects")

        # Auto-classify if needed
        if panel_mode == "auto":
            panel_mode = classify_page_style(img_bgr, adaptive_rects, w, h, config.debug)
            pdebug(f"[Auto] Classified: {panel_mode}")

        # Store decision context
        self._decision_context = {
            "panel_mode_input": config.panel_mode,
            "panel_mode_used": panel_mode,
            "adaptive_initial_count": len(adaptive_rects),
        }

        # Run all enabled detection routes in parallel-ready fashion
        route_results: Dict[str, List[QRectF]] = {
            "adaptive": adaptive_rects,
        }

        # Route 2: Line segment detection (LSD)
        if config.use_line_detection:
            try:
                lsd_rects = line_based_detection(gray, w, h, page_point_size, config, img_bgr)
                if lsd_rects:
                    route_results["lsd"] = lsd_rects
                    pdebug(f"LSD: {len(lsd_rects)} rects")
            except Exception as e:
                pdebug(f"LSD failed: {e}")

        # Route 3: Contour hierarchy
        if config.use_hierarchy:
            try:
                hierarchy_rects = hierarchy_based_detection(gray, w, h, page_point_size, config)
                if hierarchy_rects:
                    route_results["hierarchy"] = hierarchy_rects
                    pdebug(f"Hierarchy: {len(hierarchy_rects)} rects")
            except Exception as e:
                pdebug(f"Hierarchy failed: {e}")

        # Route 4: Gutter-based (for grid layouts)
        gutter_rects = gutter_based_detection(gray, L, w, h, page_point_size, config, img_bgr)
        if gutter_rects:
            route_results["gutter"] = gutter_rects
            pdebug(f"Gutter: {len(gutter_rects)} rects")

        # === Route selection based on page style ===
        if panel_mode == "classic_franco_belge":
            rects = self._classic_route_selection(route_results, page_point_size)
        else:
            rects = self._modern_route_selection(
                route_results, gray, L, img_bgr, w, h, page_point_size, config
            )

        return rects

    def _classic_route_selection(
        self,
        route_results: Dict[str, List[QRectF]],
        page_point_size: QSizeF,
    ) -> List[QRectF]:
        """Select best route for classic Franco-Belge style."""
        pdebug("[Policy] CLASSIC_FRANCO_BELGE")

        # Score all routes and pick the best
        best_name, best_rects = select_best_result(route_results, page_point_size)
        self._decision_context["route_chosen"] = best_name
        self._decision_context["routes_available"] = {
            k: len(v) for k, v in route_results.items()
        }

        return best_rects

    def _modern_route_selection(
        self,
        route_results: Dict[str, List[QRectF]],
        gray: NDArray,
        L: NDArray,
        img_bgr: NDArray,
        w: int,
        h: int,
        page_point_size: QSizeF,
        config: "DetectorConfig",
    ) -> List[QRectF]:
        """Select best route for modern/complex style."""
        pdebug("[Policy] MODERN")

        # Header/footer removal on all routes
        cleaned_results = {}
        for name, rects in route_results.items():
            cleaned = remove_header_footer_strips(rects, float(page_point_size.height()))
            if cleaned:
                cleaned_results[name] = cleaned

        if not cleaned_results:
            cleaned_results = route_results

        # Score and select best
        best_name, best_rects = select_best_result(cleaned_results, page_point_size)

        # Freeform fallback if best result is weak
        if config.use_freeform_fallback:
            det_score = score_detection_result(best_rects, page_point_size)
            if det_score < 0.45:
                pdebug(f"[Freeform] Triggering: det_score={det_score:.3f} < 0.45")
                self._decision_context["freeform_triggered"] = True
                freeform_rects = freeform_detection(img_bgr, w, h, page_point_size, config)
                if freeform_rects:
                    ff_score = score_detection_result(freeform_rects, page_point_size)
                    if ff_score > det_score:
                        best_rects = freeform_rects
                        best_name = "freeform"
                        pdebug(f"Freeform: {len(freeform_rects)} rects (score={ff_score:.3f})")
            else:
                self._decision_context["freeform_triggered"] = False

        self._decision_context["route_chosen"] = best_name
        self._decision_context["routes_available"] = {
            k: len(v) for k, v in route_results.items()
        }

        return best_rects

    def _post_process(
        self,
        rects: List[QRectF],
        gray: NDArray,
        L: NDArray,
        img_bgr: NDArray,
        bg_lab: NDArray,
        w: int,
        h: int,
        page_point_size: QSizeF,
    ) -> List[QRectF]:
        """Apply post-processing filters."""
        rects_backup = list(rects)

        # Title row filter
        if rects and self.config.filter_title_rows:
            rects = filter_title_rows(rects, page_point_size, self.config.debug)
            pdebug(f"After title filter: {len(rects)}")

        # Area filter
        if rects:
            rects = filter_by_area(rects, page_point_size, self.config.min_area_pct, self.config.max_area_pct)
            pdebug(f"After area filter: {len(rects)}")

        # Empty rect filter
        if rects:
            rects = filter_empty_rects(rects, gray, w, h, page_point_size)
            pdebug(f"After empty filter: {len(rects)}")

        # Lab-based filter
        if rects:
            initial = len(rects)
            rects = filter_by_lab_content(
                rects, img_bgr, gray, bg_lab, w, h, page_point_size,
                self.config.min_non_bg_ratio, self.config.min_dim_ratio, self.config.freeform_bg_delta
            )
            if len(rects) < initial:
                pdebug(f"Lab filter: {initial} -> {len(rects)}")

        # Nested suppression
        if rects:
            rects = suppress_nested_rects(
                rects, img_bgr, bg_lab, w, h, page_point_size,
                delta=self.config.freeform_bg_delta
            )
            pdebug(f"After nested suppression: {len(rects)}")

        # Splash page fallback
        rects = self._splash_page_fallback(rects, rects_backup, img_bgr, bg_lab, page_point_size)

        return rects

    def _splash_page_fallback(
        self,
        rects: List[QRectF],
        rects_backup: List[QRectF],
        img_bgr: NDArray,
        bg_lab: NDArray,
        page_point_size: QSizeF,
    ) -> List[QRectF]:
        """Handle splash pages and anti-collapse."""
        page_area = page_point_size.width() * page_point_size.height()
        panel_mode = self._decision_context.get("panel_mode_used", "unknown")

        # Single small panel on content-rich page -> promote to full page
        if rects and len(rects) == 1:
            panel_area = rects[0].width() * rects[0].height()
            area_ratio = panel_area / page_area if page_area > 0 else 0
            non_bg_page = non_bg_ratio(img_bgr, bg_lab, self.config.freeform_bg_delta)

            should_promote = area_ratio < 0.60 and non_bg_page > 0.35

            # Anti-collapse for modern mode
            if should_promote and panel_mode == "modern" and len(rects_backup) >= 2:
                pdebug("[ANTI_COLLAPSE] Skip promotion")
                should_promote = False

            if should_promote:
                pdebug(f"[splash] Promote to full-page: area_ratio={area_ratio:.2f}")
                return [QRectF(0, 0, page_point_size.width(), page_point_size.height())]

        # Empty results on content-rich page -> full page
        if not rects and img_bgr is not None:
            if panel_mode == "modern" and len(rects_backup) >= 2:
                pdebug(f"[ANTI_COLLAPSE] Restore {len(rects_backup)} panels")
                return rects_backup
            
            non_bg_page = non_bg_ratio(img_bgr, bg_lab, self.config.freeform_bg_delta)
            if non_bg_page > 0.35:
                pdebug(f"[splash] Empty -> full-page: non_bg={non_bg_page:.2f}")
                return [QRectF(0, 0, page_point_size.width(), page_point_size.height())]

        return rects

    def _log_params(self) -> None:
        """Log current detection parameters."""
        c = self.config
        pdebug(
            "params:",
            f"ab={c.adaptive_block} C={c.adaptive_C}",
            f"mk={c.morph_kernel} mi={c.morph_iter}",
            f"panel_mode={c.panel_mode}"
        )

    def _setup_debug_directory(self, page_num: int) -> str:
        """Create debug directory for this page."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        page_dir = f"page_{page_num+1:03d}_{timestamp}"
        debug_dir = os.path.join("debug_output", page_dir)
        os.makedirs(debug_dir, exist_ok=True)
        pdebug(f"[Debug] Created: {debug_dir}")
        return debug_dir

    def _export_debug_info(
        self,
        page_num: Optional[int],
        pdf_path: Optional[str],
        page_point_size: QSizeF,
        dpi: float,
        rects: List[QRectF],
    ) -> None:
        """Export debug information."""
        if not self._debug_dir:
            return

        import json
        from pathlib import Path

        # Info file
        info_data = {
            "pdf_name": Path(pdf_path).name if pdf_path else "unknown",
            "page_number": (page_num + 1) if page_num is not None else "unknown",
            "timestamp": datetime.now().isoformat(),
            "dpi": dpi,
            "page_size": {"width": page_point_size.width(), "height": page_point_size.height()},
        }

        info_path = os.path.join(self._debug_dir, "info.txt")
        try:
            with open(info_path, "w", encoding="utf-8") as f:
                for key, value in info_data.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for k, v in value.items():
                            f.write(f"  {k}: {v}\n")
                    else:
                        f.write(f"{key}: {value}\n")
        except Exception as e:
            pdebug(f"[Debug] Info export error: {e}")

        # Panels file
        panels_path = os.path.join(self._debug_dir, "panels.txt")
        try:
            scale = dpi / 72.0
            with open(panels_path, "w", encoding="utf-8") as f:
                f.write(f"panel_count: {len(rects)}\n")
                for i, rect in enumerate(rects):
                    x_px = int(rect.left() * scale)
                    y_px = int(rect.top() * scale)
                    w_px = int(rect.width() * scale)
                    h_px = int(rect.height() * scale)
                    f.write(f"panel_{i+1}: x={x_px}, y={y_px}, w={w_px}, h={h_px}\n")
        except Exception as e:
            pdebug(f"[Debug] Panels export error: {e}")

        # Decision JSON
        decision_path = os.path.join(self._debug_dir, "decision.json")
        try:
            with open(decision_path, "w", encoding="utf-8") as f:
                json.dump(self._decision_context, f, indent=2)
        except Exception as e:
            pdebug(f"[Debug] Decision export error: {e}")
