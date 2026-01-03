"""Custom PDF view widget with pan, zoom, and panel overlay.

Provides:
- Mouse-based panning (left-click drag)
- Ctrl+wheel zoom
- Panel overlay visualization
- Debug split lines
- Context menu for export
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from PySide6.QtCore import Qt, QPoint, QRectF
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QMenu
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView

from .image_utils import pdebug

if TYPE_CHECKING:
    from .detector import DebugInfo


class PannablePdfView(QPdfView):
    """PDF view with mouse panning, zoom, and panel overlay support.

    Features:
    - Left-click drag to pan when scrollbars are active
    - Ctrl+wheel to zoom
    - Green overlay rectangles for detected panels
    - Yellow dashed lines for debug split visualization
    - Right-click context menu for page export
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._panning = False
        self._last_pan_point = QPoint()
        self.setPageMode(QPdfView.PageMode.MultiPage)

        # Overlay state
        self._overlay_enabled = False
        self._overlay_rects: List[QRectF] = []
        self._debug_info: Optional[DebugInfo] = None
        self._show_debug_lines = False
        
        # Debug configuration
        self._config_debug = True  # Enable debug logging for coordinate conversion

    # --- Panning with left click ---

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self._scrollbars_active():
            self._panning = True
            self._last_pan_point = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position().toPoint() - self._last_pan_point
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self._last_pan_point = event.position().toPoint()
            self._invalidate_page_cache()  # Invalidate on scroll
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # --- Zoom with Ctrl+wheel ---

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.pixelDelta().y() or event.angleDelta().y()
            if delta:
                z = self.zoomFactor()
                factor = 1.0 + (0.0015 * delta)
                self.setZoomMode(QPdfView.ZoomMode.Custom)
                self.setZoomFactor(max(0.05, min(16.0, z * factor)))
                self._invalidate_page_cache()  # Invalidate on zoom
            return
        self._invalidate_page_cache()  # Invalidate on scroll
        super().wheelEvent(event)

    def _scrollbars_active(self) -> bool:
        """Check if scrollbars are currently needed."""
        h = self.horizontalScrollBar()
        v = self.verticalScrollBar()
        return (h and h.maximum() > h.minimum()) or (v and v.maximum() > v.minimum())

    # --- Context menu ---

    def _show_context_menu(self, global_pos: QPoint) -> None:
        """Show right-click context menu."""
        menu = QMenu(self)
        export_action = menu.addAction("Export Page as PNG")
        export_action.triggered.connect(self._export_current_page)
        menu.exec(global_pos)

    def _export_current_page(self) -> None:
        """Delegate export to main window."""
        main = self.window()
        if hasattr(main, "export_current_page"):
            main.export_current_page()

    # --- Panel overlay API ---

    def set_panel_overlay(
        self,
        rects: List[QRectF],
        enabled: bool,
        debug_info: Optional[DebugInfo] = None,
        show_debug: bool = False
    ) -> None:
        """Set panel overlay rectangles.

        Args:
            rects: Panel rectangles in page point coordinates
            enabled: Whether to show the overlay
            debug_info: Optional debug info for split lines
            show_debug: Whether to show debug split lines
        """
        self._overlay_rects = rects or []
        self._overlay_enabled = bool(enabled)
        self._debug_info = debug_info
        self._show_debug_lines = show_debug

        # Trigger repaint immediately - calculations happen in paintEvent
        self.viewport().update()

    # --- Coordinate conversion helpers ---

    def _find_page_origin(self) -> tuple[float, float, float]:
        """Find the actual page origin and scale by sampling the viewport.

        Returns (x_origin, y_origin, scale_factor) where:
        - x_origin: left edge of page in viewport coordinates
        - y_origin: top edge of page (relative to visible area, can be negative)
        - scale_factor: ratio of actual rendered size to expected size
        """
        # Prevent recursive calls during detection
        if getattr(self, '_detecting_origin', False):
            return self._fallback_page_origin()
        self._detecting_origin = True

        try:
            from PySide6.QtGui import QColor

            doc = self.document()
            if not doc:
                return (0.0, 0.0, 1.0)

            cur = self.pageNavigator().currentPage()
            page_pts = doc.pagePointSize(cur)
            z = self.zoomFactor()

            # Grab viewport to find page edge
            viewport = self.viewport()
            img = viewport.grab().toImage()

            vw = img.width()
            vh = img.height()

            if vw <= 0 or vh <= 0:
                return self._fallback_page_origin()

            # Sample background color from corners to check if we can see page edges
            corners = [(5, 5), (vw - 5, 5), (5, vh - 5), (vw - 5, vh - 5)]
            bg_samples = []
            for cx, cy in corners:
                if 0 <= cx < vw and 0 <= cy < vh:
                    c = QColor(img.pixel(cx, cy))
                    bg_samples.append(c.lightness())

            # If all corners are bright (page content), we're zoomed in - use fallback
            if bg_samples and all(l > 200 for l in bg_samples):
                # Zoomed in - page fills viewport, use fallback with stored scale
                return self._fallback_page_origin()

            bg_lightness = min(bg_samples) if bg_samples else 128
            threshold = min(250, bg_lightness + 30)

            # Find left edge of page by scanning horizontally at viewport center
            left_edge = 0
            mid_y = vh // 2
            for x in range(vw):
                c = QColor(img.pixel(x, mid_y))
                if c.lightness() > threshold:
                    left_edge = x
                    break

            # Find right edge
            right_edge = vw - 1
            for x in range(vw - 1, -1, -1):
                c = QColor(img.pixel(x, mid_y))
                if c.lightness() > threshold:
                    right_edge = x
                    break

            # Calculate actual width and scale
            actual_width = right_edge - left_edge
            expected_width = page_pts.width() * z
            detected_scale = actual_width / expected_width if expected_width > 0 else 1.0

            # Force scale to 1.0 - Qt should render at exact zoom factor
            # Auto-detection can be inaccurate due to pixel sampling artifacts
            scale = 1.0
            
            # Log the detected vs forced scale for debugging
            if self._config_debug and abs(detected_scale - 1.0) > 0.05:
                pdebug(f"Scale detection: detected={detected_scale:.3f}, using scale=1.0 (forced)")

            # Sanity check - if detection is way off, use fallback
            if detected_scale < 0.8 or detected_scale > 1.5:
                return self._fallback_page_origin()

            # Store successful scale for this page (for use when zoomed in)
            if not hasattr(self, '_page_scales'):
                self._page_scales = {}
            self._page_scales[cur] = scale

            # Find top edge (scan from top at horizontal center)
            top_edge = 0
            mid_x = (left_edge + right_edge) // 2
            for y in range(vh):
                c = QColor(img.pixel(mid_x, y))
                if c.lightness() > threshold:
                    top_edge = y
                    break

            return (float(left_edge), float(top_edge), scale)
        finally:
            self._detecting_origin = False

    def _fallback_page_origin(self) -> tuple[float, float, float]:
        """Fallback calculation when detection fails."""
        doc = self.document()
        if not doc:
            return (0.0, 0.0, 1.0)

        cur = self.pageNavigator().currentPage()
        page_pts = doc.pagePointSize(cur)
        z = self.zoomFactor()
        margins = self.documentMargins()
        spacing = self.pageSpacing()

        vw = self.viewport().width()

        # Calculate document width
        doc_content_w = 0.0
        for i in range(doc.pageCount()):
            pw = doc.pagePointSize(i).width() * z
            if pw > doc_content_w:
                doc_content_w = pw

        content_w = page_pts.width() * z
        avail_w = vw - margins.left() - margins.right()
        doc_x_offset = max(0.0, (avail_w - doc_content_w) / 2.0)
        page_x_in_doc = (doc_content_w - content_w) / 2.0

        # Y offset
        page_y_in_doc = 0.0
        for i in range(cur):
            prev_pts = doc.pagePointSize(i)
            page_y_in_doc += prev_pts.height() * z + spacing

        sx = self.horizontalScrollBar().value()
        sy = self.verticalScrollBar().value()

        x = margins.left() + doc_x_offset + page_x_in_doc - sx
        y = margins.top() + page_y_in_doc - sy

        return (x, y, 1.0)

    def _page_to_view_xy(self, x_pt: float, y_pt: float) -> tuple[float, float]:
        """Convert page point coordinates to viewport pixels.

        Simple direct calculation based on page position in viewport.
        """
        doc = self.document()
        if not doc:
            return (0.0, 0.0)

        cur = self.pageNavigator().currentPage()
        page_pts = doc.pagePointSize(cur)
        z = self.zoomFactor()
        
        # Get viewport dimensions
        vw = self.viewport().width()
        vh = self.viewport().height()
        
        # Calculate page size in pixels
        page_w = page_pts.width() * z
        page_h = page_pts.height() * z
        
        # Get document margins and spacing - force recalculation
        margins = self.documentMargins()
        spacing = self.pageSpacing()
        
        # In SinglePage mode (panel mode), page is centered if smaller than viewport
        if self.pageMode() == QPdfView.PageMode.SinglePage:
            # Center horizontally if page is smaller than viewport
            page_x = margins.left() + max(0.0, (vw - margins.left() - margins.right() - page_w) / 2.0)
            # Top margin for vertical
            page_y = margins.top()
        else:
            # In MultiPage mode, calculate position based on previous pages
            page_x = margins.left()
            page_y = margins.top()
            
            # Add heights of all previous pages
            for i in range(cur):
                prev_pts = doc.pagePointSize(i)
                page_y += prev_pts.height() * z + spacing
        
        # Subtract scroll position - get fresh values
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        sx = hbar.value() if hbar else 0
        sy = vbar.value() if vbar else 0
        
        page_x -= sx
        page_y -= sy
        
        # Convert page coordinates to viewport coordinates
        view_x = page_x + (x_pt * z)
        view_y = page_y + (y_pt * z)
        
        if self._config_debug:
            pdebug(f"_page_to_view: pt=({x_pt:.1f},{y_pt:.1f}) z={z:.3f} mode={self.pageMode()} vp={vw}x{vh} page={page_w:.1f}x{page_h:.1f} margins=({margins.left():.1f},{margins.top():.1f}) scroll=({sx},{sy}) -> pos=({page_x:.1f},{page_y:.1f}) view=({view_x:.1f},{view_y:.1f})")
        
        return (view_x, view_y)

    def _invalidate_page_cache(self) -> None:
        """Invalidate cached page origin (call on scroll, zoom, page change)."""
        self._cached_page_origin = None

    def _page_rect_to_view(self, r: QRectF) -> QRectF:
        """Convert page point rectangle to viewport pixels."""
        x, y = self._page_to_view_xy(r.left(), r.top())
        
        z = self.zoomFactor()
        w = r.width() * z
        h = r.height() * z
        
        return QRectF(x, y, w, h)

    # --- Paint overlay ---

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self._overlay_enabled or not self._overlay_rects:
            return

        try:
            painter = QPainter(self.viewport())
            doc = self.document()
            cur = self.pageNavigator().currentPage()

            if not doc or cur < 0:
                return

            page_pts = doc.pagePointSize(cur)
            if page_pts.width() <= 0 or page_pts.height() <= 0:
                return

            # Draw page frame (blue debug outline)
            self._draw_page_frame(painter, page_pts)

            # Draw panel rectangles
            self._draw_panels(painter)

            # Draw debug split lines
            if self._show_debug_lines and self._debug_info:
                self._draw_debug_lines(painter)

        except Exception:
            pass

    def _draw_page_frame(self, painter: QPainter, page_pts) -> None:
        """Draw blue outline around page."""
        page_rect = self._page_rect_to_view(
            QRectF(0, 0, page_pts.width(), page_pts.height())
        )
        pen = QPen(QColor(0, 120, 255, 200), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(page_rect)

    def _draw_panels(self, painter: QPainter) -> None:
        """Draw green panel overlay rectangles with indices."""
        pen = QPen(QColor(0, 200, 0, 220), 2)
        fill = QColor(0, 200, 0, 55)
        text_pen = QPen(QColor(0, 0, 0, 255), 1)

        painter.setPen(pen)
        painter.setBrush(fill)

        for idx, r in enumerate(self._overlay_rects):
            if r.isEmpty():
                continue

            vr = self._page_rect_to_view(r)
            painter.drawRect(vr)

            # Draw panel index
            painter.setPen(text_pen)
            painter.drawText(
                vr.adjusted(3, 3, -3, -3),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                f"#{idx + 1}"
            )
            painter.setPen(pen)

    def _draw_debug_lines(self, painter: QPainter) -> None:
        """Draw yellow dashed lines for debug splits."""
        pen = QPen(QColor(255, 215, 0, 230), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Vertical splits
        for (x_pt, y_pt, w_pt, h_pt) in self._debug_info.vertical_splits:
            r = QRectF(x_pt, y_pt, max(0.5, w_pt), h_pt)
            vr = self._page_rect_to_view(r)
            painter.drawLine(int(vr.left()), int(vr.top()), int(vr.left()), int(vr.bottom()))

        # Horizontal splits
        for (x_pt, y_pt, w_pt, h_pt) in self._debug_info.horizontal_splits:
            r = QRectF(x_pt, y_pt, w_pt, max(0.5, h_pt))
            vr = self._page_rect_to_view(r)
            painter.drawLine(int(vr.left()), int(vr.top()), int(vr.right()), int(vr.top()))
