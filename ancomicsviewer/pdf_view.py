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
from PySide6.QtPdfWidgets import QPdfView

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
            return
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
        self.viewport().update()

    # --- Coordinate conversion ---

    def _effective_zoom_factor(self) -> float:
        """Get the effective zoom factor, accounting for FitInView/FitToWidth modes.

        In FitInView or FitToWidth modes, zoomFactor() may not reflect the actual
        zoom used for rendering. This method calculates the effective zoom based
        on the current zoom mode and viewport size.
        """
        doc = self.document()
        if not doc:
            return self.zoomFactor()

        cur = self.pageNavigator().currentPage()
        if cur < 0 or cur >= doc.pageCount():
            return self.zoomFactor()

        page_pts = doc.pagePointSize(cur)
        if page_pts.width() <= 0 or page_pts.height() <= 0:
            return self.zoomFactor()

        zoom_mode = self.zoomMode()

        if zoom_mode == QPdfView.ZoomMode.FitInView:
            # Calculate zoom to fit page in viewport
            margins = self.documentMargins()
            vw = self.viewport().width() - margins.left() - margins.right()
            vh = self.viewport().height() - margins.top() - margins.bottom()

            if vw <= 0 or vh <= 0:
                return self.zoomFactor()

            # FitInView: scale to fit both width and height
            z_w = vw / page_pts.width()
            z_h = vh / page_pts.height()
            return min(z_w, z_h)

        elif zoom_mode == QPdfView.ZoomMode.FitToWidth:
            # Calculate zoom to fit page width
            margins = self.documentMargins()
            vw = self.viewport().width() - margins.left() - margins.right()

            if vw <= 0:
                return self.zoomFactor()

            return vw / page_pts.width()

        else:
            # Custom mode: use reported zoomFactor
            return self.zoomFactor()

    def _page_to_view_xy(self, x_pt: float, y_pt: float) -> tuple[float, float]:
        """Convert page point coordinates to viewport pixels.

        Args:
            x_pt: X coordinate in page points (72 DPI)
            y_pt: Y coordinate in page points (72 DPI)

        Returns:
            Tuple of (x, y) in viewport pixel coordinates
        """
        doc = self.document()
        if not doc:
            return (0.0, 0.0)

        cur = self.pageNavigator().currentPage()
        page_pts = doc.pagePointSize(cur)
        z = self._effective_zoom_factor()

        # Get viewport dimensions and margins
        vw = self.viewport().width()
        vh = self.viewport().height()
        margins = self.documentMargins()
        spacing = self.pageSpacing()

        # Calculate page size in pixels
        page_w = page_pts.width() * z
        page_h = page_pts.height() * z

        # Calculate page position based on page mode and zoom mode
        zoom_mode = self.zoomMode()

        if self.pageMode() == QPdfView.PageMode.SinglePage:
            # SinglePage mode: page is centered
            avail_w = vw - margins.left() - margins.right()
            avail_h = vh - margins.top() - margins.bottom()

            # In FitInView, page is centered both horizontally and vertically
            if zoom_mode == QPdfView.ZoomMode.FitInView:
                page_x = margins.left() + max(0.0, (avail_w - page_w) / 2.0)
                page_y = margins.top() + max(0.0, (avail_h - page_h) / 2.0)
            else:
                page_x = margins.left() + max(0.0, (avail_w - page_w) / 2.0)
                page_y = margins.top()
        else:
            # MultiPage mode: pages stacked vertically
            avail_w = vw - margins.left() - margins.right()
            page_x = margins.left() + max(0.0, (avail_w - page_w) / 2.0)
            page_y = margins.top()
            for i in range(cur):
                prev_pts = doc.pagePointSize(i)
                page_y += prev_pts.height() * z + spacing

        # Apply scroll offset
        page_x -= self.horizontalScrollBar().value()
        page_y -= self.verticalScrollBar().value()

        # Convert page coordinates to viewport coordinates
        view_x = page_x + (x_pt * z)
        view_y = page_y + (y_pt * z)

        return (view_x, view_y)

    def _page_rect_to_view(self, r: QRectF) -> QRectF:
        """Convert page point rectangle to viewport pixels."""
        x, y = self._page_to_view_xy(r.left(), r.top())
        z = self._effective_zoom_factor()
        return QRectF(x, y, r.width() * z, r.height() * z)

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
        """Draw blue outline around page for debugging."""
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
