"""Custom PDF view widget with pan, zoom, and panel overlay.

Provides:
- Mouse-based panning (left-click drag)
- Ctrl+wheel zoom
- Panel overlay visualization
- Debug split lines
- Context menu for export
- Panel editing mode (E to toggle)
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from PySide6.QtCore import Qt, QPoint, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import QMenu
from PySide6.QtPdfWidgets import QPdfView

from .panel_editor import PanelEditor, EditHandle

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
    - Panel editing mode (E to toggle, drag corners to resize)
    """

    # Signals for panel editing
    panels_modified = Signal(list)  # Emitted when panels are modified in edit mode
    edit_mode_changed = Signal(bool)  # Emitted when edit mode is toggled

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

        # Panel editor
        self._panel_editor = PanelEditor()
        self._panel_editor.set_callbacks(
            on_panels_changed=self._on_editor_panels_changed,
            on_edit_mode_changed=self._on_editor_mode_changed,
        )

        # Enable keyboard input
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # --- Panning with left click ---

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())
            event.accept()
            return

        # Edit mode handling
        if self._panel_editor.edit_mode and event.button() == Qt.MouseButton.LeftButton:
            pos_pt = self._view_to_page_pt(event.position())
            z = self._effective_zoom_factor()
            
            # Check if clicking on a handle of selected panel
            handle = self._panel_editor.get_handle_at(pos_pt, z)
            
            if handle != EditHandle.NONE:
                # Start dragging handle
                self._panel_editor.start_drag(pos_pt, handle)
                event.accept()
                return
            
            # Check if clicking on a panel
            if self._panel_editor.select_panel_at(pos_pt, z):
                handle = self._panel_editor.get_handle_at(pos_pt, z)
                if handle != EditHandle.NONE:
                    self._panel_editor.start_drag(pos_pt, handle)
                self.viewport().update()
                event.accept()
                return
            
            # Start drawing new panel (with Shift)
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._panel_editor.start_new_panel(pos_pt)
                event.accept()
                return
            
            # Click on empty space - deselect
            self._panel_editor._selected_index = -1
            self.viewport().update()

        if event.button() == Qt.MouseButton.LeftButton and self._scrollbars_active():
            if not self._panel_editor.is_dragging and not self._panel_editor.is_drawing:
                self._panning = True
                self._last_pan_point = event.position().toPoint()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Edit mode: update drag or drawing
        if self._panel_editor.edit_mode:
            pos_pt = self._view_to_page_pt(event.position())
            z = self._effective_zoom_factor()
            
            if self._panel_editor.is_dragging:
                self._panel_editor.update_drag(pos_pt)
                self.viewport().update()
                event.accept()
                return
            
            if self._panel_editor.is_drawing:
                self._panel_editor.update_new_panel(pos_pt)
                self.viewport().update()
                event.accept()
                return
            
            # Update cursor based on handle under mouse
            handle = self._panel_editor.get_handle_at(pos_pt, z)
            self.setCursor(self._panel_editor.get_cursor_for_handle(handle))
        
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
        if event.button() == Qt.MouseButton.LeftButton:
            if self._panel_editor.is_dragging:
                self._panel_editor.end_drag()
                self.viewport().update()
                event.accept()
                return
            
            if self._panel_editor.is_drawing:
                self._panel_editor.end_new_panel()
                self.viewport().update()
                event.accept()
                return
            
            if self._panning:
                self._panning = False
                if not self._panel_editor.edit_mode:
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                return
        
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Handle keyboard events for panel editing."""
        if self._panel_editor.edit_mode:
            # Delete selected panel
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                if self._panel_editor.delete_selected():
                    self.viewport().update()
                event.accept()
                return
            
            # Undo changes
            if event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._panel_editor.undo()
                self.viewport().update()
                event.accept()
                return
            
            # Escape: exit edit mode
            if event.key() == Qt.Key.Key_Escape:
                self._panel_editor.edit_mode = False
                self.viewport().update()
                event.accept()
                return
        
        # Toggle edit mode with E
        if event.key() == Qt.Key.Key_E and self._overlay_enabled:
            self._panel_editor.edit_mode = not self._panel_editor.edit_mode
            if self._panel_editor.edit_mode:
                self._panel_editor.set_panels(self._overlay_rects)
            self.viewport().update()
            event.accept()
            return
        
        super().keyPressEvent(event)

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

    # --- Panel editing callbacks ---

    def _on_editor_panels_changed(self, panels: List[QRectF]) -> None:
        """Handle panels modified in editor."""
        self._overlay_rects = panels
        self.panels_modified.emit(panels)

    def _on_editor_mode_changed(self, enabled: bool) -> None:
        """Handle edit mode changed."""
        self.edit_mode_changed.emit(enabled)
        if not enabled:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    @property
    def is_edit_mode(self) -> bool:
        """Check if edit mode is active."""
        return self._panel_editor.edit_mode

    @property
    def panels_modified_flag(self) -> bool:
        """Check if panels were modified."""
        return self._panel_editor.modified

    def get_edited_panels(self) -> List[QRectF]:
        """Get current edited panels."""
        return self._panel_editor.get_panels()

    def set_edit_mode(self, enabled: bool) -> None:
        """Set edit mode from main window."""
        if enabled:
            self._panel_editor.set_panels(self._overlay_rects)
        self._panel_editor.edit_mode = enabled
        self.viewport().update()

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

    def _view_to_page_pt(self, view_pos) -> QPointF:
        """Convert viewport position to page point coordinates.

        Args:
            view_pos: Position in viewport pixels (QPoint or QPointF)

        Returns:
            Position in page point coordinates
        """
        doc = self.document()
        if not doc:
            return QPointF(0, 0)

        cur = self.pageNavigator().currentPage()
        if cur < 0 or cur >= doc.pageCount():
            return QPointF(0, 0)

        page_pts = doc.pagePointSize(cur)
        z = self._effective_zoom_factor()
        if z <= 0:
            return QPointF(0, 0)

        # Get viewport dimensions and margins
        vw = self.viewport().width()
        vh = self.viewport().height()
        margins = self.documentMargins()
        spacing = self.pageSpacing()

        # Calculate page size in pixels
        page_w = page_pts.width() * z
        page_h = page_pts.height() * z

        # Calculate page position
        zoom_mode = self.zoomMode()

        if self.pageMode() == QPdfView.PageMode.SinglePage:
            avail_w = vw - margins.left() - margins.right()
            avail_h = vh - margins.top() - margins.bottom()

            if zoom_mode == QPdfView.ZoomMode.FitInView:
                page_x = margins.left() + max(0.0, (avail_w - page_w) / 2.0)
                page_y = margins.top() + max(0.0, (avail_h - page_h) / 2.0)
            else:
                page_x = margins.left() + max(0.0, (avail_w - page_w) / 2.0)
                page_y = margins.top()
        else:
            avail_w = vw - margins.left() - margins.right()
            page_x = margins.left() + max(0.0, (avail_w - page_w) / 2.0)
            page_y = margins.top()
            for i in range(cur):
                prev_pts = doc.pagePointSize(i)
                page_y += prev_pts.height() * z + spacing

        # Apply scroll offset
        page_x -= self.horizontalScrollBar().value()
        page_y -= self.verticalScrollBar().value()

        # Convert viewport position to page coordinates
        view_x = view_pos.x() if hasattr(view_pos, 'x') else view_pos.x()
        view_y = view_pos.y() if hasattr(view_pos, 'y') else view_pos.y()

        pt_x = (view_x - page_x) / z
        pt_y = (view_y - page_y) / z

        return QPointF(pt_x, pt_y)

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

            # Draw edit mode handles
            if self._panel_editor.edit_mode:
                self._draw_edit_handles(painter)

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
        selected_idx = self._panel_editor.selected_index if self._panel_editor.edit_mode else -1

        for idx, r in enumerate(self._overlay_rects):
            if r.isEmpty():
                continue

            vr = self._page_rect_to_view(r)

            # Different style for selected panel in edit mode
            if idx == selected_idx:
                pen = QPen(QColor(255, 165, 0, 255), 3)  # Orange, thicker
                fill = QColor(255, 165, 0, 80)
            else:
                pen = QPen(QColor(0, 200, 0, 220), 2)
                fill = QColor(0, 200, 0, 55)

            painter.setPen(pen)
            painter.setBrush(fill)
            painter.drawRect(vr)

            # Draw panel index
            text_pen = QPen(QColor(0, 0, 0, 255), 1)
            painter.setPen(text_pen)
            painter.drawText(
                vr.adjusted(3, 3, -3, -3),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                f"#{idx + 1}"
            )

    def _draw_edit_handles(self, painter: QPainter) -> None:
        """Draw resize handles for selected panel in edit mode."""
        selected_idx = self._panel_editor.selected_index
        if selected_idx < 0 or selected_idx >= len(self._overlay_rects):
            return

        rect = self._overlay_rects[selected_idx]
        vr = self._page_rect_to_view(rect)

        # Handle styling
        handle_size = 10
        handle_pen = QPen(QColor(255, 255, 255, 255), 1)
        handle_fill = QBrush(QColor(255, 165, 0, 255))

        painter.setPen(handle_pen)
        painter.setBrush(handle_fill)

        # Corner handles
        corners = [
            QRectF(vr.left() - handle_size/2, vr.top() - handle_size/2, handle_size, handle_size),
            QRectF(vr.right() - handle_size/2, vr.top() - handle_size/2, handle_size, handle_size),
            QRectF(vr.left() - handle_size/2, vr.bottom() - handle_size/2, handle_size, handle_size),
            QRectF(vr.right() - handle_size/2, vr.bottom() - handle_size/2, handle_size, handle_size),
        ]
        for corner in corners:
            painter.drawRect(corner)

        # Edge handles
        edges = [
            QRectF(vr.center().x() - handle_size/2, vr.top() - handle_size/2, handle_size, handle_size),
            QRectF(vr.center().x() - handle_size/2, vr.bottom() - handle_size/2, handle_size, handle_size),
            QRectF(vr.left() - handle_size/2, vr.center().y() - handle_size/2, handle_size, handle_size),
            QRectF(vr.right() - handle_size/2, vr.center().y() - handle_size/2, handle_size, handle_size),
        ]
        for edge in edges:
            painter.drawRect(edge)

        # Draw edit mode indicator
        painter.setPen(QPen(QColor(255, 165, 0, 255), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        font = painter.font()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(10, 20, "Mode Édition (E: quitter, Suppr: supprimer, Shift+Clic: nouveau)")

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
