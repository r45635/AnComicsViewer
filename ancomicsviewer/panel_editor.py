"""Panel editor widget for manual panel adjustment.

Allows users to:
- Drag panel corners to resize
- Create new panels by drawing
- Delete panels with keyboard shortcuts
- Save corrections for reuse
"""

from __future__ import annotations

import json
import os
from typing import Optional, List, Tuple, Callable
from enum import Enum, auto

from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QSizeF
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QCursor, QMouseEvent, QKeyEvent
from PySide6.QtWidgets import QWidget


class EditHandle(Enum):
    """Handle positions for panel resizing."""
    NONE = auto()
    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()
    MOVE = auto()


class PanelEditor:
    """Panel editing logic and state management.
    
    This class handles the editing logic without being a widget itself,
    so it can be integrated into PannablePdfView.
    """
    
    # Handle size in pixels
    HANDLE_SIZE = 12
    HANDLE_HALF = 6
    
    def __init__(self):
        self._panels: List[QRectF] = []
        self._original_panels: List[QRectF] = []  # For undo
        self._selected_index: int = -1
        self._hovered_handle: EditHandle = EditHandle.NONE
        self._dragging: bool = False
        self._drag_start: QPointF = QPointF()
        self._rect_start: QRectF = QRectF()
        self._drawing_new: bool = False
        self._draw_start: QPointF = QPointF()
        self._edit_mode: bool = False
        self._modified: bool = False
        
        # Callbacks
        self._on_panels_changed: Optional[Callable[[List[QRectF]], None]] = None
        self._on_edit_mode_changed: Optional[Callable[[bool], None]] = None
    
    def set_callbacks(
        self,
        on_panels_changed: Optional[Callable[[List[QRectF]], None]] = None,
        on_edit_mode_changed: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Set callback functions."""
        self._on_panels_changed = on_panels_changed
        self._on_edit_mode_changed = on_edit_mode_changed
    
    def set_panels(self, panels: List[QRectF]) -> None:
        """Set the panels to edit."""
        self._panels = [QRectF(p) for p in panels]
        self._original_panels = [QRectF(p) for p in panels]
        self._modified = False
        self._selected_index = -1
    
    def get_panels(self) -> List[QRectF]:
        """Get current panels."""
        return self._panels
    
    @property
    def edit_mode(self) -> bool:
        """Check if edit mode is active."""
        return self._edit_mode
    
    @edit_mode.setter
    def edit_mode(self, value: bool) -> None:
        """Set edit mode."""
        if self._edit_mode != value:
            self._edit_mode = value
            if not value:
                self._selected_index = -1
                self._dragging = False
                self._drawing_new = False
            if self._on_edit_mode_changed:
                self._on_edit_mode_changed(value)
    
    @property
    def modified(self) -> bool:
        """Check if panels were modified."""
        return self._modified
    
    @property
    def selected_index(self) -> int:
        """Get selected panel index."""
        return self._selected_index
    
    def _notify_changed(self) -> None:
        """Notify that panels changed."""
        self._modified = True
        if self._on_panels_changed:
            self._on_panels_changed(self._panels)
    
    def undo(self) -> None:
        """Undo all changes."""
        self._panels = [QRectF(p) for p in self._original_panels]
        self._selected_index = -1
        self._modified = False
        self._notify_changed()
    
    def delete_selected(self) -> bool:
        """Delete selected panel."""
        if 0 <= self._selected_index < len(self._panels):
            del self._panels[self._selected_index]
            self._selected_index = -1
            self._notify_changed()
            return True
        return False
    
    def select_panel_at(self, pos: QPointF, zoom: float = 1.0) -> bool:
        """Select panel at position.
        
        Args:
            pos: Position in page points
            zoom: Current zoom factor
            
        Returns:
            True if a panel was selected
        """
        # Check from last to first (topmost first)
        for i in range(len(self._panels) - 1, -1, -1):
            if self._panels[i].contains(pos):
                self._selected_index = i
                return True
        
        self._selected_index = -1
        return False
    
    def get_handle_at(self, pos: QPointF, zoom: float = 1.0) -> EditHandle:
        """Get the resize handle at position.
        
        Args:
            pos: Position in page points
            zoom: Current zoom factor
            
        Returns:
            Handle at position or NONE
        """
        if self._selected_index < 0 or self._selected_index >= len(self._panels):
            return EditHandle.NONE
        
        rect = self._panels[self._selected_index]
        handle_size_pt = self.HANDLE_SIZE / zoom
        
        handles = {
            EditHandle.TOP_LEFT: QPointF(rect.left(), rect.top()),
            EditHandle.TOP_RIGHT: QPointF(rect.right(), rect.top()),
            EditHandle.BOTTOM_LEFT: QPointF(rect.left(), rect.bottom()),
            EditHandle.BOTTOM_RIGHT: QPointF(rect.right(), rect.bottom()),
            EditHandle.TOP: QPointF(rect.center().x(), rect.top()),
            EditHandle.BOTTOM: QPointF(rect.center().x(), rect.bottom()),
            EditHandle.LEFT: QPointF(rect.left(), rect.center().y()),
            EditHandle.RIGHT: QPointF(rect.right(), rect.center().y()),
        }
        
        for handle, center in handles.items():
            if abs(pos.x() - center.x()) <= handle_size_pt and \
               abs(pos.y() - center.y()) <= handle_size_pt:
                return handle
        
        # Check if inside rect (for move)
        if rect.contains(pos):
            return EditHandle.MOVE
        
        return EditHandle.NONE
    
    def start_drag(self, pos: QPointF, handle: EditHandle) -> None:
        """Start dragging a handle."""
        if self._selected_index < 0 or self._selected_index >= len(self._panels):
            return
        
        self._dragging = True
        self._hovered_handle = handle
        self._drag_start = QPointF(pos)
        self._rect_start = QRectF(self._panels[self._selected_index])
    
    def update_drag(self, pos: QPointF) -> None:
        """Update drag position."""
        if not self._dragging or self._selected_index < 0:
            return
        
        delta = pos - self._drag_start
        rect = QRectF(self._rect_start)
        
        handle = self._hovered_handle
        
        if handle == EditHandle.MOVE:
            rect.translate(delta)
        elif handle == EditHandle.TOP_LEFT:
            rect.setTopLeft(rect.topLeft() + delta)
        elif handle == EditHandle.TOP_RIGHT:
            rect.setTopRight(rect.topRight() + delta)
        elif handle == EditHandle.BOTTOM_LEFT:
            rect.setBottomLeft(rect.bottomLeft() + delta)
        elif handle == EditHandle.BOTTOM_RIGHT:
            rect.setBottomRight(rect.bottomRight() + delta)
        elif handle == EditHandle.TOP:
            rect.setTop(rect.top() + delta.y())
        elif handle == EditHandle.BOTTOM:
            rect.setBottom(rect.bottom() + delta.y())
        elif handle == EditHandle.LEFT:
            rect.setLeft(rect.left() + delta.x())
        elif handle == EditHandle.RIGHT:
            rect.setRight(rect.right() + delta.x())
        
        # Normalize rect (ensure width/height positive)
        rect = rect.normalized()
        
        # Minimum size
        if rect.width() >= 10 and rect.height() >= 10:
            self._panels[self._selected_index] = rect
    
    def end_drag(self) -> None:
        """End dragging."""
        if self._dragging:
            self._dragging = False
            if self._rect_start != self._panels[self._selected_index]:
                self._notify_changed()
    
    def start_new_panel(self, pos: QPointF) -> None:
        """Start drawing a new panel."""
        self._drawing_new = True
        self._draw_start = QPointF(pos)
        
        # Add a temporary panel
        self._panels.append(QRectF(pos.x(), pos.y(), 1, 1))
        self._selected_index = len(self._panels) - 1
    
    def update_new_panel(self, pos: QPointF) -> None:
        """Update the new panel being drawn."""
        if not self._drawing_new or self._selected_index < 0:
            return
        
        rect = QRectF(self._draw_start, pos).normalized()
        self._panels[self._selected_index] = rect
    
    def end_new_panel(self) -> bool:
        """End drawing a new panel.
        
        Returns:
            True if panel was created (large enough)
        """
        if not self._drawing_new or self._selected_index < 0:
            return False
        
        self._drawing_new = False
        rect = self._panels[self._selected_index]
        
        # Remove if too small
        if rect.width() < 20 or rect.height() < 20:
            del self._panels[self._selected_index]
            self._selected_index = -1
            return False
        
        self._notify_changed()
        return True
    
    def get_cursor_for_handle(self, handle: EditHandle) -> QCursor:
        """Get appropriate cursor for handle."""
        cursors = {
            EditHandle.NONE: Qt.CursorShape.ArrowCursor,
            EditHandle.TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
            EditHandle.BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
            EditHandle.TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
            EditHandle.BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
            EditHandle.TOP: Qt.CursorShape.SizeVerCursor,
            EditHandle.BOTTOM: Qt.CursorShape.SizeVerCursor,
            EditHandle.LEFT: Qt.CursorShape.SizeHorCursor,
            EditHandle.RIGHT: Qt.CursorShape.SizeHorCursor,
            EditHandle.MOVE: Qt.CursorShape.SizeAllCursor,
        }
        return QCursor(cursors.get(handle, Qt.CursorShape.ArrowCursor))
    
    @property
    def is_dragging(self) -> bool:
        """Check if currently dragging."""
        return self._dragging
    
    @property
    def is_drawing(self) -> bool:
        """Check if currently drawing new panel."""
        return self._drawing_new


class PanelCorrections:
    """Manages saved panel corrections per PDF/page."""
    
    def __init__(self, corrections_dir: Optional[str] = None):
        if corrections_dir is None:
            # Default to user data directory
            corrections_dir = os.path.expanduser("~/.ancomicsviewer/corrections")
        self._corrections_dir = corrections_dir
        os.makedirs(corrections_dir, exist_ok=True)
    
    def _get_key(self, pdf_path: str, page_num: int) -> str:
        """Generate a key for a PDF page."""
        import hashlib
        pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:12]
        return f"{pdf_hash}_page{page_num:04d}"
    
    def _get_path(self, key: str) -> str:
        """Get file path for a correction."""
        return os.path.join(self._corrections_dir, f"{key}.json")
    
    def save(
        self,
        pdf_path: str,
        page_num: int,
        panels: List[QRectF],
        page_size: QSizeF,
    ) -> bool:
        """Save panel corrections.
        
        Args:
            pdf_path: Path to PDF
            page_num: Page number (0-indexed)
            panels: List of panel rectangles in page points
            page_size: Page size in points
            
        Returns:
            True if saved successfully
        """
        key = self._get_key(pdf_path, page_num)
        path = self._get_path(key)
        
        data = {
            "pdf_path": pdf_path,
            "page_num": page_num,
            "page_width": page_size.width(),
            "page_height": page_size.height(),
            "panels": [
                {
                    "x": r.x(),
                    "y": r.y(),
                    "width": r.width(),
                    "height": r.height(),
                }
                for r in panels
            ],
        }
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False
    
    def load(self, pdf_path: str, page_num: int) -> Optional[List[QRectF]]:
        """Load saved panel corrections.
        
        Args:
            pdf_path: Path to PDF
            page_num: Page number (0-indexed)
            
        Returns:
            List of panel rectangles or None if not found
        """
        key = self._get_key(pdf_path, page_num)
        path = self._get_path(key)
        
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            panels = []
            for p in data.get("panels", []):
                panels.append(QRectF(
                    p["x"],
                    p["y"],
                    p["width"],
                    p["height"],
                ))
            return panels
        except Exception:
            return None
    
    def delete(self, pdf_path: str, page_num: int) -> bool:
        """Delete saved corrections.
        
        Args:
            pdf_path: Path to PDF
            page_num: Page number (0-indexed)
            
        Returns:
            True if deleted
        """
        key = self._get_key(pdf_path, page_num)
        path = self._get_path(key)
        
        if os.path.exists(path):
            try:
                os.remove(path)
                return True
            except Exception:
                pass
        return False
    
    def has_corrections(self, pdf_path: str, page_num: int) -> bool:
        """Check if corrections exist for a page."""
        key = self._get_key(pdf_path, page_num)
        path = self._get_path(key)
        return os.path.exists(path)
