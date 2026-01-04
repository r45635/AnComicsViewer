"""Main window for AnComicsViewer.

Provides the primary application interface with:
- PDF document loading and navigation
- Panel detection integration
- Toolbar with navigation and zoom controls
- Settings menu with presets
"""

from __future__ import annotations

import os
import glob
import shutil
import time
import traceback
import subprocess
import json
from datetime import datetime
from typing import Optional, List
from concurrent.futures import Future
import hashlib

from PySide6.QtCore import Qt, QPointF, QSizeF, QTimer, QSize
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QToolBar,
    QToolButton,
    QWidget,
    QMenu,
)
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtCore import QRectF

from .config import DetectorConfig, AppConfig, PRESETS
from .detector import PanelDetector
from .pdf_view import PannablePdfView
from .dialogs import PanelTuningDialog
from .cache import PanelCache
from .image_utils import pdebug


class ComicsView(QMainWindow):
    """Main application window for AnComicsViewer.

    Provides a full-featured PDF comics reader with:
    - Panel detection and navigation
    - Multiple zoom modes
    - Configurable reading direction (LTR/RTL)
    - Preset configurations for different comic styles
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ComicsView — PDF Reader")
        self.resize(980, 1000)

        # Configuration
        self._app_config = AppConfig()
        self._detector_config = DetectorConfig(debug=self._app_config.debug_panels)

        # Core components
        self.document: Optional[QPdfDocument] = None
        self.view = PannablePdfView(self)
        self.setCentralWidget(self.view)

        # Detector with threading support
        self._detector = PanelDetector(self._detector_config)

        # State
        self._current_path: Optional[str] = None
        self._panel_mode = False
        self._panel_index = -1
        self._panel_cache = PanelCache(max_pages=100)

        # Background detection future
        self._detection_future: Optional[Future] = None

        # Drag & drop
        self.setAcceptDrops(True)

        # Build UI
        self._build_toolbar()
        self.setStatusBar(QStatusBar(self))
        self._update_status()

        # Connect signals
        self.view.pageNavigator().currentPageChanged.connect(self._on_page_changed)
        self.view.pageNavigator().currentPageChanged.connect(self._update_status)

        # Auto-load sample PDF if available
        self._auto_load_sample()

    def _auto_load_sample(self) -> None:
        """Auto-load a sample PDF from samples_PDF directory if available."""
        import glob
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        samples_dir = os.path.join(script_dir, "samples_PDF")
        
        if os.path.exists(samples_dir):
            pdf_files = glob.glob(os.path.join(samples_dir, "*.pdf"))
            if pdf_files:
                # Load the first PDF found
                QTimer.singleShot(100, lambda: self.load_pdf(pdf_files[0]))

    # ========== UI Building ==========

    def _build_toolbar(self) -> None:
        """Build the main toolbar."""
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        # Open
        self._add_btn(tb, "📂", "Open PDF (Ctrl+O)", self.action_open,
                     QKeySequence(QKeySequence.StandardKey.Open))
        self._add_separator(tb)

        # Navigation
        self._add_btn(tb, "⏮", "First page (Home)", self.nav_first,
                     QKeySequence(QKeySequence.StandardKey.MoveToStartOfDocument))
        self._add_btn(tb, "⏪", "Previous page (PgUp)", self.nav_prev)
        self._add_btn(tb, "⏩", "Next page (PgDn)", self.nav_next)
        self._add_btn(tb, "⏭", "Last page (End)", self.nav_last,
                     QKeySequence(QKeySequence.StandardKey.MoveToEndOfDocument))
        self._add_separator(tb)

        # Zoom
        self._add_btn(tb, "🔍+", "Zoom in (Ctrl++)", self.zoom_in,
                     QKeySequence(QKeySequence.StandardKey.ZoomIn))
        self._add_btn(tb, "🔍-", "Zoom out (Ctrl+-)", self.zoom_out,
                     QKeySequence(QKeySequence.StandardKey.ZoomOut))
        self._add_btn(tb, "📏", "Fit to width (Ctrl+1)", self.fit_width,
                     QKeySequence("Ctrl+1"))
        self._add_btn(tb, "🗎", "Fit to page (Ctrl+0)", self.fit_page,
                     QKeySequence("Ctrl+0"))
        self._add_separator(tb)

        # Debug export
        self._add_btn(tb, "💾", "Sauvegarder debug de la page", self._save_page_debug_assets)
        self._add_separator(tb)

        # Panels
        self._add_btn(tb, "▦", "Toggle panel overlay (Ctrl+2)", self.toggle_panels,
                     QKeySequence("Ctrl+2"))
        self._add_btn(tb, "◀", "Previous panel (Shift+N)", self.panel_prev,
                     QKeySequence("Shift+N"))
        self._add_btn(tb, "▶", "Next panel (N)", self.panel_next,
                     QKeySequence("N"))
        self._add_separator(tb)

        # Settings button with menu
        settings_btn = self._build_settings_menu()
        tb.addWidget(settings_btn)

    def _add_btn(
        self, tb: QToolBar, emoji: str, tooltip: str,
        slot, shortcut: Optional[QKeySequence] = None
    ) -> QToolButton:
        """Add a toolbar button."""
        btn = QToolButton()
        btn.setText(emoji)
        btn.setToolTip(tooltip)
        btn.setAutoRaise(True)
        btn.clicked.connect(slot)

        if shortcut:
            act = QAction(self)
            act.setShortcut(shortcut)
            act.triggered.connect(slot)
            self.addAction(act)

        tb.addWidget(btn)
        return btn

    def _add_separator(self, tb: QToolBar) -> None:
        """Add a toolbar separator."""
        sep = QWidget()
        sep.setFixedWidth(6)
        tb.addWidget(sep)

    def _build_settings_menu(self) -> QToolButton:
        """Build settings menu button."""
        btn = QToolButton()
        btn.setText("⚙️")
        btn.setToolTip("Panel settings & debug")
        btn.setAutoRaise(True)

        menu = QMenu(btn)

        # Debug toggle
        act_debug = menu.addAction("Debug logs")
        act_debug.setCheckable(True)
        act_debug.setChecked(self._app_config.debug_panels)
        act_debug.toggled.connect(self._on_toggle_debug)

        # Canny fallback
        act_canny = menu.addAction("Use Canny fallback")
        act_canny.setCheckable(True)
        act_canny.setChecked(self._detector_config.use_canny_fallback)
        act_canny.toggled.connect(self._on_toggle_canny)

        # RTL reading
        act_rtl = menu.addAction("Reading RTL (manga)")
        act_rtl.setCheckable(True)
        act_rtl.setChecked(self._detector_config.reading_rtl)
        act_rtl.toggled.connect(self._on_toggle_rtl)

        menu.addSeparator()

        # DPI settings
        dpi150 = menu.addAction("Detection DPI: 150")
        dpi200 = menu.addAction("Detection DPI: 200")
        dpi150.triggered.connect(lambda: self._set_det_dpi(150.0))
        dpi200.triggered.connect(lambda: self._set_det_dpi(200.0))

        menu.addSeparator()

        # Re-run detection
        rerun = menu.addAction("Re-run detection (this page)")
        rerun.triggered.connect(self._rerun_detection_current_page)
        rerun_all = menu.addAction("Re-run detection (all pages)")
        rerun_all.triggered.connect(self._rerun_detection_all)

        menu.addSeparator()

        # Advanced tuning
        adv = menu.addAction("Advanced tuning…")
        adv.triggered.connect(self._open_tuning_dialog)

        # Presets submenu
        presets_menu = menu.addMenu("Presets")
        for name in PRESETS:
            action = presets_menu.addAction(name)
            action.triggered.connect(lambda checked, n=name: self._apply_preset(n))

        menu.addSeparator()

        # Framing submenu
        self._build_framing_menu(menu)

        btn.setMenu(menu)
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        return btn

    def _build_framing_menu(self, parent_menu: QMenu) -> None:
        """Build panel framing submenu."""
        frame_menu = parent_menu.addMenu("Panel framing")

        act_fit = frame_menu.addAction("Fit (show context)")
        act_fill = frame_menu.addAction("Fill (hide neighbors)")
        act_center = frame_menu.addAction("Center Fit")

        for a in (act_fit, act_fill, act_center):
            a.setCheckable(True)
        act_fit.setChecked(self._app_config.panel_framing == "fit")
        act_fill.setChecked(self._app_config.panel_framing == "fill")
        act_center.setChecked(self._app_config.panel_framing == "center")

        def set_frame(mode: str) -> None:
            self._app_config.panel_framing = mode
            act_fit.setChecked(mode == "fit")
            act_fill.setChecked(mode == "fill")
            act_center.setChecked(mode == "center")
            self._refocus_current_panel()
            self.statusBar().showMessage(f"Panel framing: {mode}", 1500)

        act_fit.triggered.connect(lambda: set_frame("fit"))
        act_fill.triggered.connect(lambda: set_frame("fill"))
        act_center.triggered.connect(lambda: set_frame("center"))

        # Cycle shortcut (F)
        act_cycle = QAction(self)
        act_cycle.setShortcut("F")
        act_cycle.triggered.connect(lambda: set_frame(
            {"fit": "fill", "fill": "center", "center": "fit"}[self._app_config.panel_framing]
        ))
        self.addAction(act_cycle)

    # ========== Settings Handlers ==========

    def _on_toggle_debug(self, checked: bool) -> None:
        self._app_config.debug_panels = checked
        self._detector_config.debug = checked
        self._detector = PanelDetector(self._detector_config)
        self.statusBar().showMessage(f"Debug logs {'ON' if checked else 'OFF'}", 1500)

    def _on_toggle_canny(self, checked: bool) -> None:
        self._detector_config.use_canny_fallback = checked
        self._detector = PanelDetector(self._detector_config)
        self.statusBar().showMessage(f"Canny fallback {'ON' if checked else 'OFF'}", 1500)

    def _on_toggle_rtl(self, checked: bool) -> None:
        self._detector_config.reading_rtl = checked
        self._detector = PanelDetector(self._detector_config)
        self.statusBar().showMessage(f"Reading {'RTL' if checked else 'LTR'}", 1500)

    def _set_det_dpi(self, dpi: float) -> None:
        self._app_config.detection_dpi = dpi
        self.statusBar().showMessage(f"Detection DPI set to {int(dpi)}", 1500)

    def _rerun_detection_current_page(self) -> None:
        cur = self.view.pageNavigator().currentPage()
        self._panel_cache.invalidate_page(cur)
        self._ensure_panels(force=True)
        self.statusBar().showMessage("Detection re-run", 1500)

    def _rerun_detection_all(self) -> None:
        self._panel_cache.clear()
        self._panel_index = -1
        self._ensure_panels(force=True)
        self._update_overlay()
        self.statusBar().showMessage("Re-run detection completed", 1500)

    def _open_tuning_dialog(self) -> None:
        def on_apply(new_config: DetectorConfig, new_dpi: float) -> None:
            self._detector_config = new_config
            self._app_config.detection_dpi = new_dpi
            self._detector = PanelDetector(self._detector_config)
            self._panel_cache.clear()
            self._panel_index = -1
            self._ensure_panels(force=True)
            self._update_overlay()
            self.statusBar().showMessage("Panel tuning applied", 1500)

        dlg = PanelTuningDialog(
            self,
            self._detector_config,
            self._app_config.detection_dpi,
            on_apply=on_apply
        )
        dlg.exec()

    def _apply_preset(self, name: str) -> None:
        if name not in PRESETS:
            return

        det_config, app_config = PRESETS[name]
        self._detector_config = det_config.copy()
        self._detector_config.debug = self._app_config.debug_panels  # Preserve debug
        self._app_config.detection_dpi = app_config.detection_dpi
        self._app_config.panel_framing = app_config.panel_framing

        self._detector = PanelDetector(self._detector_config)
        self._panel_cache.clear()
        self._panel_index = -1
        self._ensure_panels(force=True)
        self._update_overlay()
        self.statusBar().showMessage(f"Preset '{name}' applied", 1500)

    # ========== File Operations ==========

    def action_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF", self._default_dir(), "PDF Files (*.pdf)"
        )
        if path:
            self.load_pdf(path)

    def _default_dir(self) -> str:
        if self._current_path and os.path.exists(self._current_path):
            return os.path.dirname(self._current_path)
        return os.path.expanduser("~")

    def load_pdf(self, path: str) -> bool:
        """Load a PDF file."""
        if not path or not os.path.exists(path) or not path.lower().endswith(".pdf"):
            QMessageBox.warning(self, "Invalid File", "Please select a valid PDF file.")
            return False

        # Release previous document
        if self.document is not None:
            try:
                # First disconnect from view
                self.view.setDocument(None)
                # Clear cache before closing document
                self._panel_cache.clear()
                # Close and delete
                self.document.close()
                self.document.deleteLater()
                self.document = None
            except Exception as e:
                pdebug(f"Error releasing document: {e}")
                self.document = None

        doc = QPdfDocument(self)
        err = doc.load(path)

        # Check for successful load
        success = self._check_load_success(err, doc)

        if not success or doc.pageCount() <= 0:
            QMessageBox.critical(self, "Load Error", "Failed to load PDF.")
            doc.deleteLater()
            return False

        self.document = doc
        self._current_path = path
        self.view.setDocument(self.document)
        self.setWindowTitle(f"ComicsView — {os.path.basename(path)}")

        self.fit_page()
        self._panel_cache.clear()
        self._panel_index = -1
        
        # If panel mode is already on, update after layout stabilizes
        if self._panel_mode:
            QTimer.singleShot(200, lambda: (
                self._ensure_panels(force=True),
                self._update_overlay()
            ))
        
        self._update_status()
        return True

    def _check_load_success(self, err, doc: QPdfDocument) -> bool:
        """Check if PDF loaded successfully (handles Qt version differences)."""
        try:
            if hasattr(QPdfDocument.Error, "None_") and err == QPdfDocument.Error.None_:
                return True
        except Exception:
            pass
        try:
            if hasattr(QPdfDocument.Error, "NoError") and err == QPdfDocument.Error.NoError:
                return True
        except Exception:
            pass
        if err == 0 or getattr(err, "value", 0) == 0:
            return True
        return False

    def export_current_page(self) -> None:
        """Export current page as PNG."""
        if not self.document or self.document.status() != self.document.Status.Ready:
            QMessageBox.warning(self, "Export Error", "No PDF loaded.")
            return

        page = self.view.pageNavigator().currentPage()
        base = os.path.splitext(os.path.basename(self._current_path or "page"))[0]
        filename = f"{base}_page_{page + 1:03d}.png"

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export Page as PNG",
            os.path.join(self._default_dir(), filename),
            "PNG Files (*.png)"
        )
        if not out_path:
            return

        # Render at 200 DPI
        pt = self.document.pagePointSize(page)
        scale = 200.0 / 72.0
        qsize = QSizeF(pt.width() * scale, pt.height() * scale).toSize()
        img = self.document.render(page, qsize)

        if img.isNull() or not img.save(out_path):
            QMessageBox.critical(self, "Export Error", "Failed to save image.")
            return

        self.statusBar().showMessage(f"Exported {os.path.basename(out_path)}", 2500)

    # ========== Navigation ==========

    def nav_first(self) -> None:
        if self.document:
            self.view.pageNavigator().jump(0, QPointF(0, 0))

    def nav_last(self) -> None:
        if self.document:
            self.view.pageNavigator().jump(self.document.pageCount() - 1, QPointF(0, 0))

    def nav_prev(self) -> None:
        if self.document:
            cur = self.view.pageNavigator().currentPage()
            if cur > 0:
                self.view.pageNavigator().jump(cur - 1, QPointF(0, 0))

    def nav_next(self) -> None:
        if self.document:
            cur = self.view.pageNavigator().currentPage()
            if cur < self.document.pageCount() - 1:
                self.view.pageNavigator().jump(cur + 1, QPointF(0, 0))

    # ========== Zoom ==========

    def zoom_in(self) -> None:
        if not self.document:
            return
        z = self.view.zoomFactor()
        self.view.setZoomMode(QPdfView.ZoomMode.Custom)
        self.view.setZoomFactor(min(16.0, z * 1.25))
        self._update_status()

    def zoom_out(self) -> None:
        if not self.document:
            return
        z = self.view.zoomFactor()
        self.view.setZoomMode(QPdfView.ZoomMode.Custom)
        self.view.setZoomFactor(max(0.05, z * 0.8))
        self._update_status()

    def fit_width(self) -> None:
        if self.document:
            self.view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
            self._update_status()

    def fit_page(self) -> None:
        if self.document:
            self.view.setZoomMode(QPdfView.ZoomMode.FitInView)
            self._update_status()

    # ========== Panel Detection & Navigation ==========

    def toggle_panels(self) -> None:
        """Toggle panel overlay mode."""
        if not self.document:
            return

        self._panel_mode = not self._panel_mode
        self.view.setPageMode(
            QPdfView.PageMode.SinglePage if self._panel_mode else QPdfView.PageMode.MultiPage
        )
        
        # In panel mode, force zoom to 1.0 to match detection coordinates
        if self._panel_mode:
            self.view.setZoomMode(QPdfView.ZoomMode.Custom)
            self.view.setZoomFactor(1.0)
        
        self._panel_index = -1
        self._ensure_panels(force=True)
        
        # Force Qt to process layout changes before updating overlay
        if self._panel_mode:
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()  # Process pending layout events
            QTimer.singleShot(100, self._update_overlay_delayed)
            # Always start with a full-page best-fit view when entering reading mode
            self._show_full_then_first_panel(delay_ms=0, auto_first=False)
        else:
            self._update_overlay()
        
        self.statusBar().showMessage(
            "Panel mode ON" if self._panel_mode else "Panel mode OFF", 2000
        )
    
    def _update_overlay_delayed(self) -> None:
        """Update overlay after a delay to ensure viewport is stable."""
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()  # Ensure layout is complete
        self._update_overlay()
        # Force viewport repaint
        self.view.viewport().update()

    def panel_next(self) -> None:
        """Navigate to next panel."""
        if not (self.document and self._panel_mode):
            return

        try:
            self._ensure_panels()
            cur = self.view.pageNavigator().currentPage()
            rects = self._panel_cache.get(cur) or []

            if not rects:
                self.statusBar().showMessage("No panels detected on this page", 2000)
                return

            if self._panel_index >= len(rects) - 1:
                # Auto-advance to next page when at last panel
                nav = self.view.pageNavigator()
                cur = nav.currentPage()
                if cur < self.document.pageCount() - 1:
                    nav.jump(cur + 1, QPointF(0, 0))
                    self._panel_index = -1  # will be set on page change
                else:
                    # Stay on last panel if no further pages
                    self._panel_index = len(rects) - 1
                    self._focus_panel(rects[self._panel_index])
            else:
                self._panel_index += 1
                self._focus_panel(rects[self._panel_index])
                pdebug(f"panel_next -> {self._panel_index + 1}/{len(rects)}")

        except Exception:
            pdebug(f"panel_next error:\n{traceback.format_exc()}")

    def panel_prev(self) -> None:
        """Navigate to previous panel."""
        if not (self.document and self._panel_mode):
            return

        try:
            self._ensure_panels()
            cur = self.view.pageNavigator().currentPage()
            rects = self._panel_cache.get(cur) or []

            if not rects:
                self.statusBar().showMessage("No panels detected on this page", 2000)
                return

            self._panel_index = (self._panel_index - 1) % len(rects)
            self._focus_panel(rects[self._panel_index])
            pdebug(f"panel_prev -> {self._panel_index + 1}/{len(rects)}")

        except Exception:
            pdebug(f"panel_prev error:\n{traceback.format_exc()}")

    def _ensure_panels(self, force: bool = False) -> None:
        """Ensure panels are detected for current page."""
        if not self.document:
            return

        cur = self.view.pageNavigator().currentPage()

        # Update config hash for cache validation
        config_hash = hash(str(self._detector_config.to_dict()))
        self._panel_cache.set_config_hash(config_hash)

        if not force and cur in self._panel_cache:
            return

        # Safety check: ensure document is still valid
        if not self.document or self.document.status() != self.document.Status.Ready:
            return

        try:
            pt = self.document.pagePointSize(cur)
            dpi = self._app_config.detection_dpi
            scale = dpi / 72.0
            qsize = QSizeF(pt.width() * scale, pt.height() * scale).toSize()
            qimg = self.document.render(cur, qsize)

            pdebug(f"Detection: page_points=({pt.width():.1f}x{pt.height():.1f}) dpi={dpi} scale={scale:.3f} img_size=({qimg.width()}x{qimg.height()})")
            
            rects = self._detector.detect_panels(qimg, pt)
            if rects:
                pdebug(f"First rect in page points: ({rects[0].left():.1f},{rects[0].top():.1f},{rects[0].width():.1f}x{rects[0].height():.1f})")
            
            self._panel_cache.put(cur, rects)
            pdebug(f"ensure_panels: page={cur}, panels={len(rects)} @ {int(dpi)} DPI")

        except Exception:
            pdebug(f"ensure_panels error:\n{traceback.format_exc()}")
            self._panel_cache.put(cur, [])

    def _update_overlay(self) -> None:
        """Update the panel overlay display."""
        cur = self.view.pageNavigator().currentPage()
        rects = self._panel_cache.get(cur) or []
        debug_info = self._detector.last_debug if self._app_config.debug_panels else None

        self.view.set_panel_overlay(
            rects,
            self._panel_mode,
            debug_info=debug_info,
            show_debug=self._app_config.debug_panels
        )

    def _focus_panel(self, rect: QRectF) -> None:
        """Zoom and scroll to focus on a panel."""
        try:
            if not self.document or not rect or rect.isEmpty():
                return

            vw = self.view.viewport().width()
            vh = self.view.viewport().height()
            mode = self._app_config.panel_framing

            # Calculate zoom
            if mode == "fill":
                z = max(vw / max(1e-6, rect.width()), vh / max(1e-6, rect.height()))
                z = min(16.0, max(0.05, z * 1.01))
            else:
                z = min(vw / max(1e-6, rect.width()), vh / max(1e-6, rect.height()))
                z = min(16.0, max(0.05, z * 0.96))

            self.view.setZoomMode(QPdfView.ZoomMode.Custom)
            self.view.setZoomFactor(z)

            # Scroll after zoom is applied
            QTimer.singleShot(0, lambda: self._scroll_to_panel(rect, mode))
            pdebug(f"focus idx={self._panel_index} -> zoom={z:.2f} ({mode})")

        except Exception:
            pdebug(f"focus_panel error:\n{traceback.format_exc()}")

    def _scroll_to_panel(self, rect: QRectF, mode: str, margin_px: int = 12) -> None:
        """Scroll viewport to show panel."""
        try:
            doc = self.document
            if not doc:
                return

            cur = self.view.pageNavigator().currentPage()
            page_pts = doc.pagePointSize(cur)
            z = self.view.zoomFactor()
            vw = self.view.viewport().width()
            vh = self.view.viewport().height()

            content_w = page_pts.width() * z
            content_h = page_pts.height() * z
            pad_x = max(0.0, (vw - content_w) / 2.0)
            pad_y = max(0.0, (vh - content_h) / 2.0)

            hbar = self.view.horizontalScrollBar()
            vbar = self.view.verticalScrollBar()

            if mode == "fill":
                vis_w_pt = vw / z
                vis_h_pt = vh / z
                x_pt = min(max(rect.left(), rect.right() - vis_w_pt),
                          rect.left() + max(0.0, rect.width() - vis_w_pt))
                y_pt = min(max(rect.top(), rect.bottom() - vis_h_pt),
                          rect.top() + max(0.0, rect.height() - vis_h_pt))
            elif mode == "center":
                x_pt = rect.center().x() - (vw / z) / 2.0
                y_pt = rect.center().y() - (vh / z) / 2.0
            else:  # fit
                x_pt = rect.left() - (margin_px / z)
                y_pt = rect.top() - (margin_px / z)

            target_x = int(max(0.0, pad_x + x_pt * z))
            target_y = int(max(0.0, pad_y + y_pt * z))

            if hbar:
                hbar.setValue(min(max(0, target_x), hbar.maximum()))
            if vbar:
                vbar.setValue(min(max(0, target_y), vbar.maximum()))

            self.view.viewport().update()

        except Exception:
            pdebug(f"scroll_to_panel error:\n{traceback.format_exc()}")

    def _refocus_current_panel(self) -> None:
        """Re-focus current panel with updated framing."""
        if not self._panel_mode:
            return

        cur = self.view.pageNavigator().currentPage()
        rects = self._panel_cache.get(cur) or []

        if rects and 0 <= self._panel_index < len(rects):
            self._focus_panel(rects[self._panel_index])

    # ========== Event Handlers ==========

    def _on_page_changed(self) -> None:
        """Handle page change events."""
        try:
            if self._panel_mode:
                self._panel_index = -1
                self._ensure_panels()
                self._update_overlay()
                # In reading mode: show full page best-fit, then auto-focus first panel after delay
                self._show_full_then_first_panel(delay_ms=2000, auto_first=True)
        except Exception:
            pdebug(f"page_changed error:\n{traceback.format_exc()}")

    def _show_full_then_first_panel(self, delay_ms: int = 2000, auto_first: bool = True) -> None:
        """Show full-page best fit, then optionally focus the first panel after a delay."""
        if not self.document:
            return
        cur = self.view.pageNavigator().currentPage()
        rects_cached = self._panel_cache.get(cur) or []
        debug_info = self._detector.last_debug if self._app_config.debug_panels else None

        self.view.setZoomMode(QPdfView.ZoomMode.FitInView)
        # Force layout to apply new zoom, then re-apply overlay to match scale
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception:
            pass

        if self.document and self.view.pageNavigator().currentPage() == cur:
            self.view.set_panel_overlay(
                rects_cached,
                self._panel_mode,
                debug_info=debug_info,
                show_debug=self._app_config.debug_panels,
            )
            self.view.viewport().update()

        # Small delayed refresh as safeguard
        QTimer.singleShot(50, self._update_overlay_delayed)
        self._update_status()

        if not auto_first:
            return

        rects = rects_cached
        if not rects:
            return

        def focus_first_panel() -> None:
            # Only focus if still on same page
            if not self.document or self.view.pageNavigator().currentPage() != cur:
                return
            self._panel_index = 0
            self._focus_panel(rects[0])

        QTimer.singleShot(delay_ms, focus_first_panel)

    def _save_page_debug_assets(self) -> None:
        """Run detection with debug enabled for current page and archive debug outputs."""
        if not self.document:
            self.statusBar().showMessage("Aucun document chargé", 2000)
            return

        cur = self.view.pageNavigator().currentPage()

        # Compute paths
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dbg_dir = os.path.join(root, "debug_output")
        os.makedirs(dbg_dir, exist_ok=True)

        # Render current page to PNG (original view) for archive
        try:
            pt = self.document.pagePointSize(cur)
            dpi = self._app_config.detection_dpi
            scale = dpi / 72.0
            qsize = QSizeF(pt.width() * scale, pt.height() * scale).toSize()
            img = self.document.render(cur, qsize)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = os.path.join(dbg_dir, f"page_{cur+1:03d}_{ts}")
            os.makedirs(dest, exist_ok=True)
            img.save(os.path.join(dest, "page_render.png"))
        except Exception:
            pdebug(f"[debug-save] render error:\n{traceback.format_exc()}")
            self.statusBar().showMessage("Erreur rendu page", 3000)
            return

        # Temporarily enable debug, rerun detection for this page
        prev_debug_cfg = self._detector_config.debug
        prev_debug_app = self._app_config.debug_panels
        config_snapshot = None
        try:
            self._detector_config.debug = True
            self._app_config.debug_panels = True
            config_snapshot = {
                "detector_config": self._detector_config.to_dict(),
                "app_config": {
                    "detection_dpi": self._app_config.detection_dpi,
                    "panel_framing": self._app_config.panel_framing,
                    "debug_panels": self._app_config.debug_panels,
                },
            }
            if hasattr(self._panel_cache, "invalidate_page"):
                self._panel_cache.invalidate_page(cur)
            self._ensure_panels(force=True)
        except Exception:
            pdebug(f"[debug-save] error:\n{traceback.format_exc()}")
            self.statusBar().showMessage("Erreur lors de la génération des debug", 3000)
            return
        finally:
            self._detector_config.debug = prev_debug_cfg
            self._app_config.debug_panels = prev_debug_app

        # Collect debug files produced in dbg_dir (files only in root)
        files = [f for f in glob.glob(os.path.join(dbg_dir, "*")) if os.path.isfile(f)]
        copied = 0
        for f in files:
            # Skip files already inside this dest
            if os.path.commonpath([dest, f]) == dest:
                continue
            try:
                shutil.copy2(f, dest)
                copied += 1
            except Exception:
                pdebug(f"[debug-save] copy error for {f}:\n{traceback.format_exc()}")

        # Write panel coordinates (based on page_render pixels)
        try:
            panel_txt = os.path.join(dest, "panels.txt")
            rects = self._panel_cache.get(cur) or []
            scale_px = scale  # dpi/72, same scale used to render page_render
            with open(panel_txt, "w", encoding="utf-8") as fh:
                fh.write(f"page_render_size_px: {img.width()}x{img.height()}\n")
                fh.write(f"page_points: {pt.width():.2f}x{pt.height():.2f}\n")
                fh.write(f"dpi: {dpi}\n")
                fh.write(f"full_page_rect_px: x=0, y=0, w={img.width()}, h={img.height()}\n")
                fh.write(f"panel_count: {len(rects)}\n")
                for idx, r in enumerate(rects, start=1):
                    x_px = int(round(r.left() * scale_px))
                    y_px = int(round(r.top() * scale_px))
                    w_px = int(round(r.width() * scale_px))
                    h_px = int(round(r.height() * scale_px))
                    fh.write(
                        f"panel_id: {idx} | rect_px: x={x_px}, y={y_px}, w={w_px}, h={h_px} | "
                        f"rect_points: x={r.left():.2f}, y={r.top():.2f}, w={r.width():.2f}, h={r.height():.2f}\n"
                    )
        except Exception:
            pdebug(f"[debug-save] panel write error:\n{traceback.format_exc()}")

        # Write info text
        info_path = os.path.join(dest, "info.txt")
        pdf_path = self._current_path or ""
        pdf_name = os.path.basename(pdf_path) if pdf_path else ""
        try:
            rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
        except Exception:
            rev = "unknown"
        with open(info_path, "w", encoding="utf-8") as fh:
            fh.write(f"pdf_name: {pdf_name}\n")
            fh.write(f"pdf_path: {pdf_path}\n")
            fh.write(f"page_number: {cur+1}\n")
            fh.write(f"timestamp: {datetime.now().isoformat()}\n")
            fh.write(f"project_root: {root}\n")
            fh.write(f"git_revision: {rev}\n")

        config_path = os.path.join(dest, "config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(
                    config_snapshot
                    or {
                        "detector_config": self._detector_config.to_dict(),
                        "app_config": {
                            "detection_dpi": self._app_config.detection_dpi,
                            "panel_framing": self._app_config.panel_framing,
                            "debug_panels": self._app_config.debug_panels,
                        },
                    },
                    fh,
                    indent=2,
                    sort_keys=True,
                )
        except Exception:
            pdebug(f"[debug-save] config write error:\n{traceback.format_exc()}")

        self.statusBar().showMessage(
            f"Debug sauvegardé: page {cur+1} ({copied} fichiers + page_render.png) -> {os.path.relpath(dest, root)}",
            5000,
        )

    def _update_status(self) -> None:
        """Update status bar."""
        try:
            if not self.document:
                self.statusBar().showMessage("No document loaded")
                return

            nav = self.view.pageNavigator()
            current = nav.currentPage() + 1
            total = self.document.pageCount()
            zoom = int(self.view.zoomFactor() * 100)
            self.statusBar().showMessage(f"Page {current}/{total} | Zoom {zoom}%")

        except Exception:
            self.statusBar().showMessage("Status update error")

    # ========== Drag & Drop ==========

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith('.pdf'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith('.pdf'):
                self.load_pdf(path)
                return
