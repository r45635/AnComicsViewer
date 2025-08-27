#!/usr/bin/env python3
"""
AnComicsViewer MINI — compact, robust, with Reading Mode
- PySide6 PDF viewer with optional YOLO overlays (panels/balloons)
- Reading Mode: navigate panels (and overlapping balloons) in reading order
- Centered framing: each target region is fit & centered in the window
- Configurable: enable/disable reading mode, show full page before first panel, L→R & top→bottom
- NEW: Multiple balloons attached to a panel are merged into a single viewing unit
- NEW: Double-click anywhere advances to the next unit when Reading Mode is ON
"""

from __future__ import annotations
import sys, os, json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QAction, QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QToolBar, QWidget, QVBoxLayout, QStatusBar
)

# Optional deps
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---------------- utils ----------------

def qimage_to_rgb(qimg: QImage) -> np.ndarray:
    if qimg.format() != QImage.Format.Format_RGBA8888:
        qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    h, w = qimg.height(), qimg.width()
    buf = bytes(qimg.constBits())
    arr = np.frombuffer(buf, dtype=np.uint8)[: h * qimg.bytesPerLine()].reshape(h, qimg.bytesPerLine())[:, : w * 4]
    return arr.reshape(h, w, 4)[:, :, :3].copy()

@dataclass
class Detection:
    cls: int
    conf: float
    rect: QRectF  # (x,y,w,h) in image coords

# ---------------- graphics ----------------
class BBoxItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, label: str, color: QColor, parent=None):
        super().__init__(rect, parent)
        pen = QPen(color); pen.setCosmetic(True); pen.setWidthF(1.2)
        self.setPen(pen); self.setBrush(Qt.NoBrush); self.setZValue(10)

class ClickableGraphicsView(QGraphicsView):
    def mouseDoubleClickEvent(self, event):
        # In Reading Mode, a double-click anywhere advances to next unit
        try:
            wnd = self.window()
            if getattr(wnd, 'cfg', None) and getattr(wnd.cfg, 'reading_mode', False):
                wnd.next_step(); return
        except Exception:
            pass
        super().mouseDoubleClickEvent(event)

# ---------------- config ----------------
@dataclass
class AppConfig:
    reading_mode: bool = True
    show_full_page_before_first_panel: bool = True
    direction: str = "LR_TB"  # Left-to-right, Top-to-bottom

# ---------------- main window ----------------
class PdfYoloViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnComicsViewer MINI")
        self.resize(1100, 800)

        self.scene = QGraphicsScene(self)
        self.view = ClickableGraphicsView(self.scene, self)
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setAlignment(Qt.AlignCenter)  # keep content centered

        central = QWidget(self); lay = QVBoxLayout(central); lay.setContentsMargins(0,0,0,0); lay.addWidget(self.view)
        self.setCentralWidget(central)
        self.status = QStatusBar(self); self.setStatusBar(self.status)

        self.pdf = None; self.page_index = 0
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.qimage_current: Optional[QImage] = None
        self.model = None
        self.class_names = ["panel", "balloon"]
        self.conf_thres, self.iou_thres, self.max_det = 0.15, 0.6, 200
        self.show_panels, self.show_balloons = True, True
        self.cfg = AppConfig()

        # reading state
        self.read_units: List[QRectF] = []
        self.read_index: int = -1
        self.fullpage_shown_on_page: bool = False

        self._build_ui()
        self._auto_load_model()
        self._auto_load_last_pdf()

    # ---------- UI ----------
    def _build_ui(self):
        tb = QToolBar("Main", self); tb.setMovable(False); self.addToolBar(tb)

        a_open = QAction("Open PDF…", self); a_open.triggered.connect(self.open_pdf); tb.addAction(a_open)
        a_model = QAction("Load model…", self); a_model.triggered.connect(self.load_model); tb.addAction(a_model)
        self.model_status = QAction("🔴 no model", self); self.model_status.setEnabled(False); tb.addAction(self.model_status)
        tb.addSeparator()
        a_prev = QAction("◀ Prev", self); a_prev.triggered.connect(self.prev_step); tb.addAction(a_prev)
        a_next = QAction("Next ▶", self); a_next.triggered.connect(self.next_step); tb.addAction(a_next)
        tb.addSeparator()
        a_reset = QAction("Reset", self); a_reset.triggered.connect(self.reset_view); tb.addAction(a_reset)
        tb.addSeparator()
        a_pan = QAction("Panels", self); a_pan.setCheckable(True); a_pan.setChecked(True); a_pan.toggled.connect(self._toggle_panels); tb.addAction(a_pan)
        a_bal = QAction("Balloons", self); a_bal.setCheckable(True); a_bal.setChecked(True); a_bal.toggled.connect(self._toggle_balloons); tb.addAction(a_bal)
        tb.addSeparator()
        a_read = QAction("Reading Mode", self); a_read.setCheckable(True); a_read.setChecked(self.cfg.reading_mode); a_read.toggled.connect(self._toggle_reading_mode); tb.addAction(a_read)
        a_full = QAction("Full page before 1st", self); a_full.setCheckable(True); a_full.setChecked(self.cfg.show_full_page_before_first_panel); a_full.toggled.connect(self._toggle_fullpage_pref); tb.addAction(a_full)
        a_dir = QAction("Direction L→R, T→B", self); a_dir.setCheckable(True); a_dir.setChecked(True); a_dir.toggled.connect(self._toggle_direction); tb.addAction(a_dir)

    # ---------- PDF ----------
    def open_pdf(self):
        if fitz is None:
            QMessageBox.critical(self, "Missing PyMuPDF", "pip install pymupdf"); return
        path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF (*.pdf)")
        if not path: return
        try:
            self.pdf = fitz.open(path); self.page_index = 0
            self.status.showMessage(f"PDF: {os.path.basename(path)} • {len(self.pdf)} pages")
            self.load_page(self.page_index)
            self._save_session()
        except Exception as e:
            QMessageBox.critical(self, "PDF error", str(e))

    def load_page(self, index: int):
        if not self.pdf or index < 0 or index >= len(self.pdf): return
        self.page_index = index
        page = self.pdf[index]
        dpi = 300; zoom = dpi / 72.0  # Increased DPI for better detection
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888).copy()
        self._set_page_image(qimg)
        self._run_detection()
        self._prepare_reading_units()
        self.fullpage_shown_on_page = False
        if self.cfg.reading_mode:
            if self.cfg.show_full_page_before_first_panel:
                self.fit_full_page(); self.read_index = -1  # wait for Next
            else:
                self.read_index = -1; self.next_step()  # auto jump to first panel
        self._save_session()

    def next_page(self):
        if self.pdf and self.page_index + 1 < len(self.pdf): self.load_page(self.page_index + 1)

    def prev_page(self):
        if self.pdf and self.page_index - 1 >= 0: self.load_page(self.page_index - 1)

    # ---------- image & overlays ----------
    def _set_page_image(self, qimg: QImage):
        self.scene.clear(); self.qimage_current = qimg
        pix = QPixmap.fromImage(qimg); pix.setDevicePixelRatio(1.0)
        self.pixmap_item = QGraphicsPixmapItem(pix); self.pixmap_item.setZValue(0); self.scene.addItem(self.pixmap_item)
        self.reset_view()

    def _clear_overlays(self):
        if not self.pixmap_item: return
        for ch in list(self.pixmap_item.childItems()): self.scene.removeItem(ch)

    def _draw_detections(self, dets: List[Detection]):
        if not self.pixmap_item: return
        self._clear_overlays()
        for d in dets:
            c = QColor(35,197,83) if d.cls == 0 else QColor(41,121,255)
            BBoxItem(d.rect, "", c, parent=self.pixmap_item)

    # ---------- YOLO ----------
    def _auto_load_model(self):
        if YOLO is None: self.status.showMessage("YOLO unavailable – detection off"); return
        preferred = "anComicsViewer_v01.pt"
        if os.path.exists(preferred):
            try: self._load_model(preferred); return
            except Exception: pass
        self.model_status.setText("🔴 no model")

    def load_model(self):
        if YOLO is None:
            QMessageBox.critical(self, "Missing Ultralytics", "pip install ultralytics"); return
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO model", "", "PT (*.pt)")
        if not path: return
        try:
            self._load_model(path); self.status.showMessage(f"Model: {os.path.basename(path)}")
            if self.qimage_current is not None: self._run_detection()
        except Exception as e:
            QMessageBox.critical(self, "Model error", str(e)); self.model_status.setText("🔴 model error")

    def _load_model(self, path: str):
        import torch
        orig = torch.load
        def patched(f, map_location=None, pickle_module=None, weights_only=None, **kw):
            return orig(f, map_location=map_location, pickle_module=pickle_module, weights_only=False if weights_only is None else weights_only, **kw)
        torch.load = patched
        try:
            self.model = YOLO(path)
        finally:
            torch.load = orig
        self.model_status.setText(f"🟢 {os.path.basename(path)}")

    def _run_detection(self):
        self.dets: List[Detection] = []
        if self.qimage_current is None or self.model is None:
            return

        # --- class-wise thresholds & params ---
        PANEL_CONF = 0.12    # more permissive for panels
        BAL_CONF   = 0.25    # a bit stricter for balloons
        IOU_MERGE  = 0.60    # merge boxes of same class if overlap is high

        rgb = qimage_to_rgb(self.qimage_current)
        H, W = rgb.shape[:2]

        # dynamic imgsz (multiple of 32), capped to avoid OOM
        max_side = max(H, W)
        imgsz = min(1536, max(960, max_side // 2))
        imgsz -= imgsz % 32

        def predict_once(img):
            res = self.model.predict(
                source=img,
                imgsz=imgsz,
                conf=min(PANEL_CONF, BAL_CONF),  # use lower of the two; filter per class after
                iou=0.6,
                max_det=300,
                augment=True,                    # TTA helps on gutters/pale areas
                verbose=False,
                device="mps" if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK","1") else None
            )[0]
            out = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls  = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                for (x1,y1,x2,y2), c, p in zip(xyxy, cls, conf):
                    # class-wise threshold filtering
                    if c == 0 and p < PANEL_CONF:  continue
                    if c == 1 and p < BAL_CONF:    continue
                    out.append((c, float(p), QRectF(float(x1), float(y1), float(x2-x1), float(y2-y1))))
            return out

        # pass 1: raw image
        dets_accum = predict_once(rgb)

        # pass 2: slight gamma tweak (helps faint borders/text)
        try:
            g = np.clip((rgb / 255.0) ** 0.9, 0, 1.0)
            g = (g * 255).astype(np.uint8)
            dets_accum += predict_once(g)
        except Exception:
            pass

        # merge by IoU (class-consistent)
        def iou(a: QRectF, b: QRectF) -> float:
            inter = a.intersected(b)
            if inter.isEmpty(): return 0.0
            ia = inter.width() * inter.height()
            ua = a.width()*a.height() + b.width()*b.height() - ia
            return ia / max(ua, 1e-6)

        merged = []
        for c, p, r in sorted(dets_accum, key=lambda x: x[1], reverse=True):
            keep = True
            for j, (cj, pj, rj) in enumerate(merged):
                if c == cj and iou(r, rj) > IOU_MERGE:
                    merged[j] = (cj, max(pj, p), r.united(rj))
                    keep = False
                    break
            if keep:
                merged.append((c, p, r))

        # panel shape sanity: drop tiny/noisy/extreme aspect boxes
        for c, p, r in merged:
            if c == 0:
                if r.width()*r.height() < 0.0025 * (H * W):  # <0.25% of page
                    continue
                ar = r.width() / max(r.height(), 1)
                if ar < 0.25 or ar > 4.0:
                    continue
            self.dets.append(Detection(c, p, r))

        # (Optional) Minimum recall fallback
        if not any(d.cls == 0 for d in self.dets):
            bal_top = [d.rect for d in self.dets if d.cls == 1 and d.rect.center().y() <= H * 0.4]
            if bal_top:
                u = QRectF(bal_top[0])
                for b in bal_top[1:]:
                    u = u.united(b)
                self.dets.insert(0, Detection(0, 0.5, u))  # provisional panel

        self._draw_detections(self.dets)
        self.status.showMessage(f"Page {self.page_index+1}: {len(self.dets)} detections (imgsz={imgsz})")

    # ---------- reading mode ----------
    def _prepare_reading_units(self):
        """Build reading units so EACH PANEL + ALL attached balloons are a single frame.
        Standalone balloons (not attached to any panel) become independent frames.
        """
        self.read_units = []
        if not getattr(self, 'dets', None):
            self.read_index = -1
            return

        panels = [d.rect for d in self.dets if d.cls == 0]
        balloons = [d.rect for d in self.dets if d.cls == 1]
        assigned = set()

        def attached(b: QRectF, base: QRectF) -> bool:
            # Consider attached if:
            # 1) balloon center is inside a slightly expanded panel, OR
            # 2) Intersection covers ≥5% of the balloon area
            expanded = base.adjusted(-0.05*base.width(), -0.05*base.height(), 0.05*base.width(), 0.05*base.height())
            if expanded.contains(b.center()):
                return True
            inter = base.intersected(b)
            if inter.isEmpty():
                return False
            return (inter.width()*inter.height()) >= 0.05*(b.width()*b.height())

        # For each panel, iteratively union with ALL attached balloons (handles multiple balloons)
        for p in panels:
            merged = QRectF(p)
            changed = True
            while changed:
                changed = False
                for i, b in enumerate(balloons):
                    # Attach if connected to current merged region
                    if attached(b, merged):
                        u = merged.united(b)
                        if u != merged:
                            merged = u; changed = True
                        assigned.add(i)
            self.read_units.append(merged)

        # Standalone balloons → own units
        for i, b in enumerate(balloons):
            if i not in assigned:
                self.read_units.append(QRectF(b))

        # Sort reading order (L→R, T→B)
        self.read_units = self._sort_reading_order(self.read_units, self.cfg.direction)
        self.read_index = -1

    @staticmethod
    def _sort_reading_order(rects: List[QRectF], direction: str) -> List[QRectF]:
        if not rects: return []
        rects = sorted(rects, key=lambda r: (r.top(), r.left()))
        rows: List[List[QRectF]] = []
        for r in rects:
            placed = False
            for row in rows:
                ref = row[0]
                if abs(r.center().y() - ref.center().y()) <= max(ref.height(), r.height())*0.4:
                    row.append(r); placed = True; break
            if not placed: rows.append([r])
        for row in rows:
            row.sort(key=lambda x: x.left())
        out: List[QRectF] = []
        for row in rows: out.extend(row)
        return out

    def next_step(self):
        if not self.cfg.reading_mode:
            self.next_page(); return
        if not self.read_units:
            self.next_page(); return
        if not self.fullpage_shown_on_page and self.cfg.show_full_page_before_first_panel and self.read_index < 0:
            self.fit_full_page(); self.fullpage_shown_on_page = True; return
        self.read_index += 1
        if self.read_index >= len(self.read_units):
            self.next_page(); return
        self._focus_rect(self.read_units[self.read_index])

    def prev_step(self):
        if not self.cfg.reading_mode:
            self.prev_page(); return
        if not self.read_units:
            self.prev_page(); return
        if self.read_index <= 0:
            self.read_index = -1
            if self.cfg.show_full_page_before_first_panel:
                self.fit_full_page(); self.fullpage_shown_on_page = True
            else:
                self.prev_page()
            return
        self.read_index -= 1
        self._focus_rect(self.read_units[self.read_index])

    def _focus_rect(self, r: QRectF):
        if not self.pixmap_item: return
        pad = 0.06
        rr = QRectF(r)
        rr.adjust(-r.width()*pad, -r.height()*pad, r.width()*pad, r.height()*pad)
        rr = rr.intersected(self.pixmap_item.boundingRect())
        self.view.fitInView(self.pixmap_item.mapRectToScene(rr), Qt.KeepAspectRatio)
        self.view.centerOn(self.pixmap_item.mapRectToScene(rr).center())

    def fit_full_page(self):
        if not self.pixmap_item: return
        self.view.fitInView(self.pixmap_item.sceneBoundingRect(), Qt.KeepAspectRatio)
        self.view.centerOn(self.pixmap_item)

    # ---------- zoom/reset ----------
    def reset_view(self):
        self.view.resetTransform()
        self.view.centerOn(self.pixmap_item)

    # ---------- toggles ----------
    def _toggle_panels(self, checked: bool):
        self.show_panels = checked; self._run_detection(); self._prepare_reading_units()

    def _toggle_balloons(self, checked: bool):
        self.show_balloons = checked; self._run_detection(); self._prepare_reading_units()

    def _toggle_reading_mode(self, checked: bool):
        self.cfg.reading_mode = checked
        if checked:
            self._prepare_reading_units()
            if self.cfg.show_full_page_before_first_panel:
                self.fit_full_page(); self.read_index = -1; self.fullpage_shown_on_page = True
            else:
                self.read_index = -1; self.next_step()

    def _toggle_fullpage_pref(self, checked: bool):
        self.cfg.show_full_page_before_first_panel = checked

    def _toggle_direction(self, checked: bool):
        self.cfg.direction = "LR_TB" if checked else "TB_LR"

    # ---------- session ----------
    def _cfg_path(self) -> str:
        return os.path.expanduser("~/.ancomicsviewer_session.json")

    def _save_session(self):
        if not self.pdf: return
        try:
            with open(self._cfg_path(), 'w', encoding='utf-8') as f:
                json.dump({"last_pdf": getattr(self.pdf, 'name', None), "last_page": self.page_index}, f)
        except Exception: pass

    def _load_session(self) -> dict:
        try:
            p = self._cfg_path()
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception: pass
        return {}

    def _auto_load_last_pdf(self):
        if fitz is None: return
        s = self._load_session(); last = s.get('last_pdf'); idx = s.get('last_page', 0)
        if not last or not os.path.isfile(last): return
        try:
            self.pdf = fitz.open(last)
            self.page_index = idx if 0 <= idx < len(self.pdf) else 0
            self.status.showMessage(f"Reopened: {os.path.basename(last)} • {len(self.pdf)} pages")
            self.load_page(self.page_index)
        except Exception:
            try: os.remove(self._cfg_path())
            except Exception: pass

# ---------------- main ----------------

def main():
    app = QApplication(sys.argv)
    w = PdfYoloViewer(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
