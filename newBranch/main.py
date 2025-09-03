#!/usr/bin/env python3
"""
AnComicsViewer MINI — compact, robust, with Reading Mode
- PySide6 PDF viewer with optional YOLO overlays (panels/balloons)
- Reading Mode: navigate panels (and overlapping balloons) in reading order
- Centered framing: each target region is fit & centered in the w            res = self.model.predict(
                source=img, imgsz=imgsz, conf=min(PANEL_CONF, BAL_CONF),
                iou=0.6, max_det=MAX_DET, augment=False, verbose=False,
                device="mps" if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "1") else None
            )[0]
- Configurable: enable/disable reading mode, show full page before first panel, L→R & top→bottom
- NEW: Multiple balloons attached to a panel are merged into a single viewing unit
- NEW: Double-click anywhere advances to the next unit when Reading Mode is ON
"""

from __future__ import annotations
import sys, os, json, yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QAction, QImage, QPixmap, QPen, QColor, QPainter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsSimpleTextItem,
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

# --- Helper functions for the optimized pipeline ---

def _area(r: QRectF) -> float:
    return max(0.0, r.width() * r.height())

def _iou(a: QRectF, b: QRectF) -> float:
    inter = a.intersected(b)
    if inter.isEmpty(): return 0.0
    return (inter.width()*inter.height()) / max(1e-6, _area(a) + _area(b) - inter.width()*inter.height())

def _containment(a: QRectF, b: QRectF) -> float:
    """portion de b couverte par a"""
    inter = a.intersected(b)
    if inter.isEmpty(): return 0.0
    return (inter.width()*inter.height()) / max(1e-6, _area(b))

def _overlap_frac(a: QRectF, b: QRectF) -> float:
    inter = a.intersected(b)
    if inter.isEmpty(): return 0.0
    return (inter.width()*inter.height()) / max(1e-6, _area(b))

def _merge_adjacent_panels(rects: list[QRectF], iou_thr: float, dist_thr: float) -> list[QRectF]:
    """fusion IoU + proximité centres (anti faux-splits)"""
    merged, used = [], [False]*len(rects)
    for i, r1 in enumerate(rects):
        if used[i]: continue
        cur = QRectF(r1)
        for j, r2 in enumerate(rects):
            if i==j or used[j]: continue
            iou = _iou(cur, r2)
            dx = abs(cur.center().x() - r2.center().x()) / max(cur.width(),  r2.width(),  1.0)
            dy = abs(cur.center().y() - r2.center().y()) / max(cur.height(), r2.height(), 1.0)
            if iou > iou_thr or dx < dist_thr or dy < dist_thr:
                cur = cur.united(r2); used[j] = True
        merged.append(cur)
    return merged

def _row_band_merge(rects: list[QRectF], same_row_overlap: float, gap_pct: float, page_w: float) -> list[QRectF]:
    """fusion horizontale dans une même rangée (overlap vertical fort + petit gap horizontal)"""
    if not rects: return rects
    def v_overlap(a: QRectF, b: QRectF) -> float:
        inter_h = max(0.0, min(a.bottom(), b.bottom()) - max(a.top(), b.top()))
        return inter_h / max(1e-6, min(a.height(), b.height()))
    rows, used = [], [False]*len(rects)
    for i, r in enumerate(rects):
        if used[i]: continue
        bucket = [r]; used[i]=True
        for j, s in enumerate(rects):
            if used[j]: continue
            if v_overlap(r, s) >= same_row_overlap: bucket.append(s); used[j]=True
        rows.append(bucket)
    merged_rows, max_gap = [], page_w*gap_pct
    for bucket in rows:
        bucket = sorted(bucket, key=lambda rr: rr.left())
        cur = bucket[0]
        for nx in bucket[1:]:
            if (nx.left() - cur.right()) <= max_gap: cur = cur.united(nx)
            else: merged_rows.append(cur); cur = nx
        merged_rows.append(cur)
    return merged_rows



@dataclass
class Detection:
    cls: int
    rect: QRectF  # (x,y,w,h) in image coords
    conf: float = 0.0

# ---------------- graphics ----------------
class BBoxItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, color: QColor, parent=None, label: str | None = None):
        super().__init__(rect, parent)
        pen = QPen(color); pen.setCosmetic(True); pen.setWidthF(1.2)
        self.setPen(pen); self.setBrush(Qt.NoBrush); self.setZValue(10)
        if label:
            t = QGraphicsSimpleTextItem(label, self)
            t.setBrush(color)
            t.setFlag(QGraphicsSimpleTextItem.ItemIgnoresTransformations, True)
            t.setPos(rect.left()+2, rect.top()+2)

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
        a_zoom_in = QAction("🔍+", self); a_zoom_in.triggered.connect(self.zoom_in); a_zoom_in.setShortcut("Ctrl++"); tb.addAction(a_zoom_in)
        a_zoom_out = QAction("🔍-", self); a_zoom_out.triggered.connect(self.zoom_out); a_zoom_out.setShortcut("Ctrl+-"); tb.addAction(a_zoom_out)
        a_fit_window = QAction("Fit Window", self); a_fit_window.triggered.connect(self.fit_to_window); a_fit_window.setShortcut("Ctrl+0"); tb.addAction(a_fit_window)
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

    def _draw_detections(self, debug_tiles=None):
        if not self.pixmap_item: return
        # nettoyer d'abord
        for ch in list(self.pixmap_item.childItems()):
            self.scene.removeItem(ch)
        # dessiner les détections
        for d in self.dets:
            color = QColor(35,197,83) if d.cls==0 else QColor(41,121,255)
            label = ("panel" if d.cls==0 else "balloon") + f" {d.conf:.2f}"
            BBoxItem(d.rect, color, self.pixmap_item, label=label)
        
        # dessiner les tuiles pour debug si demandé
        if debug_tiles:
            for i, (x1, y1, x2, y2) in enumerate(debug_tiles):
                tile_rect = QRectF(x1, y1, x2-x1, y2-y1)
                BBoxItem(tile_rect, QColor(255, 165, 0, 80), self.pixmap_item, label=f"tile_{i}")  # Orange semi-transparent

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
        """Fast & Stable pipeline: single-pass preferred, capped tiling, robust merges, cover heuristic."""
        if self.qimage_current is None or self.model is None:
            self.dets = []; self._draw_detections(); return

        # --- Charger config YAML (optionnel) ---
        config = {}
        try:
            import yaml, os
            for cand in ("config/detect.yaml", "detect.yaml"):
                if os.path.exists(cand):
                    with open(cand, "r", encoding="utf-8") as f:
                        config.update(yaml.safe_load(f) or {})
                    break
        except Exception:
            pass

        # --- Unpack ---
        PANEL_CONF   = float(config.get('panel_conf', 0.18))
        PANEL_MIN_P  = float(config.get('panel_area_min_pct', 0.012))
        BAL_CONF     = float(config.get('balloon_conf', 0.22))
        BAL_MIN_PCT  = float(config.get('balloon_area_min_pct', 0.0006))
        BAL_MIN_W    = int(config.get('balloon_min_w', 30))
        BAL_MIN_H    = int(config.get('balloon_min_h', 22))
        MAX_BAL      = int(config.get('max_balloons', 12))
        IMGSZ_MAX    = int(config.get('imgsz_max', 1280))
        TILE_TGT     = int(config.get('tile_target', 896))
        TILE_OV      = float(config.get('tile_overlap', 0.20))
        FORCE_TILE   = bool(config.get('force_tiling', False))
        MAX_DET      = int(config.get('max_det', 400))
        PM_IOU       = float(config.get('panel_merge_iou', 0.40))
        PM_CONT      = float(config.get('panel_merge_cont', 0.70))
        ROW_OV       = float(config.get('panel_row_overlap', 0.55))
        ROW_GAP_PCT  = float(config.get('panel_row_gap_pct', 0.03))
        FULL_PCT     = float(config.get('full_page_panel_pct', 0.80))
        FULL_KEEP    = bool(config.get('full_page_keep_balloons', True))
        FULL_BAL_PCT = float(config.get('full_page_balloon_overlap_pct', 0.10))

        # --- Préparation image ---
        qimg = self.qimage_current
        W, H = qimg.width(), qimg.height()
        PAGE_AREA = float(W*H)
        imgsz = min(IMGSZ_MAX, max(W, H))

        # --- Décider tiling ---
        nx = ny = 1
        if (max(W, H) > TILE_TGT*1.15) or FORCE_TILE:
            # cap à 2 ou 3 tuiles selon taille
            nx = ny = 2 if max(W, H) < TILE_TGT*2.2 else 3

        # --- Predict helper (single pass on numpy RGB) ---
        import numpy as np
        def _to_rgb(q: QImage) -> np.ndarray:
            if q.format() != QImage.Format.Format_RGBA8888:
                q_ = q.convertToFormat(QImage.Format.Format_RGBA8888)
            else:
                q_ = q
            h, w = q_.height(), q_.width()
            buf = bytes(q_.constBits())
            arr = np.frombuffer(buf, dtype=np.uint8)[: h*q_.bytesPerLine()].reshape(h, q_.bytesPerLine())[:, : w*4]
            return arr.reshape(h, w, 4)[:, :, :3].copy()

        def _predict_rgb(np_rgb):
            res = self.model.predict(
                source=np_rgb, imgsz=imgsz, conf=min(PANEL_CONF, BAL_CONF),
                iou=0.6, max_det=MAX_DET, augment=False, verbose=False,
                device="mps" if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "1") else None
            )[0]
            out = []
            if res and getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls  = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                for (x1,y1,x2,y2), c, p in zip(xyxy, cls, conf):
                    out.append((int(c), float(p), QRectF(float(x1), float(y1), float(x2-x1), float(y2-y1))))
            return out

        # --- Exécution single/tiles ---
        dets_raw = []
        if nx == 1 and ny == 1:
            dets_raw = _predict_rgb(_to_rgb(qimg))
            tiles_str = "1x1"
        else:
            # tiling simple, overlap proportionnel
            dx, dy = W/nx, H/ny
            ox, oy = dx*TILE_OV, dy*TILE_OV
            for iy in range(ny):
                for ix in range(nx):
                    x1 = max(0, int(ix*dx - ox)); y1 = max(0, int(iy*dy - oy))
                    x2 = min(W, int((ix+1)*dx + ox)); y2 = min(H, int((iy+1)*dy + oy))
                    sub = qimg.copy(x1, y1, x2-x1, y2-y1)
                    preds = _predict_rgb(_to_rgb(sub))
                    # remapper en coords page
                    for (c, p, r) in preds:
                        r.translate(x1, y1)
                        dets_raw.append((c, p, r))
            tiles_str = f"{nx}x{ny}"

        # --- Séparer / filtrer ---
        panels   = [(c,p,r) for (c,p,r) in dets_raw if c==0 and p>=PANEL_CONF and _area(r) >= PANEL_MIN_P*PAGE_AREA]
        balloons = [(c,p,r) for (c,p,r) in dets_raw if c==1 and p>=BAL_CONF   and _area(r) >= BAL_MIN_PCT*PAGE_AREA
                    and r.width()>=BAL_MIN_W and r.height()>=BAL_MIN_H]

        # --- Fusion hiérarchique panels ---
        # 1) IoU + proximité centres
        pan_rects = _merge_adjacent_panels([r for (_,_,r) in panels], iou_thr=float(config.get('panel_merge_iou',0.40)),
                                           dist_thr=float(config.get('panel_merge_dist',0.02)))
        # 2) Row/band merge (même rangée)
        pan_rects = _row_band_merge(pan_rects, same_row_overlap=ROW_OV, gap_pct=ROW_GAP_PCT, page_w=W)
        # 3) Containment hiérarchique (garde grands, enlève petits encapsulés)
        kept = []
        for i, ri in enumerate(pan_rects):
            drop = False
            for j, rj in enumerate(pan_rects):
                if i==j: continue
                if _containment(rj, ri) > PM_CONT: drop = True; break
            if not drop: kept.append(ri)
        pan_rects = kept
        # reconstruire panels avec une conf raisonnable (max des confs initiales)
        best_conf = max([p for (_,p,__) in panels], default=0.5)
        panels = [(0, best_conf, rr) for rr in pan_rects]

        # --- Anti-couverture: panel couvrant ~toute la page ---
        if panels:
            largest = max(panels, key=lambda t: _area(t[2]))
            if _area(largest[2]) / max(1e-6, PAGE_AREA) >= FULL_PCT:
                page_rect = QRectF(0,0,W,H)
                panels = [(0, largest[1], page_rect)]
                if FULL_KEEP:
                    balloons = [(c,p,r) for (c,p,r) in balloons if _overlap_frac(r, page_rect) >= FULL_BAL_PCT]
                else:
                    balloons = []

        # --- Limiter le nombre de ballons (stabilité UI) ---
        balloons = sorted(balloons, key=lambda t: t[1], reverse=True)[:MAX_BAL]

        # --- Sortie (dessin) ---
        self.dets = [Detection(c, r, p) for c, p, r in panels + balloons]
        self._draw_detections()

        # --- Status ---
        if hasattr(self, "status") and self.status:
            self.status.showMessage(f"Page {self.page_index+1}: panels={len(panels)}, balloons={len(balloons)} (imgsz={imgsz}, tiles={tiles_str})")

    # ---------- reading mode ----------
    def _prepare_reading_units(self):
        """Une unité = panel ∪ (balloons qui chevauchent le panel). Ballons restants = unités seules."""
        self.read_units = []
        if not getattr(self, "dets", None):
            self.read_index = -1; return

        # séparer
        panels   = [r for (c,p,r) in [(d.cls, d.conf, d.rect) for d in self.dets] if c==0]
        balloons = [r for (c,p,r) in [(d.cls, d.conf, d.rect) for d in self.dets] if c==1]

        # associer ballons aux panels
        attached = set()
        for pr in panels:
            merged = QRectF(pr)
            for i, br in enumerate(balloons):
                if i in attached: continue
                if not merged.intersected(br).isEmpty():
                    merged = merged.united(br); attached.add(i)
            self.read_units.append(merged)

        # ballons seuls
        for i, br in enumerate(balloons):
            if i not in attached: self.read_units.append(QRectF(br))

        # ordre de lecture
        # direction configurable: self.cfg.direction in {"LR_TB", "TB_LR"} (par ex.)
        if getattr(self, "cfg", None) and getattr(self.cfg, "direction", "LR_TB") == "LR_TB":
            self.read_units.sort(key=lambda r: (r.top(), r.left()))
        else:
            self.read_units.sort(key=lambda r: (r.left(), r.top()))

        self.read_index = -1


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
        pad=0.08
        rr=QRectF(r); rr.adjust(-r.width()*pad,-r.height()*pad,r.width()*pad,r.height()*pad)
        rr = rr.intersected(self.pixmap_item.boundingRect())
        rect_scene = self.pixmap_item.mapRectToScene(rr)
        self.view.fitInView(rect_scene, Qt.AspectRatioMode.KeepAspectRatio)
        self.view.centerOn(rect_scene.center())

    def fit_full_page(self):
        if not self.pixmap_item: return
        self.view.fitInView(self.pixmap_item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.view.centerOn(self.pixmap_item)

    # ---------- zoom/reset ----------
    def reset_view(self):
        self.view.resetTransform()
        self.view.centerOn(self.pixmap_item)

    def zoom_in(self):
        """Zoom avant de 25%"""
        if self.pixmap_item:
            self.view.scale(1.25, 1.25)

    def zoom_out(self):
        """Zoom arrière de 20%"""
        if self.pixmap_item:
            self.view.scale(0.8, 0.8)

    def fit_to_window(self):
        """Ajuste l'image pour qu'elle tienne dans la fenêtre"""
        if self.pixmap_item:
            self.view.fitInView(self.pixmap_item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
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
        self._prepare_reading_units() # Re-sort reading units
        self.reset_view()

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
