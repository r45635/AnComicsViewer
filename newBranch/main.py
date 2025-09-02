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

# --- Smart panel merge utility ----------------------------------------------
def _merge_adjacent_panels(rects: list[QRectF],
                           iou_thr: float,
                           dist_thr: float) -> list[QRectF]:
    """
    Fusionne les panels qui se chevauchent (IoU) ou dont les centres sont très proches,
    pour éviter la découpe en grille / faux splits.
    - iou_thr  : seuil de recouvrement relatif (0.3 recommandé)
    - dist_thr : seuil de distance relative (rapporté à la taille des boîtes) (0.02 recommandé)
    """
    def _area(r: QRectF) -> float:
        return max(0.0, r.width() * r.height())

    merged: list[QRectF] = []
    used = [False] * len(rects)

    for i, r1 in enumerate(rects):
        if used[i]:
            continue
        cur = QRectF(r1)
        for j, r2 in enumerate(rects):
            if i == j or used[j]:
                continue
            inter = cur.intersected(r2)
            inter_area = _area(inter)
            union_area = _area(cur.united(r2))
            iou = inter_area / union_area if union_area > 0 else 0.0
            # distances des centres, normalisées par la taille la plus grande de chaque axe
            dx = abs(cur.center().x() - r2.center().x()) / max(cur.width(),  r2.width(),  1.0)
            dy = abs(cur.center().y() - r2.center().y()) / max(cur.height(), r2.height(), 1.0)

            if iou > iou_thr or dx < dist_thr or dy < dist_thr:
                cur = cur.united(r2)
                used[j] = True

        merged.append(cur)
    return merged
# --- end smart panel merge ---------------------------------------------------

# --- Row-wise band merge (optional but very effective) -----------------------
def _row_band_merge(rects: list[QRectF],
                    same_row_overlap: float,
                    gap_pct: float,
                    page_w: float) -> list[QRectF]:
    """
    1) groupe les boîtes qui se recouvrent fortement en Y (même rangée),
    2) fusionne les boîtes d'une même rangée si l'écart horizontal est petit.

    same_row_overlap: fraction min d'overlap vertical entre 2 boîtes pour les considérer dans la même rangée (ex: 0.55)
    gap_pct:         écart horizontal max (en % de la page) pour autoriser la fusion (ex: 0.03)
    """
    if not rects:
        return rects

    def v_overlap(a: QRectF, b: QRectF) -> float:
        inter_h = max(0.0, min(a.bottom(), b.bottom()) - max(a.top(), b.top()))
        return inter_h / max(1e-6, min(a.height(), b.height()))

    # 1) cluster par rangées
    rows: list[list[QRectF]] = []
    used = [False]*len(rects)
    for i, r in enumerate(rects):
        if used[i]: continue
        bucket = [r]; used[i] = True
        for j, s in enumerate(rects):
            if used[j]: continue
            if v_overlap(r, s) >= same_row_overlap:
                bucket.append(s); used[j] = True
        rows.append(bucket)

    # 2) fusion à l'intérieur de chaque rangée si écart horizontal faible
    merged_rows: list[QRectF] = []
    max_gap = page_w * gap_pct
    for bucket in rows:
        bucket = sorted(bucket, key=lambda rr: rr.left())
        cur = bucket[0]
        for nx in bucket[1:]:
            gap = nx.left() - cur.right()
            if gap <= max_gap:
                cur = cur.united(nx)
            else:
                merged_rows.append(cur)
                cur = nx
        merged_rows.append(cur)

    return merged_rows
# --- end row-wise band merge -------------------------------------------------

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
        """
        Tiled + multi-pass detection with advanced filtering.
        - Loads config from config/detect.yaml if present.
        - Full image, gamma, and CLAHE passes.
        - Tiled inference with overlap.
        - Multi-stage filtering: clamp, IoU merge, panel containment, balloon filtering.
        """
        self.dets: List[Detection] = []
        if self.qimage_current is None:
            return
        if self.model is None:
            self.status.showMessage("⚠️ No model loaded — use 'Load model…'")
            return

        # --- Load config from YAML or use defaults ---
        config = {
            'panel_conf': 0.08, 'balloon_conf': 0.22, 'balloon_area_min_pct': 0.0006,
            'balloon_min_w': 30, 'balloon_min_h': 22, 'tile_target': 1024,
            'tile_overlap': 0.25, 'iou_merge': 0.55, 'panel_containment_merge': 0.65,
            'max_balloons': 20, 'page_margin_inset_pct': 0.015, 'imgsz_max': 1536
        }
        try:
            config_path = Path(__file__).parent / "config" / "detect.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        config.update(yaml_config)
        except Exception as e:
            print(f"Could not load or parse config/detect.yaml: {e}")

        # --- Unpack config values (with sane defaults) ---
        PANEL_CONF = float(config.get('panel_conf', 0.08))
        BAL_CONF   = float(config.get('balloon_conf', 0.22))

        BAL_MIN_PCT = float(config.get('balloon_area_min_pct', 0.0006))
        BAL_MIN_W   = int(config.get('balloon_min_w', 30))
        BAL_MIN_H   = int(config.get('balloon_min_h', 22))

        TILE_TGT    = int(config.get('tile_target', 1024))
        TILE_OV     = float(config.get('tile_overlap', 0.25))
        TILE_OVERLAP = TILE_OV  # Alias for compatibility

        IOU_MERGE   = float(config.get('iou_merge', 0.55))
        PANEL_CONT  = float(config.get('panel_containment_merge', 0.65))

        MAX_BAL     = int(config.get('max_balloons', 20))
        MARGIN_PCT  = float(config.get('page_margin_inset_pct', 0.015))

        IMGSZ_MAX   = int(config.get('imgsz_max', 1536))

        # --- full page config ---
        FULL_PAGE_PCT   = float(config.get('full_page_panel_pct', 0.80))
        FULL_KEEP_BAL   = bool(config.get('full_page_keep_balloons', True))
        FULL_BAL_OV_PCT = float(config.get('full_page_balloon_overlap_pct', 0.15))

        # --- tiling and debug config ---
        FORCE_TILING    = bool(config.get('force_tiling', False))
        DEBUG_TILES     = bool(config.get('debug_tiles', False))

        # --- panel merge config ---
        PANEL_MERGE_IOU = float(config.get('panel_merge_iou', 0.3))
        PANEL_MERGE_DIST = float(config.get('panel_merge_dist', 0.02))

        rgb = qimage_to_rgb(self.qimage_current)
        H, W = rgb.shape[:2]
        PAGE_AREA = float(H * W)

        max_side = max(H, W)
        imgsz = min(IMGSZ_MAX, max(960, max_side // 2))
        imgsz -= imgsz % 32

        def predict_once(img) -> list[tuple[int, float, QRectF]]:
            res = self.model.predict(
                source=img, imgsz=imgsz, conf=min(PANEL_CONF, BAL_CONF),
                iou=0.6, max_det=500, augment=True, verbose=False,
                device="mps" if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "1") else None
            )[0]
            out = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                    if (c == 0 and p < PANEL_CONF) or (c == 1 and p < BAL_CONF):
                        continue
                    out.append((c, float(p), QRectF(float(x1), float(y1), float(x2 - x1), float(y2 - y1))))
            return out

        all_dets: list[tuple[int, float, QRectF]] = []
        all_dets += predict_once(rgb)

        try:
            g = np.clip((rgb / 255.0) ** 0.9, 0, 1.0)
            g = (g * 255).astype(np.uint8)
            all_dets += predict_once(g)
        except Exception: pass

        try:
            import cv2
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
            enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            all_dets += predict_once(enhanced)
        except (ImportError, Exception): pass

        # --- Smart Tiling Logic ---
        # Si l'image est petite ou force_tiling=false, pas de tiling
        max_dim = max(H, W)
        tiles = []
        
        if not FORCE_TILING and max_dim <= TILE_TGT * 1.2:
            # Image assez petite, pas de tiling nécessaire
            pass  # tiles reste vide, on n'ajoute que les détections full-image
        else:
            # Tiling intelligent: limité à 2x2 ou 3x3 maximum
            if max_dim <= TILE_TGT * 2:
                # 2x2 tiling
                nx, ny = 2, 2
            elif max_dim <= TILE_TGT * 3:
                # 3x3 tiling  
                nx, ny = 3, 3
            else:
                # Pour très grandes images, on peut aller jusqu'à 4x4
                nx = min(4, max(2, round(W / TILE_TGT)))
                ny = min(4, max(2, round(H / TILE_TGT)))
            
            step_x = int(W / nx * (1 - TILE_OVERLAP))
            step_y = int(H / ny * (1 - TILE_OVERLAP))
            
            y = 0
            while y < H:
                x = 0
                yh = min(H, y + int(H / ny + TILE_OVERLAP * H / ny))
                while x < W:
                    xw = min(W, x + int(W / nx + TILE_OVERLAP * W / nx))
                    tiles.append((x, y, xw, yh))
                    x += step_x
                y += step_y

        # Tiled inference (seulement si on a des tuiles)
        for (x1, y1, x2, y2) in tiles:
            tile = rgb[y1:y2, x1:x2, :]
            if tile.size == 0: continue
            for c, p, r in predict_once(tile):
                r.translate(x1, y1)
                all_dets.append((c, p, r))

        # --- Post-processing Pipeline ---
        page_rect = QRectF(0, 0, W, H)
        margin_x, margin_y = W * MARGIN_PCT, H * MARGIN_PCT
        inset_page_rect = page_rect.adjusted(margin_x, margin_y, -margin_x, -margin_y)

        def iou(a: QRectF, b: QRectF) -> float:
            inter = a.intersected(b)
            if inter.isEmpty(): return 0.0
            ia = inter.width() * inter.height()
            ua = a.width() * a.height() + b.width() * b.height() - ia
            return ia / max(ua, 1e-6)

        # 1. Clamp all boxes to inset page rect
        clamped_dets = []
        for c, p, r in all_dets:
            r_clamped = r.intersected(inset_page_rect)
            if not r_clamped.isEmpty():
                clamped_dets.append((c, p, r_clamped))

        # 2. IoU Merge
        merged = []
        for c, p, r in sorted(clamped_dets, key=lambda x: x[1], reverse=True):
            is_merged = False
            for i, (mc, mp, mr) in enumerate(merged):
                if c == mc and iou(r, mr) > IOU_MERGE:
                    merged[i] = (mc, max(p, mp), mr.united(r))
                    is_merged = True
                    break
            if not is_merged:
                merged.append((c, p, r))

        panels = [d for d in merged if d[0] == 0]
        balloons = [d for d in merged if d[0] == 1]

        # 2.5. Additional cleanup: Remove high-overlap duplicates for panels
        # Si deux panels ont IoU > 0.7, ne garder que le plus grand
        cleaned_panels = []
        for i, (c_i, p_i, r_i) in enumerate(panels):
            keep = True
            for j, (c_j, p_j, r_j) in enumerate(cleaned_panels):
                if iou(r_i, r_j) > 0.7:
                    # Si le nouveau panel est plus petit, on l'ignore
                    area_i = r_i.width() * r_i.height()
                    area_j = r_j.width() * r_j.height()
                    if area_i <= area_j:
                        keep = False
                        break
                    else:
                        # Le nouveau est plus grand, on remplace l'ancien
                        cleaned_panels[j] = (c_i, max(p_i, p_j), r_i)
                        keep = False
                        break
            if keep:
                cleaned_panels.append((c_i, p_i, r_i))
        
        panels = cleaned_panels

        # 3. Panel Containment Merge
        def containment(a: QRectF, b: QRectF) -> float:
            inter = a.intersected(b)
            if inter.isEmpty(): return 0.0
            return (inter.width() * inter.height()) / max(min(a.width() * a.height(), b.width() * b.height()), 1e-6)

        final_panels = []
        if panels:
            used = set()
            for i, (c_i, p_i, r_i) in enumerate(panels):
                if i in used: continue
                current_cluster = [r_i]
                current_conf = p_i
                used.add(i)
                for j, (c_j, p_j, r_j) in enumerate(panels[i + 1:], i + 1):
                    if j in used: continue
                    if containment(r_i, r_j) > PANEL_CONT:
                        current_cluster.append(r_j)
                        current_conf = max(current_conf, p_j)
                        used.add(j)
                
                final_r = current_cluster[0]
                for rect in current_cluster[1:]: final_r = final_r.united(rect)
                final_panels.append((0, current_conf, final_r))

        # 4. Balloon Filtering
        final_balloons = []
        if balloons:
            # Size filter
            size_filtered = [
                (c, p, r) for c, p, r in balloons
                if r.width() * r.height() >= BAL_MIN_PCT * PAGE_AREA
                and r.width() >= BAL_MIN_W and r.height() >= BAL_MIN_H
            ]
            # IoU merge (again, for balloons specifically)
            merged_balloons = []
            for c, p, r in sorted(size_filtered, key=lambda x: x[1], reverse=True):
                is_merged = False
                for i, (mc, mp, mr) in enumerate(merged_balloons):
                    if iou(r, mr) > 0.5:
                        merged_balloons[i] = (mc, max(p, mp), mr.united(r))
                        is_merged = True; break
                if not is_merged: merged_balloons.append((c, p, r))
            
            # Keep top N by confidence
            final_balloons = sorted(merged_balloons, key=lambda x: x[1], reverse=True)[:MAX_BAL]

        # --- Final Assembly ---
        panels = final_panels
        balloons = final_balloons

        # --- Fusion intelligente des panels (post-process) --------------------------
        if config.get('enable_panel_merge', True):
            PM_IOU  = float(config.get('panel_merge_iou', 0.3))
            PM_DIST = float(config.get('panel_merge_dist', 0.02))

            # extraire uniquement les QRectF des panels
            _pan_rects = [r for (c, p, r) in panels]
            if _pan_rects:
                _pan_rects = _merge_adjacent_panels(_pan_rects, iou_thr=PM_IOU, dist_thr=PM_DIST)
                # reconstruire "panels" en conservant une confiance raisonnable (max des fusionnés, si tu veux aller plus loin)
                if len(_pan_rects) != len(panels):
                    best_conf = max((p for (_, p, __) in panels), default=0.5)
                    panels = [(0, best_conf, rr) for rr in _pan_rects]
        # --- end fusion intelligente -------------------------------------------------

        # --- Optional: suppression de panels encapsulés -----------------------------
        def _contains_ratio(a: QRectF, b: QRectF) -> float:
            inter = a.intersected(b)
            if inter.isEmpty():
                return 0.0
            return (inter.width() * inter.height()) / max(1e-6, b.width()*b.height())

        kept = []
        for i, (c_i, p_i, r_i) in enumerate(panels):
            drop = False
            for j, (c_j, p_j, r_j) in enumerate(panels):
                if i == j:
                    continue
                # si r_i est contenu à >85% dans r_j, on le supprime
                if _contains_ratio(r_j, r_i) > 0.85:
                    drop = True
                    break
            if not drop:
                kept.append((c_i, p_i, r_i))
        panels = kept
        # --- end optional ------------------------------------------------------------

        # --- Apply row-wise band merge ----------------------------------------------
        if config.get('enable_row_merge', True):
            ROW_OVERLAP  = float(config.get('panel_row_overlap', 0.55))  # 55% de recouvrement vertical = même rangée
            ROW_GAP_PCT  = float(config.get('panel_row_gap_pct', 0.03))  # gap horizontal max = 3% de la page

            _pan_rects = [r for (_, _, r) in panels]
            _pan_rects = _row_band_merge(_pan_rects, same_row_overlap=ROW_OVERLAP, gap_pct=ROW_GAP_PCT, page_w=W)

            # reconstruire les panels (on conserve une conf raisonnable)
            if _pan_rects:
                best_conf = max((p for (_, p, __) in panels), default=0.5)
                panels = [(0, best_conf, rr) for rr in _pan_rects]
        # ---------------------------------------------------------------------------

        # --- Tiled-grid artifact suppression ----------------------------------------
        # Si beaucoup de "panels" ont une taille proche de la tuile, disposés en grille,
        # et qu'il n'y a pas de gouttières blanches (gutter) -> considérer comme artefact.
        
        if config.get('enable_antigrille', True):
            from statistics import median

            def _area(r: QRectF) -> float:
                return max(0.0, r.width() * r.height())

            def _approx_equal(a, b, tol):
                return abs(a - b) <= tol

            def _detect_tiling_artifacts(panels, W, H, tile_tgt, overlap):
                min_count = int(config.get('antigrille_min_count', 8))
                tile_tol = float(config.get('antigrille_tile_match', 0.35))
                grid_fill = float(config.get('antigrille_grid_fill', 0.5))
                
                n = len(panels)
                if n < min_count:
                    return False

                # taille théorique d'une tuile
                nx = max(2, round(W / tile_tgt))
                ny = max(2, round(H / tile_tgt))
                tile_w = W / nx
                tile_h = H / ny

                ws = [r.width()  for (_, _, r) in panels]
                hs = [r.height() for (_, _, r) in panels]
                if not ws or not hs:
                    return False

                mw, mh = median(ws), median(hs)

                # 1) la médiane des dimensions de "panel" colle à la tuile
                tol_w = tile_tol * tile_w
                tol_h = tile_tol * tile_h
                looks_like_tiles = (
                    (tile_w - tol_w) <= mw <= (tile_w + tol_w) and
                    (tile_h - tol_h) <= mh <= (tile_h + tol_h)
                )

                if not looks_like_tiles:
                    return False

                # 2) assez de boxes pour former une grille
                grid_like_count = nx * ny
                if n < max(min_count, round(grid_fill * grid_like_count)):
                    return False

                # 3) pas de "gouttières" blanches marquées
                cols = [0] * nx
                rows = [0] * ny
                for (_, _, r) in panels:
                    cx = min(nx - 1, max(0, int((r.center().x() / max(1.0, W)) * nx)))
                    ry = min(ny - 1, max(0, int((r.center().y() / max(1.0, H)) * ny)))
                    cols[cx] += 1
                    rows[ry] += 1
                # si bcp de colonnes ET de lignes sont remplies de façon homogène -> grille
                filled_cols = sum(1 for c in cols if c >= max(1, n // (nx * 3)))
                filled_rows = sum(1 for r in rows if r >= max(1, n // (ny * 3)))
                uniform_grid = (filled_cols >= max(2, nx - 1)) and (filled_rows >= max(2, ny - 1))

                return uniform_grid

            # appliquer l'heuristique
            if _detect_tiling_artifacts(panels, W, H, TILE_TGT, TILE_OVERLAP):
                # on collapse en un seul panel = page entière
                if panels:
                    best_conf = max(p for (_, p, __) in panels)
                else:
                    best_conf = 0.5
                page_rect = QRectF(0, 0, W, H)
                panels = [(0, best_conf, page_rect)]

                # on filtre les ballons pour ne garder que ceux qui chevauchent la page (quasi tous)
                if FULL_KEEP_BAL:
                    def _overlap_frac(a: QRectF, b: QRectF) -> float:
                        inter = a.intersected(b)
                        if inter.isEmpty(): return 0.0
                        return (inter.width() * inter.height()) / max(1e-6, _area(b))
                    balloons = [(c, p, r) for (c, p, r) in balloons
                                if _overlap_frac(r, page_rect) >= FULL_BAL_OV_PCT]
                else:
                    balloons = []
        # --- end tiled-grid suppression ---------------------------------------------

        # Convertir en objets Detection
        self.dets = [Detection(c, r, p) for c, p, r in panels + balloons]

        # --- Full-Page Panel Detection ---
        if self.dets:
            panels = [d for d in self.dets if d.cls == 0]
            balloons = [d for d in self.dets if d.cls == 1]
            
            # Chercher le panel avec la plus grande couverture
            best_coverage = 0.0
            best_panel = None
            page_area = H * W
            
            for panel in panels:
                panel_area = panel.rect.width() * panel.rect.height()
                coverage = panel_area / page_area
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_panel = panel
            
            # Si le meilleur panel couvre plus de FULL_PAGE_PCT de la page
            if best_panel and best_coverage >= FULL_PAGE_PCT:
                # SUPPRIMER TOUS LES AUTRES PANELS (même ceux du tiling)
                if not FULL_KEEP_BAL:
                    # Mode simple: un seul panel, pas de bulles
                    self.dets = [best_panel]
                else:
                    # Garder seulement les bulles qui chevauchent significativement avec LE panel
                    filtered_balloons = []
                    for balloon in balloons:
                        inter = best_panel.rect.intersected(balloon.rect)
                        if not inter.isEmpty():
                            balloon_area = balloon.rect.width() * balloon.rect.height()
                            overlap_ratio = (inter.width() * inter.height()) / balloon_area
                            if overlap_ratio >= FULL_BAL_OV_PCT:
                                filtered_balloons.append(balloon)
                    
                    # Reconstruire: SEUL le best_panel + bulles filtrées
                    self.dets = [best_panel] + filtered_balloons

        if not self.dets:
            if self.pixmap_item:
                BBoxItem(QRectF(10, 10, 100, 100), QColor(255, 0, 0), parent=self.pixmap_item, label="debug")
            self.status.showMessage(f"Page {self.page_index+1}: 0 detections — check model/thresholds")
            return

        self._draw_detections(tiles if DEBUG_TILES else [])
        n_pan = sum(1 for d in self.dets if d.cls == 0)
        n_bal = sum(1 for d in self.dets if d.cls == 1)
        tiling_msg = f", no_tiling" if len(tiles) == 0 else f", tiles={len(tiles)}"
        self.status.showMessage(f"Page {self.page_index+1}: panels={n_pan}, balloons={n_bal} (imgsz={imgsz}{tiling_msg})")

    # ---------- reading mode ----------
    def _prepare_reading_units(self):
        self.read_units=[]
        if not getattr(self, 'dets', None):
            self.read_index = -1
            return

        panels=[d.rect for d in self.dets if d.cls==0]
        balloons=[d.rect for d in self.dets if d.cls==1]
        assigned=set()

        def attached(b: QRectF, base: QRectF) -> bool:
            # attachée si centre de la bulle dans panel étendu (5%) OU recouvrement >=5% de la bulle
            expanded = base.adjusted(-0.05*base.width(), -0.05*base.height(),
                                     +0.05*base.width(), +0.05*base.height())
            if expanded.contains(b.center()):
                return True
            inter = base.intersected(b)
            if inter.isEmpty():
                return False
            return (inter.width()*inter.height()) >= 0.05*(b.width()*b.height())

        for p in panels:
            merged = QRectF(p)
            changed = True
            # itératif pour englober PLUSIEURS bulles
            while changed:
                changed = False
                for i, b in enumerate(balloons):
                    if i in assigned:
                        continue
                    if attached(b, merged):
                        merged = merged.united(b); assigned.add(i); changed = True
            self.read_units.append(merged)

        # bulles orphelines -> unités
        for i, b in enumerate(balloons):
            if i not in assigned: self.read_units.append(QRectF(b))

        # Tri intelligent par rangées
        self.read_units = self._sort_reading_order(self.read_units, self.cfg.direction)
        self.read_index=-1

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
