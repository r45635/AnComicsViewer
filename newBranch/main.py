#!/usr/bin/env python3
"""
AnComicsViewer MINI — with Pixel↔PDF calibration & quality metrics

Adds to the minimal stable viewer:
- Pixel→PDF and PDF→Pixel conversions based on render DPI and page size (points)
- Per-page quality metrics derived from panel↔balloon relations
- Optional JSON export of metrics via --metrics-out

Usage:
  python main_minimal_calib.py --pdf your.pdf --page 0 --metrics-out outputs/metrics.json
  python main_minimal_calib.py --config detect_with_merge.yaml --debug-detect --save-debug-overlays debug
"""

from __future__ import annotations
import sys, os, json, argparse, time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QAction, QImage, QPixmap, QPen, QColor, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsSimpleTextItem,
    QToolBar, QWidget, QVBoxLayout, QStatusBar, QComboBox
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

# ---------------- GLOBALS ----------------
GLOBAL_CONFIG: Dict[str, Any] = {}
DEBUG_DETECT: bool = False
DEBUG_OVERLAY_DIR: Optional[str] = None
METRICS_OUT: Optional[str] = None

# ---------------- HELPERS ----------------
def _area(r: QRectF) -> float:
    return max(0.0, r.width() * r.height())

def _iou(a: QRectF, b: QRectF) -> float:
    inter = a.intersected(b)
    if inter.isEmpty():
        return 0.0
    i = inter.width() * inter.height()
    return i / max(1e-6, _area(a) + _area(b) - i)

def _overlap_frac(a: QRectF, b: QRectF) -> float:
    """Return the fraction of rectangle b that is contained within rectangle a."""
    inter = a.intersected(b)
    if inter.isEmpty():
        return 0.0
    return (inter.width() * inter.height()) / max(1e-6, _area(b))

def apply_nms_class_aware(dets: List[Tuple[int, float, QRectF]], iou_thr: float) -> List[Tuple[int, float, QRectF]]:
    """Apply NMS within each class separately (class-aware NMS)."""
    dets = sorted(dets, key=lambda x: x[1], reverse=True)
    kept: List[Tuple[int, float, QRectF]] = []
    while dets:
        cur = dets.pop(0)
        kept.append(cur)
        # Only suppress detections of the same class
        dets = [d for d in dets if d[0] != cur[0] or _iou(cur[2], d[2]) < iou_thr]
    return kept

def apply_nms(dets: List[Tuple[int, float, QRectF]], iou_thr: float) -> List[Tuple[int, float, QRectF]]:
    # class-aware NMS (legacy wrapper)
    return apply_nms_class_aware(dets, iou_thr)

def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️  Config load error: {e}")
        return {}

def qimage_to_rgb(qimg: QImage) -> np.ndarray:
    if qimg.format() != QImage.Format.Format_RGBA8888:
        qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    h, w = qimg.height(), qimg.width()
    ptr = qimg.constBits()
    bytes_per_line = qimg.bytesPerLine()
    arr = np.frombuffer(ptr, dtype=np.uint8)[: h * bytes_per_line]
    arr = arr.reshape(h, bytes_per_line)[:, : w * 4].reshape(h, w, 4)
    return arr[:, :, :3].copy()

def save_debug_overlay(image: QImage, panels, balloons, filename, step_name):
    if not DEBUG_DETECT or DEBUG_OVERLAY_DIR is None:
        return
    try:
        import cv2
        os.makedirs(DEBUG_OVERLAY_DIR, exist_ok=True)
        rgb = qimage_to_rgb(image)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # panels red
        for _, _, r in panels:
            x1, y1 = int(r.left()), int(r.top())
            x2, y2 = int(r.right()), int(r.bottom())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # balloons blue
        for _, _, r in balloons:
            x1, y1 = int(r.left()), int(r.top())
            x2, y2 = int(r.right()), int(r.bottom())
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        out = os.path.join(DEBUG_OVERLAY_DIR, f"{filename}_{step_name}.png")
        cv2.imwrite(out, img)
        print(f"💾 overlay: {out}")
    except Exception as e:
        print(f"⚠️ overlay error: {e}")

# ---------------- UI ITEMS ----------------
class BBoxItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, color: QColor, parent=None, label: str | None = None):
        super().__init__(rect, parent)
        pen = QPen(color)
        pen.setCosmetic(True)
        pen.setWidthF(1.2)
        self.setPen(pen)
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.setZValue(10)
        if label:
            t = QGraphicsSimpleTextItem(label, self)
            t.setBrush(color)
            t.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            t.setPos(rect.left() + 2, rect.top() + 2)

class ClickableGraphicsView(QGraphicsView):
    def mouseDoubleClickEvent(self, event):
        try:
            wnd = self.window()
            if getattr(wnd, 'cfg', None) and getattr(wnd.cfg, 'reading_mode', False):
                wnd.next_step()
                return
        except Exception:
            pass
        super().mouseDoubleClickEvent(event)

# ---------------- CONFIG ----------------
@dataclass
class AppConfig:
    reading_mode: bool = True
    show_full_page_before_first_panel: bool = True
    direction: str = "LR_TB"  # Left-to-right, Top-to-bottom
    detect_mode: str = "HYBRID"  # HYBRID, YOLO, RULES

# ---------------- MAIN WINDOW ----------------
class PdfYoloViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnComicsViewer MINI (Calib+Metrics)")
        self.resize(1100, 800)

        self.scene = QGraphicsScene(self)
        self.view = ClickableGraphicsView(self.scene, self)
        self.view.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        central = QWidget(self)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        self.setCentralWidget(central)

        self.status = QStatusBar(self)
        self.setStatusBar(self.status)

        self.pdf = None
        self.page_index = 0
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.qimage_current: Optional[QImage] = None
        self.model = None
        self.class_names = ["panel", "balloon"]
        self.balloon_class_index = 1
        self.balloon_detection_disabled = False

        self.show_panels, self.show_balloons = True, True
        self.cfg = AppConfig()

        self.read_units: List[QRectF] = []
        self.read_index: int = -1
        self.fullpage_shown_on_page: bool = False

        self._is_detecting = False
        self.dets: List[Tuple[int, float, QRectF]] = []

        # Calibration state
        self.render_dpi: float = 300.0
        self.page_size_pts: Tuple[float, float] = (0.0, 0.0)  # (w,h) in PDF points (1/72 inch)
        self.image_size_px: Tuple[int, int] = (0, 0)          # (W,H) in pixels for current rendered page

        self._build_ui()
        self._auto_load_model()
        self.auto_reload_last_file()

    # ---------- UI ----------
    def _build_ui(self):
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        a_open = QAction("Open PDF…", self)
        a_open.triggered.connect(self.open_pdf)
        tb.addAction(a_open)

        a_model = QAction("Load model…", self)
        a_model.triggered.connect(self.load_model)
        tb.addAction(a_model)

        self.model_status = QAction("🔴 no model", self)
        self.model_status.setEnabled(False)
        tb.addAction(self.model_status)

        tb.addSeparator()
        a_prev = QAction("◀ Prev", self)
        a_prev.triggered.connect(self.prev_step)
        tb.addAction(a_prev)

        a_next = QAction("Next ▶", self)
        a_next.triggered.connect(self.next_step)
        tb.addAction(a_next)

        tb.addSeparator()
        a_fit_window = QAction("Fit Window", self)
        a_fit_window.triggered.connect(self.fit_full_page)
        a_fit_window.setShortcut("Ctrl+0")
        tb.addAction(a_fit_window)

        tb.addSeparator()
        a_pan = QAction("Panels", self)
        a_pan.setCheckable(True)
        a_pan.setChecked(True)
        a_pan.toggled.connect(self._toggle_panels)
        tb.addAction(a_pan)

        self.a_bal = QAction("Balloons", self)
        self.a_bal.setCheckable(True)
        self.a_bal.setChecked(True)
        self.a_bal.toggled.connect(self._toggle_balloons)
        tb.addAction(self.a_bal)

        QShortcut(QKeySequence("Space"), self).activated.connect(self.next_step)
        QShortcut(QKeySequence("Shift+Space"), self).activated.connect(self.prev_step)
        QShortcut(QKeySequence("Return"), self).activated.connect(self.next_step)

        tb.addSeparator()
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem("HYBRID", "hybrid")
        self.mode_combo.addItem("YOLO", "yolo_only")
        self.mode_combo.addItem("RULES", "rules_only")
        self.mode_combo.currentIndexChanged.connect(self._change_detect_mode_idx)
        tb.addWidget(self.mode_combo)

    # ---------- PDF ----------
    def open_pdf(self):
        if fitz is None:
            QMessageBox.critical(self, "Missing PyMuPDF", "pip install pymupdf")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF (*.pdf)")
        if not path:
            return
        try:
            self.pdf = fitz.open(path)
            self.page_index = 0
            self.status.showMessage(f"PDF: {os.path.basename(path)} • {len(self.pdf)} pages")
            self.load_page(self.page_index)
            # Sauvegarder pour la prochaine fois
            self.save_user_settings(path, self.page_index)
        except Exception as e:
            QMessageBox.critical(self, "PDF error", str(e))

    def load_page(self, index: int):
        if not self.pdf or index < 0 or index >= len(self.pdf):
            return
        self.page_index = index
        page = self.pdf[index]

        # Store PDF page size in points (1 pt = 1/72 inch)
        self.page_size_pts = (float(page.rect.width), float(page.rect.height))

        # Render at fixed DPI for stable calibration
        dpi = self.render_dpi
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888).copy()

        # Store image size in pixels for calibration
        self.image_size_px = (qimg.width(), qimg.height())

        self._set_page_image(qimg)
        self._run_detection()
        self._prepare_reading_units()
        self.fullpage_shown_on_page = False
        if self.cfg.reading_mode and self.cfg.show_full_page_before_first_panel:
            self.fit_full_page()
            self.read_index = -1
            self.fullpage_shown_on_page = True

    def next_page(self):
        if self.pdf and self.page_index + 1 < len(self.pdf):
            self.load_page(self.page_index + 1)
            # Sauvegarder la nouvelle page
            if hasattr(self, 'pdf') and self.pdf and hasattr(self.pdf, 'name'):
                self.save_user_settings(self.pdf.name, self.page_index)

    def prev_page(self):
        if self.pdf and self.page_index - 1 >= 0:
            self.load_page(self.page_index - 1)
            # Sauvegarder la nouvelle page
            if hasattr(self, 'pdf') and self.pdf and hasattr(self.pdf, 'name'):
                self.save_user_settings(self.pdf.name, self.page_index)

    # ---------- image & overlays ----------
    def _set_page_image(self, qimg: QImage):
        self.scene.clear()
        self.qimage_current = qimg
        pix = QPixmap.fromImage(qimg)
        pix.setDevicePixelRatio(1.0)
        self.pixmap_item = QGraphicsPixmapItem(pix)
        self.pixmap_item.setZValue(0)
        self.scene.addItem(self.pixmap_item)
        self.fit_full_page()

    def _draw_detections(self):
        if not self.pixmap_item:
            return
        for ch in list(self.pixmap_item.childItems()):
            self.scene.removeItem(ch)

        for c, p, r in self.dets:
            if c == 0 and not self.show_panels:
                continue
            if c == 1 and (self.balloon_detection_disabled or not self.show_balloons):
                continue
            color = QColor(35, 197, 83) if c == 0 else QColor(41, 121, 255)
            label = ("panel" if c == 0 else "balloon") + f" {p:.2f}"
            BBoxItem(r, color, self.pixmap_item, label=label)

    # ---------- YOLO ----------
    def _auto_load_model(self):
        if YOLO is None:
            self.status.showMessage("YOLO unavailable – detection off")
            return

        candidates = [
            # Chemins relatifs dans le repo
            "./runs/detect/ancomics_final_optimized9/weights/best.pt",
            "./runs/detect/ancomics_final_optimized7/weights/best.pt", 
            "./runs/detect/ancomics_final_optimized6/weights/best.pt",
            "./dataset_improved/runs/detect/ancomics_improved4/weights/best.pt",
            "./dataset_improved/yolov8m.pt",
            # Chemins absolus (legacy)
            "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/runs/detect/ancomics_final_optimized9/weights/best.pt",
            "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/runs/detect/ancomics_final_optimized7/weights/best.pt",
            "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/runs/detect/ancomics_final_optimized_v2/weights/best.pt",
            "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/yolov8s.pt",
            # Fichiers dans le répertoire courant
            "anComicsViewer_v01.pt",
            "ancomics_improved.pt",
        ]
        
        print("🔍 Recherche de modèles YOLO...")
        for path in candidates:
            abs_path = os.path.abspath(path) if not os.path.isabs(path) else path
            print(f"   Teste: {path}")
            if os.path.exists(abs_path):
                try:
                    print(f"   ✅ Trouvé: {abs_path}")
                    self._load_model(abs_path)
                    self.status.showMessage(f"✅ Model loaded: {os.path.basename(abs_path)}")
                    return
                except Exception as e:
                    print(f"   ⚠️  Échec chargement {abs_path}: {e}")
        
        print("❌ Aucun modèle YOLO trouvé")
        self.model_status.setText("🔴 no model - use RULES mode")
        self.status.showMessage("No YOLO model found - switch to RULES mode for basic detection")

    def auto_reload_last_file(self):
        """Charge automatiquement le dernier fichier PDF ouvert"""
        settings = self.load_user_settings()
        last_file = settings.get("last_pdf_file")
        last_page = settings.get("last_page", 0)
        
        if last_file and os.path.exists(last_file):
            try:
                self.pdf = fitz.open(last_file)
                self.page_index = last_page
                # S'assurer que l'index de page est valide
                if self.page_index >= len(self.pdf):
                    self.page_index = 0
                self.load_page(self.page_index)
                self.status.showMessage(f"✅ Rechargé: {os.path.basename(last_file)} (page {self.page_index + 1})")
                print(f"✅ Auto-reload: {last_file} à la page {self.page_index + 1}")
            except Exception as e:
                print(f"❌ Échec auto-reload {last_file}: {e}")
                self.status.showMessage(f"⚠️  Failed to reload: {os.path.basename(last_file)}")
        else:
            print("ℹ️  Aucun fichier précédent à recharger")

    def load_user_settings(self) -> dict:
        """Charge les paramètres utilisateur depuis le fichier JSON"""
        settings_file = os.path.expanduser("~/.ancomics_settings.json")
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Erreur lecture settings: {e}")
        return {}

    def save_user_settings(self, pdf_path: str, page_index: int):
        """Sauvegarde les paramètres utilisateur dans le fichier JSON"""
        settings_file = os.path.expanduser("~/.ancomics_settings.json")
        settings = {
            "last_pdf_file": pdf_path,
            "last_page": page_index,
            "timestamp": time.time()
        }
        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"⚠️  Erreur sauvegarde settings: {e}")

    def load_model(self):
        if YOLO is None:
            QMessageBox.critical(self, "Missing Ultralytics", "pip install ultralytics")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO model", "", "PT (*.pt)")
        if not path:
            return
        try:
            self._load_model(path)
            self.status.showMessage(f"Model: {os.path.basename(path)}")
            if self.qimage_current is not None:
                self._run_detection()
        except Exception as e:
            QMessageBox.critical(self, "Model error", str(e))
            self.model_status.setText("🔴 model error")

    def _load_model(self, path: str):
        self.model = YOLO(path)
        # Balloon class detection availability
        self.balloon_detection_disabled = False
        self.balloon_class_index = 1
        if hasattr(self.model, "names") and self.model.names:
            if "balloon" not in self.model.names.values():
                self.balloon_detection_disabled = True
            else:
                for k, v in self.model.names.items():
                    if v == "balloon":
                        self.balloon_class_index = k
                        break

        status_text = f"🟢 {os.path.basename(path)}"
        if self.balloon_detection_disabled and hasattr(self, 'a_bal'):
            status_text += " (balloons disabled)"
            self.a_bal.setEnabled(False)
            self.a_bal.setChecked(False)
        elif hasattr(self, 'a_bal'):
            self.a_bal.setEnabled(True)
            self.a_bal.setChecked(True)
        self.model_status.setText(status_text)

    # ---------- CALIBRATION ----------
    def pixel_to_pdf_rect(self, r: QRectF) -> QRectF:
        """Convert a rectangle in image pixels to PDF points (72 dpi units)."""
        Wpx, Hpx = self.image_size_px
        Wpt, Hpt = self.page_size_pts
        if Wpx <= 0 or Hpx <= 0 or Wpt <= 0 or Hpt <= 0:
            return QRectF(r)
        sx = Wpt / float(Wpx)
        sy = Hpt / float(Hpx)
        return QRectF(r.left() * sx, r.top() * sy, r.width() * sx, r.height() * sy)

    def pdf_to_pixel_rect(self, r: QRectF) -> QRectF:
        """Convert a rectangle in PDF points to image pixels."""
        Wpx, Hpx = self.image_size_px
        Wpt, Hpt = self.page_size_pts
        if Wpx <= 0 or Hpx <= 0 or Wpt <= 0 or Hpt <= 0:
            return QRectF(r)
        sx = float(Wpx) / Wpt
        sy = float(Hpx) / Hpt
        return QRectF(r.left() * sx, r.top() * sy, r.width() * sx, r.height() * sy)

    # ---------- METRICS ----------
    def compute_quality_metrics(self, panels: List[Tuple[int,float,QRectF]], balloons: List[Tuple[int,float,QRectF]]) -> Dict[str, Any]:
        """
        Basic metrics inspired by your audit:
        - counts, area ratios, overlap and severe-overlap counts
        - quality_score = 1 - severe_ratio - 0.5*overlap_ratio (clamped to [0,1])
        Note: panels/balloons here are in *pixel space*; ratios are dimensionless.
        """
        Wpx, Hpx = self.image_size_px
        page_area = max(1e-6, float(Wpx * Hpx))
        metrics: Dict[str, Any] = {
            "page_index": self.page_index,
            "panels": len(panels),
            "balloons": len(balloons),
            "panel_area_ratios": [],
            "balloon_area_ratios": [],
            "overlaps": 0,
            "severe_overlaps": 0,
            "quality_score": 0.0,
        }
        for _, _, pr in panels:
            metrics["panel_area_ratios"].append(_area(pr) / page_area)
        for _, _, br in balloons:
            metrics["balloon_area_ratios"].append(_area(br) / page_area)

        total_pairs = max(1, len(panels) * len(balloons))
        for _, _, pr in panels:
            for _, _, br in balloons:
                iou = _iou(pr, br)
                # containment = fraction of balloon inside panel
                inter = pr.intersected(br)
                contain_pb = 0.0
                if not inter.isEmpty():
                    contain_pb = (inter.width() * inter.height()) / max(1e-6, _area(br))
                if iou > 0.10 or contain_pb > 0.60:
                    metrics["overlaps"] += 1
                if iou > 0.50 or contain_pb > 0.90:
                    metrics["severe_overlaps"] += 1

        overlap_ratio = metrics["overlaps"] / total_pairs
        severe_ratio = (metrics["severe_overlaps"] / max(1, metrics["overlaps"])) if metrics["overlaps"] else 0.0
        score = max(0.0, min(1.0, 1.0 - severe_ratio - 0.5 * overlap_ratio))
        metrics["quality_score"] = round(score, 4)
        return metrics

    # ---------- DETECTION PIPELINE ----------
    def _cfg(self, name, default):
        return type(default)(GLOBAL_CONFIG.get(name, default))

    def _refine_dets(self, panels, balloons):
        """Raffine les détections: seuils par classe, NMS par classe, priors de taille/marge,
        attache balloons→panel, clamping des quantités (lisibilité)."""
        from PySide6.QtCore import QRectF
        W, H = self.image_size_px
        page_area = max(1e-6, float(W * H))

        # défauts (surchargés par YAML si présents) - durcis pour éviter les panels fantômes
        PANEL_CONF          = self._cfg('panel_conf', 0.42)       # ↑ avant ~0.28
        BALLOON_CONF        = self._cfg('balloon_conf', 0.38)
        PANEL_AREA_MIN      = self._cfg('panel_area_min_pct', 0.040)  # 4% de la page
        PANEL_AREA_MAX      = self._cfg('panel_area_max_pct', 0.92)
        BALLOON_AREA_MIN    = self._cfg('balloon_area_min_pct', 0.0025)  # 0.25%
        BALLOON_AREA_MAX    = self._cfg('balloon_area_max_pct', 0.30)
        PANEL_NMS           = self._cfg('panel_nms_iou', 0.28)    # NMS un peu + strict
        BALLOON_NMS         = self._cfg('balloon_nms_iou', 0.25)
        MARGIN_INSET        = self._cfg('page_margin_inset_pct', 0.015)  # 1.5% bord exclu
        MIN_W               = int(self._cfg('min_box_w_px', 36))
        MIN_H               = int(self._cfg('min_box_h_px', 30))
        MAX_PANELS          = int(self._cfg('max_panels', 10))    # clamp + agressif
        MAX_BALLOONS        = int(self._cfg('max_balloons', 20))
        BALLOON_MIN_OVERLAP = self._cfg('balloon_min_overlap_panel', 0.06)  # 6% du ballon

        page_rect = QRectF(W * MARGIN_INSET, H * MARGIN_INSET,
                           W * (1 - 2 * MARGIN_INSET), H * (1 - 2 * MARGIN_INSET))

        def valid_box(c, p, r, area_min, area_max):
            if p < (PANEL_CONF if c == 0 else BALLOON_CONF): return False
            if r.width() < MIN_W or r.height() < MIN_H: return False
            ar = (r.width()*r.height()) / page_area
            if ar < area_min or ar > area_max: return False
            if r.intersected(page_rect).isEmpty(): return False
            return True

        # 1) Filtres taille/aire/marge
        panels   = [(c,p,r) for (c,p,r) in panels   if valid_box(c,p,r,PANEL_AREA_MIN,   PANEL_AREA_MAX)]
        balloons = [(c,p,r) for (c,p,r) in balloons if valid_box(c,p,r,BALLOON_AREA_MIN, BALLOON_AREA_MAX)]

        # 2) NMS par classe (seuils distincts)
        def nms_class_aware(dets, iou_thr):
            dets = sorted(dets, key=lambda x: x[1], reverse=True)
            kept = []
            while dets:
                cur = dets.pop(0)
                kept.append(cur)
                dets = [d for d in dets if (d[0] != cur[0]) or (_iou(cur[2], d[2]) < iou_thr)]
            return kept
        panels   = nms_class_aware(panels,   PANEL_NMS)
        balloons = nms_class_aware(balloons, BALLOON_NMS)

        # 3) Garder un ballon seulement s'il appartient à un panel
        if panels:
            kept_b = []
            for (c,p,r) in balloons:
                keep = False
                for (_,_,pr) in panels:
                    if pr.contains(r.center()): keep = True; break
                    inter = pr.intersected(r)
                    if not inter.isEmpty():
                        frac = (inter.width()*inter.height()) / max(1e-6, r.width()*r.height())
                        if frac >= BALLOON_MIN_OVERLAP: keep = True; break
                if keep: kept_b.append((c,p,r))
            balloons = kept_b

        # 4) Clamps pour lisibilité
        if len(panels) > MAX_PANELS:
            panels = sorted(panels, key=lambda t: t[2].width()*t[2].height(), reverse=True)[:MAX_PANELS]
        if len(balloons) > MAX_BALLOONS:
            balloons = sorted(balloons, key=lambda t: t[1], reverse=True)[:MAX_BALLOONS]

        return panels, balloons

    def _predict_np_rgb(self, np_rgb: np.ndarray, imgsz: int, conf: float, iou: float, max_det: int) -> List[Tuple[int, float, QRectF]]:
        if self.model is None:
            return []
        res = self.model.predict(
            source=np_rgb, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
            augment=False, verbose=False, device="mps" if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "1") else None
        )[0]
        out: List[Tuple[int, float, QRectF]] = []
        if res and getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, confs):
                if self.balloon_detection_disabled and c == self.balloon_class_index:
                    continue
                out.append((int(c), float(p), QRectF(float(x1), float(y1), float(x2 - x1), float(y2 - y1))))
        return out

    def _run_detection(self):
        if self._is_detecting:
            return
        self._is_detecting = True
        try:
            if self.qimage_current is None:
                self.dets = []
                self._draw_detections()
                return

            # Config
            PANEL_CONF = self._cfg('panel_conf', 0.18)
            BALLOON_CONF = self._cfg('balloon_conf', 0.22)
            PANEL_AREA_MIN = self._cfg('panel_area_min_pct', 0.02)
            PANEL_AREA_MAX = self._cfg('panel_area_max_pct', 0.80)
            BALLOON_AREA_MIN = self._cfg('balloon_area_min_pct', 0.0006)
            BALLOON_AREA_MAX = self._cfg('balloon_area_max_pct', 0.30)
            IOU_NMS = self._cfg('iou_merge', 0.25)
            MAX_DET = self._cfg('max_det', 400)
            IMGSZ_MAX = self._cfg('imgsz_max', 1536)
            TILE_TARGET = self._cfg('tile_target', 1024)
            TILE_OVERLAP = self._cfg('tile_overlap', 0.20)

            qimg = self.qimage_current
            W, H = qimg.width(), qimg.height()
            PAGE_AREA = float(W * H)
            imgsz = min(IMGSZ_MAX, max(W, H))
            imgsz = ((imgsz + 31) // 32) * 32

            mode = (getattr(self.cfg, "detect_mode", "hybrid") or "hybrid").lower()

            # Tiling grid
            nx = ny = 1
            if max(W, H) > TILE_TARGET * 1.15:
                nx = ny = 2 if max(W, H) < TILE_TARGET * 2.2 else 3

            def run_tiled_yolo() -> Tuple[List[Tuple[int,float,QRectF]], str]:
                if self.model is None:
                    return [], "N/A"
                dets_raw: List[Tuple[int, float, QRectF]] = []
                if nx == 1 and ny == 1:
                    dets_raw = self._predict_np_rgb(qimage_to_rgb(qimg), imgsz, min(PANEL_CONF, BALLOON_CONF), 0.6, MAX_DET)
                    tiles_str = "1x1"
                else:
                    tiles_str = f"{nx}x{ny}"
                    dx, dy = W / nx, H / ny
                    ox, oy = dx * TILE_OVERLAP, dy * TILE_OVERLAP
                    for iy in range(ny):
                        for ix in range(nx):
                            x1 = max(0, int(ix * dx - ox))
                            y1 = max(0, int(iy * dy - oy))
                            x2 = min(W, int((ix + 1) * dx + ox))
                            y2 = min(H, int((iy + 1) * dy + oy))
                            sub = qimg.copy(x1, y1, x2 - x1, y2 - y1)
                            preds = self._predict_np_rgb(qimage_to_rgb(sub), imgsz, min(PANEL_CONF, BALLOON_CONF), 0.6, MAX_DET)
                            for (c, p, r) in preds:
                                r.translate(x1, y1)
                                dets_raw.append((c, p, r))
                return dets_raw, tiles_str

            panels: List[Tuple[int,float,QRectF]] = []
            balloons: List[Tuple[int,float,QRectF]] = []
            tiles_str = "1x1"

            if mode == "rules_only":
                panels = [(0, 1.0, QRectF(0, 0, float(W), float(H)))]
            else:
                if self.model is None:
                    self.dets = []
                    self._draw_detections()
                    self.status.showMessage("YOLO model missing. Switch to RULES if needed.")
                    return
                dets_raw, tiles_str = run_tiled_yolo()
                # Filter by class & area
                panels = [(c, p, r) for (c, p, r) in dets_raw if c == 0 and PANEL_AREA_MIN <= (_area(r) / PAGE_AREA) <= PANEL_AREA_MAX and p >= PANEL_CONF]
                if not self.balloon_detection_disabled:
                    balloons = [(c, p, r) for (c, p, r) in dets_raw if c == self.balloon_class_index and BALLOON_AREA_MIN <= (_area(r) / PAGE_AREA) <= BALLOON_AREA_MAX and p >= BALLOON_CONF]

                # HYBRID == YOLO + class-aware NMS (coarse)
                panels = apply_nms(panels, IOU_NMS)
                balloons = apply_nms(balloons, IOU_NMS)
                # Raffinement fort (seuils par classe, NMS par classe, taille/marges, attach balloon→panel, clamps)
                panels, balloons = self._refine_dets(panels, balloons)

            # ---- Cover fallback: forcer 1 panel plein-page sur couvertures ----
            COVER_RULE_ENABLE          = self._cfg('cover_rule_enable', True)
            COVER_PAGES                = set(self._cfg('cover_pages', [0, 1]))  # 0-based
            COVER_MIN_BALLOONS         = int(self._cfg('cover_min_balloons', 0))
            COVER_FULLPAGE_MIN_PCT     = float(self._cfg('cover_fullpage_min_pct', 0.85))
            COVER_MAX_PANELS_BEFOREFORCE = int(self._cfg('cover_max_panels_before_force', 6))

            if COVER_RULE_ENABLE and self.page_index in COVER_PAGES:
                # Heuristique : peu/pas de balloons et trop de « petits panels » → c'est sans doute une cover
                W, H = self.image_size_px
                page_area = max(1e-6, float(W*H))
                largest_area_pct = max([(_area(r)/page_area) for (_,_,r) in panels], default=0.0)

                if len(balloons) <= COVER_MIN_BALLOONS and (
                    len(panels) >= COVER_MAX_PANELS_BEFOREFORCE or largest_area_pct < COVER_FULLPAGE_MIN_PCT
                ):
                    page_rect = QRectF(0, 0, float(W), float(H))
                    best_conf = max([p for (_,p,_) in panels], default=0.95)
                    panels   = [(0, best_conf, page_rect)]
                    balloons = []

            # Respect toggles
            if not self.show_panels:
                panels = []
            if self.balloon_detection_disabled or not self.show_balloons:
                balloons = []

            # Final det list in pixel space
            self.dets = panels + balloons
            self._draw_detections()

            # ---------- QUALITY METRICS ----------
            metrics = self.compute_quality_metrics(panels, balloons)

            # Optional JSON append
            if METRICS_OUT:
                try:
                    os.makedirs(os.path.dirname(METRICS_OUT) or ".", exist_ok=True)
                    # Append-as-list semantics
                    if os.path.exists(METRICS_OUT):
                        with open(METRICS_OUT, "r", encoding="utf-8") as f:
                            existing = json.load(f)
                        if not isinstance(existing, list):
                            existing = [existing]
                        existing.append(metrics)
                        with open(METRICS_OUT, "w", encoding="utf-8") as f:
                            json.dump(existing, f, indent=2)
                    else:
                        with open(METRICS_OUT, "w", encoding="utf-8") as f:
                            json.dump([metrics], f, indent=2)
                except Exception as e:
                    print(f"⚠️  Could not write metrics to {METRICS_OUT}: {e}")

            # Status
            btxt = f"balloons={len(balloons)}" if not self.balloon_detection_disabled else "balloons=DISABLED"
            self.status.showMessage(
                f"Page {self.page_index+1}: panels={len(panels)}, {btxt} | tiles={tiles_str} | quality={metrics['quality_score']:.3f}"
            )

            # ---------- OPTIONAL: PDF-space export for panels/balloons ----------
            # Example of how to convert pixel boxes to PDF points (for later use/exports)
            # pdf_panels = [self.pixel_to_pdf_rect(r) for (_,_,r) in panels]
            # pdf_balloons = [self.pixel_to_pdf_rect(r) for (_,_,r) in balloons]

            # Optional debug overlay
            if DEBUG_DETECT and DEBUG_OVERLAY_DIR is not None:
                save_debug_overlay(self.qimage_current, panels, balloons, f"page_{self.page_index}", "final")

        finally:
            self._is_detecting = False

    # ---------- Reading mode ----------
    def _prepare_reading_units(self):
        self.read_units = []
        if not self.dets:
            self.read_index = -1
            return
        panels = [r for (c, p, r) in self.dets if c == 0]
        balloons = [r for (c, p, r) in self.dets if c == 1]
        attached = set()
        for pr in panels:
            merged = QRectF(pr)
            for i, br in enumerate(balloons):
                if i in attached:
                    continue
                if not merged.intersected(br).isEmpty():
                    merged = merged.united(br)
                    attached.add(i)
            self.read_units.append(merged)
        for i, br in enumerate(balloons):
            if i not in attached:
                self.read_units.append(QRectF(br))
        # Sort reading units top-to-bottom, left-to-right
        self.read_units.sort(key=lambda r: (r.top(), r.left()))
        self.read_index = -1

    def next_step(self):
        if not self.cfg.reading_mode:
            self.next_page()
            return
        if not self.read_units:
            self.next_page()
            return
        if not self.fullpage_shown_on_page and self.cfg.show_full_page_before_first_panel and self.read_index < 0:
            self.fit_full_page()
            self.fullpage_shown_on_page = True
            return
        self.read_index += 1
        if self.read_index >= len(self.read_units):
            self.next_page()
            return
        self._focus_rect(self.read_units[self.read_index])

    def prev_step(self):
        if not self.cfg.reading_mode:
            self.prev_page()
            return
        if not self.read_units:
            self.prev_page()
            return
        if self.read_index <= 0:
            if self.cfg.show_full_page_before_first_panel and not self.fullpage_shown_on_page:
                self.fit_full_page()
                self.fullpage_shown_on_page = True
                self.read_index = -1
            else:
                self.prev_page()
            return
        self.read_index -= 1
        self._focus_rect(self.read_units[self.read_index])

    def _focus_rect(self, r: QRectF):
        if not self.pixmap_item:
            return
        pad = 0.08
        rr = QRectF(r)
        rr.adjust(-r.width() * pad, -r.height() * pad, r.width() * pad, r.height() * pad)
        rr = rr.intersected(self.pixmap_item.boundingRect())
        rect_scene = self.pixmap_item.mapRectToScene(rr)
        self.view.fitInView(rect_scene, Qt.AspectRatioMode.KeepAspectRatio)
        self.view.centerOn(rect_scene.center())

    def fit_full_page(self):
        if not self.pixmap_item:
            return
        self.view.fitInView(self.pixmap_item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.view.centerOn(self.pixmap_item)

    # ---------- toggles ----------
    def _toggle_panels(self, checked: bool):
        self.show_panels = checked
        self._draw_detections()
        self._prepare_reading_units()

    def _toggle_balloons(self, checked: bool):
        if self.balloon_detection_disabled and checked:
            if hasattr(self, 'a_bal'):
                self.a_bal.setChecked(False)
            return
        self.show_balloons = checked
        self._draw_detections()
        self._prepare_reading_units()

    def _change_detect_mode_idx(self, idx: int):
        data = self.mode_combo.itemData(idx)
        mode = data if data in ("hybrid", "yolo_only", "rules_only") else "hybrid"
        self.cfg.detect_mode = mode
        self._run_detection()
        self._prepare_reading_units()

# ---------------- ENTRY ----------------
def run_app():
    global GLOBAL_CONFIG, DEBUG_DETECT, DEBUG_OVERLAY_DIR, METRICS_OUT

    parser = argparse.ArgumentParser(description="AnComicsViewer - Minimal, stable viewer (Calib+Metrics)")
    parser.add_argument('--config', type=str, default='detect_with_merge.yaml', help='Path to detection configuration YAML file')
    parser.add_argument('--debug-detect', action='store_true', help='Enable detection debug mode')
    parser.add_argument('--save-debug-overlays', type=str, default='debug', help='Directory to save debug overlay images')
    parser.add_argument('--pdf', type=str, default=None, help='Open this PDF on launch')
    parser.add_argument('--page', type=int, default=0, help='Page index to open')
    parser.add_argument('--metrics-out', type=str, default=None, help='If set, append per-page metrics to this JSON file')

    args = parser.parse_args()

    # Resolve config path if relative
    config_path = None
    if args.config:
        candidate = args.config
        if not os.path.isabs(candidate):
            here = os.path.dirname(__file__)
            c1 = os.path.join(here, 'config', candidate)
            c2 = os.path.join(here, candidate)
            if os.path.exists(c1):
                candidate = c1
            elif os.path.exists(c2):
                candidate = c2
        if os.path.exists(candidate):
            config_path = candidate
        else:
            print(f"⚠️  Configuration file not found: {args.config}")

    GLOBAL_CONFIG = load_config(config_path)
    DEBUG_DETECT = args.debug_detect
    DEBUG_OVERLAY_DIR = args.save_debug_overlays
    METRICS_OUT = args.metrics_out

    app = QApplication(sys.argv)
    viewer = PdfYoloViewer()

    # Auto-open PDF if provided
    if args.pdf and fitz is not None and os.path.exists(args.pdf):
        try:
            viewer.pdf = fitz.open(args.pdf)
            viewer.page_index = max(0, min(args.page, len(viewer.pdf) - 1))
            viewer.load_page(viewer.page_index)
        except Exception as e:
            QMessageBox.critical(viewer, "PDF error", str(e))

    viewer.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(run_app())
