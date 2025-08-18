"""AnComicsViewer — minimal PDF comics reader with heuristic panel detection.

Small, single-file viewer combining PySide6, QPdfView and OpenCV heuristics to detect
rectangular comic/manga panels and provide a tuning dialog to iteratively adjust rules.

Quick usage:
    python AnComicsViewer.py

Main components:
    - PanelDetector: OpenCV-based heuristic detector (adaptive threshold, morphology,
        light/gutter splitting and recursive projection splitting).
    - PanelTuningDialog: interactive UI to adjust tunables and re-run detection.
    - ComicsView / main window: PDF viewer with panel overlay and navigation.
    - Helpers: QImage↔numpy conversions and a safe `pdebug` logger.

This file is intentionally compact; see `README.md` for install/run notes and macOS tips.
"""

# Table of contents (high-level)
#  - Imports & environment tweaks
#  - Debug helpers: pdebug, dbg
#  - QImage <-> numpy helpers
#  - PanelDetector (detection pipeline & helpers)
#  - UI components: PanelTuningDialog, ComicsView, app entry
#  - Utilities, smoke tests and helper scripts

import os
import sys
import traceback
from typing import Optional, List

# Ensure required packages are imported
try:
    import numpy as np
except ImportError:
    np = None
try:
    import cv2
except ImportError:
    cv2 = None

def pdebug(*parts):
    """Safe console logger for Panels; never crashes on multi-lines."""
    try:
        sys.stdout.write("[Panels] " + " ".join(map(str, parts)) + "\n")
        sys.stdout.flush()
    except Exception:
        pass

# --- Environment tweaks for macOS & quiet logs ---
try:
    import warnings
    warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
    if sys.platform == "darwin" and os.environ.get("QT_MAC_WANTS_LAYER") is None:
        os.environ["QT_MAC_WANTS_LAYER"] = "1"
except Exception:
    pass

from PySide6.QtCore import Qt, QPoint, QPointF, QSize, QSizeF, QMimeData, QRectF, QTimer
from PySide6.QtGui import (
    QAction,
    QKeySequence,
    QGuiApplication,
    QPainter,
    QColor,
    QPen,
    QImage,
    QIcon,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QStyle,
    QToolBar,
    QToolButton,
    QWidget,
    QMenu,
    QDialog, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox,
    QPushButton, QVBoxLayout, QHBoxLayout,
)

# PDF modules
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView

# Optional CV deps (used for heuristic panel detection)
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
    np = None


# -----------------------------
# Debug helper
# -----------------------------
def dbg(enabled: bool, *args):
    if enabled:
        pdebug(*args)


# -----------------------------
# QImage -> numpy helper (robust for PySide6 memoryview)
# -----------------------------
def qimage_to_numpy_rgba8888(img: QImage):
    """Return a numpy array HxWx4 (uint8) from a QImage as RGBA8888.
    Works with PySide6 where QImage.constBits() returns a memoryview (no setsize()).
    """
    if img.isNull():
        return None
    if img.format() != QImage.Format.Format_RGBA8888:
        img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    mv = img.constBits()  # memoryview
    try:
        # Use bytes() for memoryview conversion
        buf = bytes(mv)
        arr = np.frombuffer(buf, dtype=np.uint8)
        arr = arr.reshape((h, bpl))
        arr = arr[:, : w * 4]  # drop stride padding
        arr = arr.reshape((h, w, 4))
        return arr
    except Exception as e:
        pdebug(f"QImage conversion failed: {e}")
        return None


# -----------------------------
# Panel Detector (heuristic v1)
# -----------------------------
class PanelDetector:
    """Detect comic panels using a fast OpenCV heuristic with optional Canny fallback.

    Steps (bright gutters typical):
      1) Adaptive threshold (INV)
      2) Morphological close
      3) Contours -> bounding boxes
      4) Filters: area %, min size, fill ratio

    If nothing is found, optional Canny fallback tries edge-based detection.

    Returns: list of QRectF in *page points* coordinates (72 dpi space).
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

        # --- Adaptive & morpho ---
        self.adaptive_block = 51      # odd (51/61) => window for adaptive threshold
        self.adaptive_C     = 5       # offset for adaptive threshold
        self.morph_kernel   = 7       # morphology kernel size
        self.morph_iter     = 2       # morphology iterations

        # --- Debug (last splits) ---
        self.last_debug = {"v": [], "h": []}

        # --- Base filters ---
        self.min_area_pct = 0.015   # 1.5% of page area
        self.max_area_pct = 0.95    # exclude nearly full-bleed
        self.min_fill_ratio = 0.55  # area(contour)/area(bbox) (hardened)
        self.min_rect_px = 80       # width/height min in px @ detection DPI
        # Fractional thresholds (stable across DPI)
        self.min_rect_frac = 0.055   # ~5.5% of the page min side
        self.min_gutter_frac = 0.015 # ~1.5% local gutter fraction
        self.proj_smooth_k = 17      # smoothing kernel for projections (anti-glyph splits)
        # legacy alias used elsewhere
        self.min_panel_px = self.min_rect_px

        self.use_canny_fallback = True
        self.reading_rtl = False  # set True for manga-style reading (right-to-left)

        # --- gutter / light split params ---
        # Gutter-based (binary mask runs)
        self.gutter_v_thresh = 0.88   # fraction of column pixels considered "gutter"
        self.gutter_h_thresh = 0.88   # fraction of row pixels considered "gutter"

        # Brightness-based (LAB L channel projections)
        self.light_col_rel = 0.15   # relative threshold for column means (0..1)
        self.light_row_rel = 0.15   # relative threshold for row means (0..1)

        # --- gutter split (brightness-based) – stricter rules ---
        self.min_gutter_px  = 12       # min gutter thickness (px)
        self.max_gutter_px_frac = 0.05  # gutter <= 5% of block size
        self.gutter_cov_min = 0.93    # minimal brightness coverage for gutter
        self.edge_margin_frac = 0.03  # avoid cutting too close to edges (3%)
        self.max_panels_per_page = 24 # safety limit

        # --- title-row filter (on/off + thresholds) ---
        self.filter_title_rows = True
        self.title_row_top_frac   = 0.20  # row located in the top 20% of the page
        self.title_row_max_h_frac = 0.10  # median row height < 10% page
        self.title_row_median_w_frac_max = 0.25  # median width < 25% page
        self.title_row_min_boxes  = 4     # at least 4 boxes (characters)
        # Additional title-row heuristic for pages with fewer but wider title boxes
        self.title_row_big_min_boxes = 2    # cas B : "peu de gros"
        self.title_row_big_w_min_frac = 0.16
        self.title_row_min_meanL  = 0.88  # row is globally bright (0..1 after /255)

    # --- Heuristic main ---
    def detect_panels(self, qimage, page_point_size):
        if cv2 is None or np is None:
            pdebug("OpenCV/NumPy not available -> no detection")
            return []
        try:
            # reset debug splits for overlay
            self.last_debug = {"v": [], "h": []}

            # Log key parameters so we can confirm tunables are applied when re-running
            pdebug(
                "params:",
                f"ab={self.adaptive_block} C={self.adaptive_C}",
                f"mk={self.morph_kernel} mi={self.morph_iter}",
                f"min_area={self.min_area_pct:.3f} max_area={self.max_area_pct:.2f}",
                f"min_fill={self.min_fill_ratio:.2f} min_px={self.min_rect_px}",
                f"light_col={self.light_col_rel:.2f} light_row={self.light_row_rel:.2f}",
                    f"gutter:min_px={self.min_gutter_px} cov>={self.gutter_cov_min:.2f} max_frac={self.max_gutter_px_frac:.2f}",
                    f"title: on={self.filter_title_rows} top<{self.title_row_top_frac:.2f} h<{self.title_row_max_h_frac:.2f} boxes>={self.title_row_min_boxes} L>={self.title_row_min_meanL:.2f}",
                    f"psk={getattr(self, 'proj_smooth_k', 17)}"
            )

            arr = qimage_to_numpy_rgba8888(qimage)
            if arr is None:
                pdebug("QImage conversion failed")
                return []
            h, w = arr.shape[:2]
            gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
            pdebug(f"Converted to gray: {w}x{h}")

            # Image de luminosité (LAB + CLAHE) pour détecter les gouttières lumineuses
            rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            L, A, B = cv2.split(lab)
            L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)  # uint8

            # Route 1: adaptive
            th = self._adaptive_route(gray)
            rects = self._rects_from_mask(th, w, h, page_point_size)
            pdebug(f"Adaptive route -> {len(rects)} rects")

            # Route 2: LAB si rien
            if not rects:
                th2 = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 51, 5)
                th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)
                rects = self._rects_from_mask(th2, w, h, page_point_size)
                pdebug(f"LAB route -> {len(rects)} rects")

            # Route 3: Canny si rien
            if not rects and self.use_canny_fallback:
                pdebug("Trying Canny fallback ...")
                th3 = self._canny_route(gray)
                rects = self._rects_from_mask(th3, w, h, page_point_size)
                pdebug(f"Canny route -> {len(rects)} rects")

            # Post: découper les bandes en cases via la luminosité L
            if rects:
                rects = self._split_rects_by_light(L, rects, w, h, page_point_size)
                pdebug(f"After light split -> {len(rects)} rects")

            # Optional: filter out title/header rows detected at top of page
            if rects and getattr(self, 'filter_title_rows', False):
                rects = self._filter_title_rows(L, rects, w, h, page_point_size)
                pdebug(f"After title-row filter -> {len(rects)} rects")

            # Ordre de lecture
            if self.reading_rtl:
                rects.sort(key=lambda r: (r.top(), -r.left()))
            else:
                rects.sort(key=lambda r: (r.top(), r.left()))
            return rects

        except Exception:
            pdebug("detect_panels error:\n" + traceback.format_exc())
            return []

    # --- Adaptive route ---
    def _adaptive_route(self, gray):
        k = int(self.adaptive_block) | 1
        C = int(self.adaptive_C)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, k, C
        )
        kernel = np.ones((int(self.morph_kernel), int(self.morph_kernel)), np.uint8)
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=int(self.morph_iter))
        return closed

    # --- Canny fallback ---
    def _canny_route(self, gray):
        edges = cv2.Canny(gray, 60, 180)
        kernel = np.ones((5, 5), np.uint8)
        dil = cv2.dilate(edges, kernel, iterations=2)
        closed = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kernel, iterations=2)
        return closed

    # --- Contours -> rects in page points ---
    def _rects_from_mask(self, mask, w, h, page_point_size: QSizeF) -> List[QRectF]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return []
        page_area_px = float(w * h)
        s = w / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0  # px per point
        # dynamic minimum size in pixels, stable across DPI
        min_px_dyn = max(self.min_rect_px, int(self.min_rect_frac * min(w, h)))
        rects: List[QRectF] = []
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            area_px = cw * ch
            area_pct = area_px / page_area_px
            if area_pct < self.min_area_pct or area_pct > self.max_area_pct:
                continue
            c_area = float(cv2.contourArea(c))
            if c_area <= 0:
                continue
            fill = c_area / float(area_px)
            # Use the tunable fill ratio instead of a hard-coded constant so the slider does something
            if fill < self.min_fill_ratio:
                continue
            if cw < min_px_dyn or ch < min_px_dyn:
                continue
            # Convert to page points
            px = x / s; py = y / s; pw = cw / s; ph = ch / s
            rects.append(QRectF(px, py, pw, ph))
        rects = self._merge_rects(rects, iou_thresh=0.25)
        return rects

    def _merge_rects(self, rects: List[QRectF], iou_thresh: float) -> List[QRectF]:
        def iou(a: QRectF, b: QRectF) -> float:
            inter = a.intersected(b)
            if inter.isEmpty():
                return 0.0
            ia = inter.width() * inter.height()
            ua = a.width() * a.height() + b.width() * b.height() - ia
            return ia / ua if ua > 0 else 0.0

        merged: List[QRectF] = []
        for r in rects:
            did_merge = False
            for i, m in enumerate(merged):
                if iou(r, m) >= iou_thresh:
                    u = QRectF(
                        min(r.left(), m.left()),
                        min(r.top(), m.top()),
                        max(r.right(), m.right()) - min(r.left(), m.left()),
                        max(r.bottom(), m.bottom()) - min(r.top(), m.top()),
                    )
                    merged[i] = u
                    did_merge = True
                    break
            if not did_merge:
                merged.append(r)
        return merged

    def _split_rects_by_gutters(self, panel_mask, rects_pts, W, H, page_point_size):
        """Convertit rects (points) -> pixels, split par gouttières, puis reconvertit."""
        gutter = 255 - panel_mask  # 255 = gouttière claire
        s = W / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0  # px/point
        out: List[QRectF] = []
        for rp in rects_pts:
            x0 = max(0, min(int(round(rp.left()   * s)), W - 1))
            y0 = max(0, min(int(round(rp.top()    * s)), H - 1))
            ww = max(1, min(int(round(rp.width()  * s)), W - x0))
            hh = max(1, min(int(round(rp.height() * s)), H - y0))
            roi = gutter[y0:y0+hh, x0:x0+ww]
            out += self._split_by_gutters(roi, x0, y0, s)
        return out

    def _split_rects_by_light(self, L_img, rects_pts, W, H, page_point_size):
        """Découpe chaque bande en cases en détectant les gouttières lumineuses dans L."""
        s = W / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0  # px/pt
        out: List[QRectF] = []
        for rp in rects_pts:
            x0 = max(0, min(int(round(rp.left()   * s)), W - 1))
            y0 = max(0, min(int(round(rp.top()    * s)), H - 1))
            ww = max(1, min(int(round(rp.width()  * s)), W - x0))
            hh = max(1, min(int(round(rp.height() * s)), H - y0))
            roi = L_img[y0:y0+hh, x0:x0+ww]
            out += self._split_by_light_recursive(roi, x0, y0, s)
        return out

    def _split_by_light_recursive(self, roi, x0, y0, s):
        """
        Split récursif d'un ROI via projections de luminosité (canal L),
        avec contraintes fortes pour éviter les splits sur bulles/blancs internes.
        """
        from PySide6.QtCore import QRectF
        import numpy as np

        h, w = roi.shape[:2]
        stack = [(0, 0, w, h)]
        res: List[QRectF] = []

        while stack:
            x, y, ww, hh = stack.pop()

            # Stop conditions
            if ww < self.min_panel_px or hh < self.min_panel_px:
                res.append(QRectF((x0+x)/s, (y0+y)/s, ww/s, hh/s)); continue
            # garde-fou global
            if len(res) + len(stack) >= self.max_panels_per_page:
                res.append(QRectF((x0+x)/s, (y0+y)/s, ww/s, hh/s)); continue

            sub = roi[y:y+hh, x:x+ww].astype(np.float32) / 255.0

            # Projections (lissées)
            col_mean = sub.mean(axis=0)
            row_mean = sub.mean(axis=1)
            k = max(9, int(getattr(self, 'proj_smooth_k', 9)))
            if ww >= k: col_mean = np.convolve(col_mean, np.ones(k)/k, mode="same")
            if hh >= k: row_mean = np.convolve(row_mean, np.ones(k)/k, mode="same")

            # Seuils relatifs
            ct = (col_mean.max() - col_mean.min()) * self.light_col_rel + col_mean.mean()
            rt = (row_mean.max() - row_mean.min()) * self.light_row_rel + row_mean.mean()
            vcand = col_mean >= ct
            hcand = row_mean >= rt

            # Runs candidats
            v_runs = self._find_runs(vcand, self.min_gutter_px)
            h_runs = self._find_runs(hcand, self.min_gutter_px)

            # Contraintes supplémentaires
            edge_margin_x = int(round(ww * self.edge_margin_frac))
            edge_margin_y = int(round(hh * self.edge_margin_frac))
            max_gutter_w  = max(self.min_gutter_px, int(round(ww * self.max_gutter_px_frac)))
            max_gutter_h  = max(self.min_gutter_px, int(round(hh * self.max_gutter_px_frac)))
            min_gutter_w  = max(self.min_gutter_px, int(round(ww * getattr(self, 'min_gutter_frac', 0.015))))
            min_gutter_h  = max(self.min_gutter_px, int(round(hh * getattr(self, 'min_gutter_frac', 0.015))))

            def filter_v(run):
                a, b = run; span = b - a
                if span < min_gutter_w or span > max_gutter_w: return None
                if a <= edge_margin_x or b >= ww - edge_margin_x:   return None
                cov = sub[:, a:b].mean()  # couverture lumineuse verticale
                if cov < self.gutter_cov_min: return None
                cut = (a + b) // 2
                if cut < self.min_panel_px or (ww - cut) < self.min_panel_px: return None
                return (span, cut)

            def filter_h(run):
                a, b = run; span = b - a
                if span < min_gutter_h or span > max_gutter_h: return None
                if a <= edge_margin_y or b >= hh - edge_margin_y:   return None
                cov = sub[a:b, :].mean()  # couverture lumineuse horizontale
                if cov < self.gutter_cov_min: return None
                cut = (a + b) // 2
                if cut < self.min_panel_px or (hh - cut) < self.min_panel_px: return None
                return (span, cut)

            # Choisir la coupe la plus plausible (span max)
            best = None
            if v_runs:
                v_opts = [filter_v(r) for r in v_runs]
                v_opts = [o for o in v_opts if o is not None]
                if v_opts:
                    v_span, v_cut = max(v_opts, key=lambda t: t[0])
                    best = ('v', v_cut, v_span)
            if h_runs:
                h_opts = [filter_h(r) for r in h_runs]
                h_opts = [o for o in h_opts if o is not None]
                if h_opts:
                    h_span, h_cut = max(h_opts, key=lambda t: t[0])
                    if (best is None) or (h_span > best[2]):
                        best = ('h', h_cut, h_span)

            if best is None:
                # Pas de gouttière “franche” -> bloc accepté tel quel
                res.append(QRectF((x0+x)/s, (y0+y)/s, ww/s, hh/s))
            else:
                axis, cut, _ = best
                # record debug split lines (in page points)
                if getattr(self, "last_debug", None) is None:
                    self.last_debug = {"v": [], "h": []}
                if axis == 'v':
                    # vertical line at cut
                    self.last_debug["v"].append(((x0 + x + cut) / s, (y0 + y) / s, 0.0, hh / s))
                else:
                    # horizontal line at cut
                    self.last_debug["h"].append(((x0 + x) / s, (y0 + y + cut) / s, ww / s, 0.0))
                if axis == 'v':
                    stack.append((x + cut, y, ww - cut, hh))
                    stack.append((x, y, cut, hh))
                else:
                    stack.append((x, y + cut, ww, hh - cut))
                    stack.append((x, y, ww, cut))

        # Filtre final
        final: List[QRectF] = []
        for r in res:
            if r.width()  * s >= self.min_panel_px and r.height() * s >= self.min_panel_px:
                final.append(r)
        return final

    def _filter_title_rows(self, L_img, rects_pts, W, H, page_point_size):
        """Supprime les rangées de 'titre' (top, basses, claires, multi-boîtes)."""
        import numpy as np
        s = W / float(page_point_size.width()) if page_point_size.width() > 0 else 1.0

        # Rects en pixels pour grouper par rangée
        items = [(int(r.left()*s), int(r.top()*s), int(r.width()*s), int(r.height()*s), r) for r in rects_pts]
        items.sort(key=lambda t: t[1])  # par y

        # Groupement simple par recouvrement vertical
        rows = []  # [y0, y1, [items...]]
        for x,y,w,h,r in items:
            placed = False
            for row in rows:
                ry0, ry1, lst = row
                # chevauchement vertical ?
                if not (y+h < ry0 or y > ry1):
                    row[0] = min(ry0, y)
                    row[1] = max(ry1, y+h)
                    lst.append((x,y,w,h,r))
                    placed = True
                    break
            if not placed:
                rows.append([y, y+h, [(x,y,w,h,r)]])

        keep = []
        page_h = float(H)
        for ry0, ry1, lst in rows:
            row_h = max(1, ry1 - ry0)
            y_center_frac  = ((ry0 + ry1)/2.0) / page_h
            median_h_frac  = (np.median([h for _,_,_,h,_ in lst]) / page_h)
            median_w_frac  = (np.median([w for _,_,w,_,_ in lst]) / float(W)) if lst else 0.0
            count          = len(lst)
            # luminosité moyenne de la bande (canal L après CLAHE)
            band = L_img[max(0,ry0):min(H,ry1), :]
            meanL = float(band.mean()) / 255.0 if band.size else 0.0

            # Two scenarios: many small boxes OR few wider boxes
            Wf = float(W)
            med_w_frac = median_w_frac

            many_small = (count >= self.title_row_min_boxes and
                          med_w_frac <= getattr(self, 'title_row_median_w_frac_max', 0.25))

            few_big = (count >= getattr(self, 'title_row_big_min_boxes', 2) and
                       med_w_frac >= getattr(self, 'title_row_big_w_min_frac', 0.16))

            drop = (
                y_center_frac < self.title_row_top_frac and
                median_h_frac < self.title_row_max_h_frac and
                meanL >= self.title_row_min_meanL and
                (many_small or few_big)
            )
            if getattr(self, 'debug', False):
                pdebug(f"[title-row] y={y_center_frac:.2f} h={median_h_frac:.3f} n={count} "
                       f"medW={med_w_frac:.2f} L={meanL:.2f} -> "
                       f"{'DROP' if drop else 'KEEP'} (many_small={many_small} few_big={few_big})")
            if drop:
                # on ignore toute la rangée (probablement un titre)
                continue
            for it in lst:
                keep.append(it[4])

        return keep

    def _split_by_gutters(self, gutter_roi, x0, y0, s):
        """Split récursif d'un ROI selon gouttières dominantes verticales/horizontales."""
        from PySide6.QtCore import QRectF
        h, w = gutter_roi.shape[:2]
        stack = [(0, 0, w, h)]
        result: List[QRectF] = []

        while stack:
            x, y, ww, hh = stack.pop()
            if ww < self.min_panel_px or hh < self.min_panel_px:
                result.append(QRectF((x0+x)/s, (y0+y)/s, ww/s, hh/s))
                continue

            sub = gutter_roi[y:y+hh, x:x+ww]
            col_ratio = (sub == 255).sum(axis=0).astype(float) / float(hh)  # fraction gouttière par colonne
            row_ratio = (sub == 255).sum(axis=1).astype(float) / float(ww)  # fraction gouttière par ligne

            v_runs = self._find_runs(col_ratio > self.gutter_v_thresh, self.min_gutter_px)
            h_runs = self._find_runs(row_ratio > self.gutter_h_thresh, self.min_gutter_px)

            best = None
            v_span = -1
            h_span = -1
            if v_runs:
                vi = int(np.argmax([b-a for a,b in v_runs]))
                va, vb = v_runs[vi]; v_span = vb - va
                v_cut = (va + vb) // 2
                if v_cut >= self.min_panel_px and (ww - v_cut) >= self.min_panel_px:
                    best = ('v', v_cut)
            if h_runs:
                hi = int(np.argmax([b-a for a,b in h_runs]))
                ha, hb = h_runs[hi]; h_span = hb - ha
                h_cut = (ha + hb) // 2
                if h_cut >= self.min_panel_px and (hh - h_cut) >= self.min_panel_px:
                    if (best is None) or (h_span > v_span):
                        best = ('h', h_cut)

            if best is None:
                result.append(QRectF((x0+x)/s, (y0+y)/s, ww/s, hh/s))
            else:
                axis, cut = best
                if axis == 'v':
                    stack.append((x + cut, y, ww - cut, hh))
                    stack.append((x, y, cut, hh))
                else:
                    stack.append((x, y + cut, ww, hh - cut))
                    stack.append((x, y, ww, cut))

        # Filtre final
        filtered: List[QRectF] = []
        for r in result:
            if r.width()  * s >= self.min_panel_px and r.height() * s >= self.min_panel_px:
                filtered.append(r)
        return filtered

    def _find_runs(self, bool_arr, min_len):
        """Retourne la liste des (start, end) de True consécutifs de longueur >= min_len."""
        runs = []
        start = None
        for i, val in enumerate(list(bool_arr) + [False]):  # sentinelle pour fermer
            if val and start is None:
                start = i
            elif (not val) and start is not None:
                if i - start >= min_len:
                    runs.append((start, i))
                start = None
        return runs


# -----------------------------
# Custom PDF View with Panning, Zoom-on-wheel, and Panel Overlay
# -----------------------------
class PannablePdfView(QPdfView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._panning = False
        self._last_pan_point = QPoint()
        self.setPageMode(QPdfView.PageMode.MultiPage)
        # Overlay
        self._overlay_enabled = False
        self._overlay_rects: List[QRectF] = []  # in page points for CURRENT page only

    # --- Pan with left click when scrollbars are available ---
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
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
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

    # --- Ctrl + Wheel for zoom ---
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
        h = self.horizontalScrollBar()
        v = self.verticalScrollBar()
        return (h and h.maximum() > h.minimum()) or (v and v.maximum() > v.minimum())

    # --- Context menu (export) ---
    def _show_context_menu(self, global_pos):
        menu = QMenu(self)
        export = menu.addAction("📄 Export Page as PNG")
        export.triggered.connect(self._export_current_page)
        menu.exec(global_pos)

    def _export_current_page(self):
        main = self.window()
        if hasattr(main, "export_current_page"):
            main.export_current_page()

    # --- Panel overlay API ---
    def set_panel_overlay(self, rects: List[QRectF], enabled: bool):
        self._overlay_rects = rects or []
        self._overlay_enabled = bool(enabled)
        self.viewport().update()

    # --- Helpers mapping page-points <-> viewport pixels ---
    def _page_to_view_xy(self, x_pt: float, y_pt: float) -> "tuple[float, float]":
        doc = self.document()
        if not doc:
            return (0.0, 0.0)
        cur = self.pageNavigator().currentPage()
        page_pts = doc.pagePointSize(cur)
        z = self.zoomFactor()
        vw = self.viewport().width()
        vh = self.viewport().height()
        content_w = page_pts.width() * z
        content_h = page_pts.height() * z
        pad_x = max(0.0, (vw - content_w) / 2.0)
        pad_y = max(0.0, (vh - content_h) / 2.0)
        sx = self.horizontalScrollBar().value()
        sy = self.verticalScrollBar().value()
        x = pad_x + (x_pt * z) - sx
        y = pad_y + (y_pt * z) - sy
        return (x, y)

    def _page_rect_to_view(self, r: QRectF) -> QRectF:
        x, y = self._page_to_view_xy(r.left(), r.top())
        w = r.width() * self.zoomFactor()
        h = r.height() * self.zoomFactor()
        return QRectF(x, y, w, h)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._overlay_enabled or not self._overlay_rects:
            return
        try:
            painter = QPainter(self.viewport())
            
            cur = self.pageNavigator().currentPage()
            doc = self.document()
            if not doc or cur < 0:
                return
            page_pts = doc.pagePointSize(cur)
            if page_pts.width() <= 0 or page_pts.height() <= 0:
                return

            # --- Cadre de page (debug) ---
            page_rect_view = self._page_rect_to_view(QRectF(0, 0, page_pts.width(), page_pts.height()))
            p2 = QPen(QColor(0, 120, 255, 200), 2)  # bleu pour le cadre de page
            painter.setPen(p2); painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(page_rect_view)

            # --- Panels ---
            pen = QPen(QColor(0, 200, 0, 220), 2)
            fill = QColor(0, 200, 0, 55)
            painter.setPen(pen); painter.setBrush(fill)
            for idx, r in enumerate(self._overlay_rects):
                if r.isEmpty(): 
                    continue
                vr = self._page_rect_to_view(r)
                painter.drawRect(vr)
                # index en surimpression
                painter.setPen(QPen(QColor(0,0,0,255), 1))
                painter.drawText(vr.adjusted(3, 3, -3, -3), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, f"#{idx+1}")
                painter.setPen(pen)

            # --- Debug: split lines (yellow dashed) ---
            main = self.window()
            dbginfo = getattr(getattr(main, "_panel_detector", None), "last_debug", None)
            if main and getattr(main, "_debug_panels", False) and dbginfo:
                pen_dbg = QPen(QColor(255, 215, 0, 230), 2, Qt.PenStyle.DashLine)
                painter.setPen(pen_dbg); painter.setBrush(Qt.BrushStyle.NoBrush)
                for (x_pt, y_pt, w_pt, h_pt) in dbginfo.get("v", []):
                    r = QRectF(x_pt, y_pt, max(0.5, w_pt), h_pt)
                    vr = self._page_rect_to_view(r)
                    painter.drawLine(vr.left(), vr.top(), vr.left(), vr.bottom())
                for (x_pt, y_pt, w_pt, h_pt) in dbginfo.get("h", []):
                    r = QRectF(x_pt, y_pt, w_pt, max(0.5, h_pt))
                    vr = self._page_rect_to_view(r)
                    painter.drawLine(vr.left(), vr.top(), vr.right(), vr.top())
        except Exception:
            pass


class PanelTuningDialog(QDialog):
    def __init__(self, parent, det: PanelDetector, det_dpi: float):
        super().__init__(parent)
        self.setWindowTitle("Panel tuning")
        self.det = det
        self.det_dpi = det_dpi

        form = QFormLayout()

        def dsb(v, mi, ma, step=0.01):
            w = QDoubleSpinBox()
            w.setRange(mi, ma)
            w.setDecimals(3)
            w.setSingleStep(step)
            w.setValue(v)
            return w

        def isb(v, mi, ma, step=1):
            w = QSpinBox()
            w.setRange(mi, ma)
            w.setSingleStep(step)
            w.setValue(v)
            return w

        def cb(v):
            w = QCheckBox()
            w.setChecked(v)
            return w

        # DPI
        self.w_dpi = isb(int(det_dpi), 72, 400, 10)
        form.addRow("Detection DPI", self.w_dpi)

        # Adaptive & morpho
        self.w_ab = isb(int(det.adaptive_block), 15, 101, 2)
        self.w_aC = isb(int(det.adaptive_C), 0, 20, 1)
        self.w_mk = isb(int(det.morph_kernel), 3, 15, 2)
        self.w_mi = isb(int(det.morph_iter), 1, 5, 1)
        form.addRow("Adaptive block (odd)", self.w_ab)
        form.addRow("Adaptive C", self.w_aC)
        form.addRow("Morph kernel", self.w_mk)
        form.addRow("Morph iter", self.w_mi)

        # Base boxes
        self.w_min_area = dsb(self.det.min_area_pct, 0.001, 0.2, 0.005)
        self.w_max_area = dsb(self.det.max_area_pct, 0.3, 0.99, 0.01)
        self.w_fill = dsb(self.det.min_fill_ratio, 0.10, 0.90, 0.01)
        self.w_min_px = isb(int(self.det.min_rect_px), 10, 200, 2)
        self.w_min_frac = dsb(getattr(self.det, 'min_rect_frac', 0.055), 0.01, 0.2, 0.005)
        form.addRow("Min area %", self.w_min_area)
        form.addRow("Max area %", self.w_max_area)
        form.addRow("Min fill ratio", self.w_fill)
        form.addRow("Min rect px", self.w_min_px)
        form.addRow("Min rect frac", self.w_min_frac)

        # Gutters / light
        self.w_lcr = dsb(self.det.light_col_rel, 0.00, 0.50, 0.01)
        self.w_lrr = dsb(self.det.light_row_rel, 0.00, 0.50, 0.01)
        self.w_cov = dsb(self.det.gutter_cov_min, 0.50, 0.99, 0.01)
        self.w_ming = isb(int(self.det.min_gutter_px), 1, 50, 1)
        self.w_maxgf = dsb(self.det.max_gutter_px_frac, 0.01, 0.20, 0.005)
        self.w_ming_frac = dsb(getattr(self.det, 'min_gutter_frac', 0.015), 0.001, 0.1, 0.001)
        self.w_emf = dsb(self.det.edge_margin_frac, 0.00, 0.20, 0.005)
        self.w_maxp = isb(int(self.det.max_panels_per_page), 4, 64, 1)
        # NEW: smoothing des projections (impair)
        self.w_psk = isb(int(getattr(self.det, 'proj_smooth_k', 17)), 5, 51, 2)
        form.addRow("Light col rel", self.w_lcr)
        form.addRow("Light row rel", self.w_lrr)
        form.addRow("Gutter cover min", self.w_cov)
        form.addRow("Min gutter px", self.w_ming)
        form.addRow("Min gutter frac", self.w_ming_frac)
        form.addRow("Max gutter frac", self.w_maxgf)
        form.addRow("Edge margin frac", self.w_emf)
        form.addRow("Max panels / page", self.w_maxp)
        form.addRow("Proj. smooth (odd px)", self.w_psk)

        # Title-row
        self.w_ftitle = cb(self.det.filter_title_rows)
        self.w_topf = dsb(self.det.title_row_top_frac, 0.05, 0.40, 0.01)
        self.w_maxhf = dsb(self.det.title_row_max_h_frac, 0.05, 0.30, 0.01)
        self.w_minb = isb(int(self.det.title_row_min_boxes), 1, 12, 1)
        self.w_minL = dsb(self.det.title_row_min_meanL, 0.40, 0.99, 0.01)
        self.w_title_medw = dsb(getattr(self.det, 'title_row_median_w_frac_max', 0.25), 0.05, 0.6, 0.01)
        form.addRow("Filter title rows", self.w_ftitle)
        form.addRow("Title top frac", self.w_topf)
        form.addRow("Title max h frac", self.w_maxhf)
        form.addRow("Title min boxes", self.w_minb)
        form.addRow("Title min mean L", self.w_minL)
        form.addRow("Title median w frac", self.w_title_medw)

        # Buttons
        btn_apply = QPushButton("Apply && Re-run")
        btn_close = QPushButton("Close")
        hb = QHBoxLayout()
        hb.addWidget(btn_apply)
        hb.addStretch(1)
        hb.addWidget(btn_close)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(hb)

        btn_apply.clicked.connect(self._apply)
        btn_close.clicked.connect(self.accept)

    def _apply(self):
        d = self.det
        # DPI
        self.det_dpi = float(self.w_dpi.value())
        # Adaptive & morpho
        d.adaptive_block = int(self.w_ab.value()) | 1
        d.adaptive_C = int(self.w_aC.value())
        d.morph_kernel = int(self.w_mk.value())
        d.morph_iter = int(self.w_mi.value())
        # Base boxes
        d.min_area_pct = float(self.w_min_area.value())
        d.max_area_pct = float(self.w_max_area.value())
        d.min_fill_ratio = float(self.w_fill.value())
        d.min_rect_px = int(self.w_min_px.value())
        d.min_rect_frac = float(self.w_min_frac.value())
        d.min_panel_px = d.min_rect_px
        # Gutters / light
        d.light_col_rel = float(self.w_lcr.value())
        d.light_row_rel = float(self.w_lrr.value())
        d.gutter_cov_min = float(self.w_cov.value())
        d.min_gutter_px = int(self.w_ming.value())
        d.min_gutter_frac = float(self.w_ming_frac.value())
        d.max_gutter_px_frac = float(self.w_maxgf.value())
        d.edge_margin_frac = float(self.w_emf.value())
        d.max_panels_per_page = int(self.w_maxp.value())
        # Projection smoothing kernel (forcer impair)
        d.proj_smooth_k = int(self.w_psk.value()) | 1
        # Title row
        d.filter_title_rows = bool(self.w_ftitle.isChecked())
        d.title_row_top_frac = float(self.w_topf.value())
        d.title_row_max_h_frac = float(self.w_maxhf.value())
        d.title_row_min_boxes = int(self.w_minb.value())
        d.title_row_min_meanL = float(self.w_minL.value())
        d.title_row_median_w_frac_max = float(self.w_title_medw.value())
        # Update requested DPI on the dialog state
        self.det_dpi = float(self.w_dpi.value())

        # Delegate to parent window to apply the tuning and force a re-run.
        parent = self.parent()
        if parent and hasattr(parent, "_apply_panel_tuning"):
            try:
                parent._apply_panel_tuning(self.det_dpi)
            except Exception:
                pdebug("_apply_panel_tuning error:\n" + traceback.format_exc())
        else:
            # Best-effort fallback: try to access common attributes if parent is missing the helper
            try:
                if parent and hasattr(parent, "_panel_cache"):
                    parent._panel_cache.clear()
                if parent and hasattr(parent, "_ensure_panels"):
                    parent._ensure_panels(force=True)
                if parent and hasattr(parent, "view"):
                    cur = parent.view.pageNavigator().currentPage()
                    rects = parent._panel_cache.get(cur, []) if hasattr(parent, "_panel_cache") else []
                    parent.view.set_panel_overlay(rects, parent._panel_mode if hasattr(parent, "_panel_mode") else False)
                    parent.view.viewport().update()
                if parent and hasattr(parent, "statusBar"):
                    parent.statusBar().showMessage("Panel tuning applied", 1500)
            except Exception:
                pdebug("Fallback apply failed:\n" + traceback.format_exc())
        # UI refresh is handled by the parent in _apply_panel_tuning()


# -----------------------------
# Main Window
# -----------------------------
class ComicsView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnComicsViewer — Lecteur PDF Comics Intelligent")
        self.resize(980, 1000)

        # Configurer l'icône de l'application
        self._setup_window_icon()

        # Core state
        self.document: Optional[QPdfDocument] = None
        self.view = PannablePdfView(self)
        self.setCentralWidget(self.view)
        self._current_path: Optional[str] = None

        # Panels state & options
        self._debug_panels = True
        self._panel_detector = PanelDetector(debug=self._debug_panels)
        self._panel_mode = False
        self._panel_framing = "fit"  # "fit" | "fill" | "center"
        self._panel_cache: dict[int, List[QRectF]] = {}
        self._panel_index = -1
        self._det_dpi = 150.0  # detection render DPI (150/200 recommended)
        
        # Auto-load trained model if available
        trained_model = "runs/detect/overfit_small/weights/best.pt"
        if os.path.exists(trained_model):
            self._ml_weights = trained_model
            pdebug(f"Auto-loaded trained model: {trained_model}")
        else:
            self._ml_weights = ""  # chemin vers .pt (à charger)

        # Drag & drop
        self.setAcceptDrops(True)

        # Compact toolbar (icons only)
        self._build_toolbar()

        # Status bar
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self._update_status()

        # Hooks
        self.view.pageNavigator().currentPageChanged.connect(self._on_page_changed)
        self.view.pageNavigator().currentPageChanged.connect(self._update_status)

    def _setup_window_icon(self):
        """Configure l'icône de la fenêtre."""
        try:
            # Essayer d'utiliser l'icône configurée par main.py
            icon_path = os.environ.get('ANCOMICSVIEWER_ICON')
            if not icon_path:
                # Fallback vers l'icône locale
                icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
            
            if os.path.exists(icon_path):
                icon = QIcon(icon_path)
                self.setWindowIcon(icon)
                # Configurer aussi l'icône de l'application
                app = QGuiApplication.instance()
                if app and hasattr(app, 'setWindowIcon'):
                    app.setWindowIcon(icon)
                pdebug(f"Window icon set: {icon_path}")
            else:
                pdebug(f"Icon not found: {icon_path}")
        except Exception as e:
            pdebug(f"Failed to set window icon: {e}")

    # ---------- UI ----------
    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        tb.setContentsMargins(0, 0, 0, 0)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        def add_btn(emoji: str, tooltip: str, slot, shortcut: Optional[QKeySequence] = None) -> QToolButton:
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

        # Open
        add_btn("📂", "Open PDF (Ctrl+O)", self.action_open, QKeySequence(QKeySequence.StandardKey.Open))
        self._add_sep(tb)

        # Navigation
        add_btn("⏮", "First page (Home)", self.nav_first, QKeySequence(QKeySequence.StandardKey.MoveToStartOfDocument))
        add_btn("⏪", "Previous page (PgUp / ←)", self.nav_prev)
        add_btn("⏩", "Next page (PgDn / →)", self.nav_next)
        add_btn("⏭", "Last page (End)", self.nav_last, QKeySequence(QKeySequence.StandardKey.MoveToEndOfDocument))
        self._add_sep(tb)

        # Zoom
        add_btn("🔍+", "Zoom in (Ctrl++)", self.zoom_in, QKeySequence(QKeySequence.StandardKey.ZoomIn))
        add_btn("🔍-", "Zoom out (Ctrl+-)", self.zoom_out, QKeySequence(QKeySequence.StandardKey.ZoomOut))
        add_btn("📏", "Fit to width (Ctrl+1)", self.fit_width, QKeySequence("Ctrl+1"))
        add_btn("🗎", "Fit to page (Ctrl+0)", self.fit_page, QKeySequence("Ctrl+0"))
        self._add_sep(tb)

        # Panels
        add_btn("▦", "Toggle panel overlay (Ctrl+2)", self.toggle_panels, QKeySequence("Ctrl+2"))
        add_btn("◀", "Previous panel (Shift+N)", self.panel_prev, QKeySequence("Shift+N"))
        add_btn("▶", "Next panel (N)", self.panel_next, QKeySequence("N"))
        self._add_sep(tb)

        # Settings / Debug menu
        settings_btn = QToolButton()
        settings_btn.setText("⚙️")
        settings_btn.setToolTip("Panel settings & debug")
        settings_btn.setAutoRaise(True)
        menu = QMenu(settings_btn)

        act_debug = menu.addAction("Debug logs")
        act_debug.setCheckable(True)
        act_debug.setChecked(self._debug_panels)
        act_debug.toggled.connect(self._on_toggle_debug)

        act_canny = menu.addAction("Use Canny fallback")
        act_canny.setCheckable(True)
        act_canny.setChecked(self._panel_detector.use_canny_fallback)
        act_canny.toggled.connect(self._on_toggle_canny)

        act_rtl = menu.addAction("Reading RTL (manga)")
        act_rtl.setCheckable(True)
        act_rtl.setChecked(self._panel_detector.reading_rtl)
        act_rtl.toggled.connect(self._on_toggle_rtl)

        menu.addSeparator()
        dpi150 = menu.addAction("Detection DPI: 150")
        dpi200 = menu.addAction("Detection DPI: 200")
        dpi150.triggered.connect(lambda: self._set_det_dpi(150.0))
        dpi200.triggered.connect(lambda: self._set_det_dpi(200.0))

        menu.addSeparator()
        rerun = menu.addAction("Re-run detection (this page)")
        rerun.triggered.connect(self._rerun_detection_current_page)
        rerun_all = menu.addAction("Re-run detection (all pages)")
        rerun_all.triggered.connect(self._rerun_detection_all)

        # Detector submenu
        det_menu = menu.addMenu("Detector")
        act_heur = det_menu.addAction("Heuristic (OpenCV)"); act_heur.setCheckable(True); act_heur.setChecked(True)
        act_ml   = det_menu.addAction("YOLOv8 Seg (ML)");    act_ml.setCheckable(True)
        act_multibd = det_menu.addAction("Multi-BD (Trained)"); act_multibd.setCheckable(True)
        act_load = det_menu.addAction("Load ML weights…")

        def _switch_heur():
            from AnComicsViewer import PanelDetector as Heur
            self._panel_detector = Heur(debug=self._debug_panels)
            act_heur.setChecked(True); act_ml.setChecked(False); act_multibd.setChecked(False)
            self._apply_panel_tuning(self._det_dpi)

        def _switch_ml():
            if not self._ml_weights:
                QMessageBox.warning(self, "ML", "Load weights (.pt) first."); return
            try:
                from detectors.yolo_seg import YoloSegPanelDetector
                self._panel_detector = YoloSegPanelDetector(weights=self._ml_weights, rtl=False)
                act_ml.setChecked(True); act_heur.setChecked(False); act_multibd.setChecked(False)
                self._apply_panel_tuning(self._det_dpi)
                QMessageBox.information(self, "ML", "Successfully switched to YOLOv8 detector!")
            except Exception as e:
                QMessageBox.critical(self, "ML Error", f"Failed to load YOLOv8 detector:\\n{str(e)}")
                # Revert to heuristic
                act_heur.setChecked(True); act_ml.setChecked(False); act_multibd.setChecked(False)

        def _switch_multibd():
            try:
                from detectors.multibd_detector import MultiBDPanelDetector
                self._panel_detector = MultiBDPanelDetector()
                act_multibd.setChecked(True); act_heur.setChecked(False); act_ml.setChecked(False)
                self._apply_panel_tuning(self._det_dpi)
                
                # Afficher les infos du modèle
                info = self._panel_detector.get_model_info()
                msg = f"✅ Modèle Multi-BD activé!\n\n"
                msg += f"📊 Performance: mAP50 {info['performance']['mAP50']}\n"
                msg += f"🎯 Entraîné sur: {', '.join(info['training_data'])}\n"
                msg += f"🔧 Seuil confiance: {info['confidence']}"
                QMessageBox.information(self, "Multi-BD Detector", msg)
            except Exception as e:
                QMessageBox.critical(self, "Multi-BD Error", f"Échec chargement détecteur Multi-BD:\\n{str(e)}")
                # Revert to heuristic
                act_heur.setChecked(True); act_ml.setChecked(False); act_multibd.setChecked(False)

        def _load_weights():
            p, _ = QFileDialog.getOpenFileName(self, "Load YOLO weights", self._default_dir(), "PT files (*.pt)")
            if p: self._ml_weights = p; QMessageBox.information(self, "ML", f"Loaded weights:\\n{p}")

        act_heur.triggered.connect(_switch_heur)
        act_ml.triggered.connect(_switch_ml)
        act_multibd.triggered.connect(_switch_multibd)
        act_load.triggered.connect(_load_weights)

        # Advanced tuning dialog
        menu.addSeparator()
        adv = menu.addAction("Advanced tuning…")
        def _open_tuning():
            dlg = PanelTuningDialog(self, self._panel_detector, self._det_dpi)
            dlg.exec()
        adv.triggered.connect(_open_tuning)

        # Presets
        presets = menu.addMenu("Presets")
        p_fb   = presets.addAction("Franco-Belge")
        p_mg   = presets.addAction("Manga")
        p_np   = presets.addAction("Newspaper")

        def _apply_preset(name):
            d = self._panel_detector
            if name == "Franco-Belge":
                self._det_dpi = 200
                d.adaptive_block, d.adaptive_C = 51, 5
                d.morph_kernel, d.morph_iter   = 7, 2
                d.min_rect_px = d.min_panel_px = 60
                d.light_col_rel, d.light_row_rel = 0.12, 0.12
                d.gutter_cov_min = 0.90
                d.min_gutter_px, d.max_gutter_px_frac = 8, 0.06
                d.edge_margin_frac = 0.03
                d.filter_title_rows = True
                d.title_row_top_frac, d.title_row_max_h_frac = 0.20, 0.12
                d.title_row_min_boxes, d.title_row_min_meanL = 4, 0.80
                d.max_panels_per_page = 20
                d.reading_rtl = False
            elif name == "Manga":
                self._det_dpi = 200
                d.adaptive_block, d.adaptive_C = 51, 4
                d.morph_kernel, d.morph_iter   = 7, 2
                d.min_rect_px = d.min_panel_px = 50
                d.light_col_rel, d.light_row_rel = 0.10, 0.10
                d.gutter_cov_min = 0.85
                d.min_gutter_px, d.max_gutter_px_frac = 6, 0.10
                d.edge_margin_frac = 0.02
                d.filter_title_rows = True
                d.title_row_top_frac, d.title_row_max_h_frac = 0.18, 0.10
                d.title_row_min_boxes, d.title_row_min_meanL = 4, 0.78
                d.max_panels_per_page = 24
                d.reading_rtl = True
            else:  # Newspaper
                self._det_dpi = 150
                d.adaptive_block, d.adaptive_C = 41, 6
                d.morph_kernel, d.morph_iter   = 5, 2
                d.min_rect_px = d.min_panel_px = 50
                d.light_col_rel, d.light_row_rel = 0.14, 0.14
                d.gutter_cov_min = 0.88
                d.min_gutter_px, d.max_gutter_px_frac = 6, 0.06
                d.edge_margin_frac = 0.03
                d.filter_title_rows = False
                d.max_panels_per_page = 16
                d.reading_rtl = False
            self._apply_panel_tuning(self._det_dpi)

        p_fb.triggered.connect(lambda: _apply_preset("Franco-Belge"))
        p_mg.triggered.connect(lambda: _apply_preset("Manga"))
        p_np.triggered.connect(lambda: _apply_preset("Newspaper"))

        settings_btn.setMenu(menu)
        settings_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        # ---- Framing submenu ----
        menu.addSeparator()
        frame_menu = menu.addMenu("Panel framing")
        act_fit = frame_menu.addAction("Fit (show context)")
        act_fill = frame_menu.addAction("Fill (hide neighbors)")
        act_center = frame_menu.addAction("Center Fit")
        for a in (act_fit, act_fill, act_center):
            a.setCheckable(True)
        act_fit.setChecked(True)

        def _set_frame(mode):
            self._panel_framing = mode
            # re-focus la case courante si on est en mode panels
            if self._panel_mode:
                cur = self.view.pageNavigator().currentPage()
                rects = self._panel_cache.get(cur, [])
                if rects and 0 <= self._panel_index < len(rects):
                    self._focus_panel(rects[self._panel_index])
            self.statusBar().showMessage(f"Panel framing: {mode}", 1500)

        act_fit.triggered.connect(lambda: (_set_frame("fit"), act_fill.setChecked(False), act_center.setChecked(False)))
        act_fill.triggered.connect(lambda: (_set_frame("fill"), act_fit.setChecked(False), act_center.setChecked(False)))
        act_center.triggered.connect(lambda: (_set_frame("center"), act_fit.setChecked(False), act_fill.setChecked(False)))

        # Raccourci pour alterner rapidement (F)
        act_cycle = QAction(self)
        act_cycle.setShortcut("F")
        def _cycle_frame():
            order = ["fit", "fill", "center"]
            i = (order.index(self._panel_framing) + 1) % len(order)
            m = order[i]
            act_fit.setChecked(m == "fit")
            act_fill.setChecked(m == "fill")
            act_center.setChecked(m == "center")
            _set_frame(m)
        act_cycle.triggered.connect(_cycle_frame)
        self.addAction(act_cycle)

        tb.addWidget(settings_btn)

    def _add_sep(self, tb: QToolBar):
        sep = QWidget()
        sep.setFixedWidth(6)
        tb.addWidget(sep)

    # ---------- Settings handlers ----------
    def _on_toggle_debug(self, checked: bool):
        self._debug_panels = checked
        self._panel_detector.debug = checked
        self.statusBar().showMessage(f"Debug logs {'ON' if checked else 'OFF'}", 1500)

    def _on_toggle_canny(self, checked: bool):
        self._panel_detector.use_canny_fallback = checked
        self.statusBar().showMessage(f"Canny fallback {'ON' if checked else 'OFF'}", 1500)

    def _on_toggle_rtl(self, checked: bool):
        self._panel_detector.reading_rtl = checked
        self.statusBar().showMessage(f"Reading {'RTL' if checked else 'LTR'}", 1500)

    def _set_det_dpi(self, dpi: float):
        self._det_dpi = dpi
        self.statusBar().showMessage(f"Detection DPI set to {int(dpi)}", 1500)

    def _rerun_detection_current_page(self):
        self._panel_cache.pop(self.view.pageNavigator().currentPage(), None)
        self._ensure_panels(force=True)
        self.statusBar().showMessage("Detection re-run", 1500)

    def _apply_panel_tuning(self, new_dpi: float):
        """Apply new tuning DPI and force a full re-run/refresh.

        Clears the entire panel cache and re-detects panels for the current page so
        UI reflects the updated parameters immediately.
        """
        try:
            self._det_dpi = float(new_dpi)
            self._panel_index = -1
            self._panel_cache.clear()
            self._ensure_panels(force=True)
            cur = self.view.pageNavigator().currentPage()
            rects = self._panel_cache.get(cur, [])
            self.view.set_panel_overlay(rects, self._panel_mode)
            self.view.viewport().update()
            self.statusBar().showMessage("Panel tuning applied", 1500)
        except Exception:
            pdebug("_apply_panel_tuning error:\n" + traceback.format_exc())

    def _rerun_detection_all(self):
        """Clear the detector cache for all pages and re-run detection for the current page."""
        try:
            self._panel_cache.clear()
            self._panel_index = -1
            self._ensure_panels(force=True)
            cur = self.view.pageNavigator().currentPage()
            rects = self._panel_cache.get(cur, [])
            self.view.set_panel_overlay(rects, self._panel_mode)
            self.view.viewport().update()
            self.statusBar().showMessage("Re-run detection completed", 1500)
        except Exception:
            pdebug("_rerun_detection_all error:\n" + traceback.format_exc())


    # ---------- File ops ----------
    def action_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", self._default_dir(), "PDF Files (*.pdf)")
        if path:
            self.load_pdf(path)

    def _default_dir(self) -> str:
        if self._current_path and os.path.exists(self._current_path):
            return os.path.dirname(self._current_path)
        return os.path.expanduser("~")

    def load_pdf(self, path: str) -> bool:
        if not path or not os.path.exists(path) or not path.lower().endswith(".pdf"):
            QMessageBox.warning(self, "Invalid File", "Please select a valid PDF file.")
            return False

        # Release previous doc fully (Windows-safe)
        if self.document is not None:
            try:
                self.view.setDocument(QPdfDocument())  # clear ref from view
                self.document.close()
                self.document.deleteLater()
            except Exception:
                pass
            self.document = None

        doc = QPdfDocument(self)
        err = doc.load(path)
        success = False
        try:
            if hasattr(QPdfDocument.Error, "None_") and err == QPdfDocument.Error.None_:
                success = True
        except Exception:
            pass
        try:
            if hasattr(QPdfDocument.Error, "NoError") and err == QPdfDocument.Error.NoError:
                success = True
        except Exception:
            pass
        if err == 0 or getattr(err, "value", 0) == 0:
            success = True

        if not success or doc.pageCount() <= 0:
            QMessageBox.critical(self, "Load Error", "Failed to load PDF (corrupted/unsupported).")
            doc.deleteLater()
            return False

        self.document = doc
        self._current_path = path
        self.view.setDocument(self.document)
        self.setWindowTitle(f"ComicsView — {os.path.basename(path)}")

        self.fit_page()  # sensible default for comics
        self._panel_cache.clear()
        self._panel_index = -1
        self._update_status()
        return True

    # ---------- Export ----------
    def export_current_page(self):
        if not self.document:
            QMessageBox.warning(self, "Export Error", "No PDF loaded.")
            return
        page = self.view.pageNavigator().currentPage()
        base = os.path.splitext(os.path.basename(self._current_path or "page"))[0]
        filename = f"{base}_page_{page + 1:03d}.png"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export Page as PNG", os.path.join(self._default_dir(), filename), "PNG Files (*.png)"
        )
        if not out_path:
            return
        # Render at ~200 DPI
        pt = self.document.pagePointSize(page)
        scale = 200.0 / 72.0
        qsize = QSizeF(pt.width() * scale, pt.height() * scale).toSize()
        img = self.document.render(page, qsize)
        if img.isNull() or not img.save(out_path):
            QMessageBox.critical(self, "Export Error", "Failed to save image.")
            return
        self.statusBar().showMessage(f"Exported {os.path.basename(out_path)}", 2500)

    # ---------- Navigation ----------
    def nav_first(self):
        if self.document:
            self.view.pageNavigator().jump(0, QPointF(0, 0))

    def nav_last(self):
        if self.document:
            self.view.pageNavigator().jump(self.document.pageCount() - 1, QPointF(0, 0))

    def nav_prev(self):
        if self.document:
            cur = self.view.pageNavigator().currentPage()
            if cur > 0:
                self.view.pageNavigator().jump(cur - 1, QPointF(0, 0))

    def nav_next(self):
        if self.document:
            cur = self.view.pageNavigator().currentPage()
            if cur < self.document.pageCount() - 1:
                self.view.pageNavigator().jump(cur + 1, QPointF(0, 0))

    # ---------- Zoom ----------
    def zoom_in(self):
        if not self.document:
            return
        z = self.view.zoomFactor()
        self.view.setZoomMode(QPdfView.ZoomMode.Custom)
        self.view.setZoomFactor(min(16.0, z * 1.25))
        self._update_status()

    def zoom_out(self):
        if not self.document:
            return
        z = self.view.zoomFactor()
        self.view.setZoomMode(QPdfView.ZoomMode.Custom)
        self.view.setZoomFactor(max(0.05, z * 0.8))
        self._update_status()

    def fit_width(self):
        if self.document:
            self.view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
            self._update_status()

    def fit_page(self):
        if self.document:
            self.view.setZoomMode(QPdfView.ZoomMode.FitInView)
            self._update_status()

    # ---------- Panels ----------
    def toggle_panels(self):
        if not self.document:
            return
        self._panel_mode = not self._panel_mode
        self.view.setPageMode(QPdfView.PageMode.SinglePage if self._panel_mode else QPdfView.PageMode.MultiPage)
        self._panel_index = -1
        self._ensure_panels(force=True)
        cur = self.view.pageNavigator().currentPage()
        rects = self._panel_cache.get(cur, [])
        pdebug(f"toggle: page={cur} rects={len(rects)}")
        self.view.set_panel_overlay(rects, self._panel_mode)
        self.statusBar().showMessage("Panel mode ON" if self._panel_mode else "Panel mode OFF", 2000)

    def panel_next(self):
        """Navigation vers la case suivante avec saut de page automatique."""
        if not (self.document and self._panel_mode):
            return
        try:
            cur = self.view.pageNavigator().currentPage()
            self._ensure_panels_for(cur)
            rects = self._panel_cache.get(cur, [])
            
            # Cas spécial : gestion de l'état initial _panel_index == -1
            if self._panel_index == -1:
                # Chercher la première page avec des panels à partir de la page courante
                for page in range(cur, self.document.pageCount()):
                    self._ensure_panels_for(page)
                    page_rects = self._panel_cache.get(page, [])
                    if page_rects:
                        if page != cur:
                            self._goto_page_and_overlay(page)
                            self.statusBar().showMessage(f"Page {page + 1}: panel 1/{len(page_rects)}", 3000)
                        self._panel_index = 0
                        self._focus_panel(page_rects[0])
                        pdebug(f"panel_next (init) -> page {page}, panel 1/{len(page_rects)}")
                        return
                # Aucune page avec panels trouvée
                self.statusBar().showMessage("No panels found in document", 2000)
                return
            
            # Si pas de panels sur la page courante, chercher la page suivante
            if not rects:
                for page in range(cur + 1, self.document.pageCount()):
                    self._ensure_panels_for(page)
                    page_rects = self._panel_cache.get(page, [])
                    if page_rects:
                        self._goto_page_and_overlay(page)
                        self._panel_index = 0
                        self._focus_panel(page_rects[0])
                        self.statusBar().showMessage(f"Page {page + 1}: panel 1/{len(page_rects)}", 3000)
                        pdebug(f"panel_next (no panels) -> page {page}, panel 1/{len(page_rects)}")
                        return
                # Aucune page suivante avec panels
                self.statusBar().showMessage("No more panels in document", 2000)
                return
            
            # Si on est sur la dernière case de la page, passer à la page suivante
            if self._panel_index >= len(rects) - 1:
                for page in range(cur + 1, self.document.pageCount()):
                    self._ensure_panels_for(page)
                    page_rects = self._panel_cache.get(page, [])
                    if page_rects:
                        self._goto_page_and_overlay(page)
                        self._panel_index = 0
                        self._focus_panel(page_rects[0])
                        self.statusBar().showMessage(f"Page {page + 1}: panel 1/{len(page_rects)}", 3000)
                        pdebug(f"panel_next (page jump) -> page {page}, panel 1/{len(page_rects)}")
                        return
                # Pas de page suivante avec panels
                self.statusBar().showMessage("Last panel reached", 2000)
                return
            
            # Cas normal : passer à la case suivante sur la même page
            self._panel_index += 1
            self._focus_panel(rects[self._panel_index])
            pdebug(f"panel_next -> {self._panel_index + 1}/{len(rects)}")
            
        except Exception:
            pdebug("panel_next error:\n" + traceback.format_exc())

    def panel_prev(self):
        """Navigation vers la case précédente avec saut de page automatique."""
        if not (self.document and self._panel_mode):
            return
        try:
            cur = self.view.pageNavigator().currentPage()
            self._ensure_panels_for(cur)
            rects = self._panel_cache.get(cur, [])
            
            # Cas spécial : gestion de l'état initial _panel_index == -1
            if self._panel_index == -1:
                # Chercher la dernière page avec des panels en partant de la page courante vers l'arrière
                for page in range(cur, -1, -1):
                    self._ensure_panels_for(page)
                    page_rects = self._panel_cache.get(page, [])
                    if page_rects:
                        if page != cur:
                            self._goto_page_and_overlay(page)
                            self.statusBar().showMessage(f"Page {page + 1}: panel {len(page_rects)}/{len(page_rects)}", 3000)
                        self._panel_index = len(page_rects) - 1
                        self._focus_panel(page_rects[self._panel_index])
                        pdebug(f"panel_prev (init) -> page {page}, panel {len(page_rects)}/{len(page_rects)}")
                        return
                # Aucune page avec panels trouvée
                self.statusBar().showMessage("No panels found in document", 2000)
                return
            
            # Si pas de panels sur la page courante, chercher la page précédente
            if not rects:
                for page in range(cur - 1, -1, -1):
                    self._ensure_panels_for(page)
                    page_rects = self._panel_cache.get(page, [])
                    if page_rects:
                        self._goto_page_and_overlay(page)
                        self._panel_index = len(page_rects) - 1
                        self._focus_panel(page_rects[self._panel_index])
                        self.statusBar().showMessage(f"Page {page + 1}: panel {len(page_rects)}/{len(page_rects)}", 3000)
                        pdebug(f"panel_prev (no panels) -> page {page}, panel {len(page_rects)}/{len(page_rects)}")
                        return
                # Aucune page précédente avec panels
                self.statusBar().showMessage("No previous panels in document", 2000)
                return
            
            # Si on est sur la première case de la page, passer à la page précédente
            if self._panel_index <= 0:
                for page in range(cur - 1, -1, -1):
                    self._ensure_panels_for(page)
                    page_rects = self._panel_cache.get(page, [])
                    if page_rects:
                        self._goto_page_and_overlay(page)
                        self._panel_index = len(page_rects) - 1
                        self._focus_panel(page_rects[self._panel_index])
                        self.statusBar().showMessage(f"Page {page + 1}: panel {len(page_rects)}/{len(page_rects)}", 3000)
                        pdebug(f"panel_prev (page jump) -> page {page}, panel {len(page_rects)}/{len(page_rects)}")
                        return
                # Pas de page précédente avec panels
                self.statusBar().showMessage("First panel reached", 2000)
                return
            
            # Cas normal : passer à la case précédente sur la même page
            self._panel_index -= 1
            self._focus_panel(rects[self._panel_index])
            pdebug(f"panel_prev -> {self._panel_index + 1}/{len(rects)}")
            
        except Exception:
            pdebug("panel_prev error:\n" + traceback.format_exc())

    def _goto_page_and_overlay(self, page: int):
        """Helper de changement de page avec overlay des panels."""
        if not self.document:
            return
        # Naviguer vers la page
        self.view.pageNavigator().jump(page, QPointF(0, 0))
        # Assurer la détection des panels
        self._ensure_panels_for(page)
        # Récupérer et afficher l'overlay
        rects = self._panel_cache.get(page, [])
        self.view.set_panel_overlay(rects, self._panel_mode)

    def _ensure_panels_for(self, page: int, force: bool = False):
        """Helper pour détecter les cases d'une page donnée."""
        if not self.document:
            pdebug("ensure_panels_for: no document")
            return
        if (not force) and page in self._panel_cache:
            return
        try:
            pt = self.document.pagePointSize(page)
            dpi = self._det_dpi
            scale = dpi / 72.0
            qsize = QSizeF(pt.width() * scale, pt.height() * scale).toSize()
            qimg = self.document.render(page, qsize)
            rects = self._panel_detector.detect_panels(qimg, pt)
            self._panel_cache[page] = rects
            pdebug(f"ensure_panels_for: page={page}, panels={len(rects)} @ {int(dpi)} DPI")
        except Exception:
            pdebug("ensure_panels_for error:\n" + traceback.format_exc())
            self._panel_cache[page] = []

    def _ensure_panels(self, force: bool=False):
        """Assure la détection des cases pour la page courante."""
        if not self.document:
            pdebug("ensure_panels: no document")
            return
        cur = self.view.pageNavigator().currentPage()
        self._ensure_panels_for(cur, force)

    def _focus_panel(self, rect: QRectF):
        """Zoom selon le framing puis scroll correctement."""
        try:
            if not self.document or not rect or rect.isEmpty():
                return
            vw = self.view.viewport().width()
            vh = self.view.viewport().height()

            # ZOOM :
            if self._panel_framing == "fill":
                # le viewport doit tenir À L’INTÉRIEUR de la case
                z = max(vw / max(1e-6, rect.width()), vh / max(1e-6, rect.height()))
                z = min(16.0, max(0.05, z * 1.01))  # léger overscan pour éviter les bordures
            else:
                # fit (show context) et center-fit : la case tient ENTIEREMENT dans le viewport
                z = min(vw / max(1e-6, rect.width()), vh / max(1e-6, rect.height()))
                z = min(16.0, max(0.05, z * 0.96))  # petite marge de confort

            self.view.setZoomMode(QPdfView.ZoomMode.Custom)
            self.view.setZoomFactor(z)
            # Scroller après que le zoom soit appliqué :
            QTimer.singleShot(0, lambda r=QRectF(rect), m=self._panel_framing: self._scroll_to_panel(r, m))
            pdebug(f"focus idx={self._panel_index} -> zoom={z:.2f} ({self._panel_framing})")
        except Exception:
            pdebug("focus_panel error:\n" + traceback.format_exc())

    def _scroll_to_panel(self, rect: QRectF, mode: str, margin_px: int = 12):
        """Positionne les scrollbars ; en 'fill' le viewport reste 100% dans la case."""
        try:
            doc = self.document
            if not doc:
                return
            cur = self.view.pageNavigator().currentPage()
            page_pts = doc.pagePointSize(cur)
            z  = self.view.zoomFactor()
            vw = self.view.viewport().width()
            vh = self.view.viewport().height()

            content_w = page_pts.width() * z
            content_h = page_pts.height() * z
            pad_x = max(0.0, (vw - content_w) / 2.0)
            pad_y = max(0.0, (vh - content_h) / 2.0)

            hbar = self.view.horizontalScrollBar()
            vbar = self.view.verticalScrollBar()
            hxmax = hbar.maximum() if hbar else 0
            hymax = vbar.maximum() if vbar else 0

            if mode == "fill":
                # Taille visible en "points de page"
                vis_w_pt = vw / z
                vis_h_pt = vh / z
                # Clamp : garder tout le viewport À L’INTÉRIEUR de la case
                x_pt = min(max(rect.left(),  rect.right()  - vis_w_pt), rect.left()  + max(0.0, rect.width()  - vis_w_pt))
                y_pt = min(max(rect.top(),   rect.bottom() - vis_h_pt), rect.top()   + max(0.0, rect.height() - vis_h_pt))
            elif mode == "center":
                # centrer strictement la case (fit)
                x_pt = rect.center().x() - (vw / z) / 2.0
                y_pt = rect.center().y() - (vh / z) / 2.0
            else:
                # fit (contexte visible) : viser coin haut-gauche avec une petite marge
                x_pt = rect.left() - (margin_px / z)
                y_pt = rect.top()  - (margin_px / z)

            # Transforme points -> pixels viewport
            target_x = int(max(0.0, pad_x + x_pt * z))
            target_y = int(max(0.0, pad_y + y_pt * z))

            if hbar: hbar.setValue(min(max(0, target_x), hxmax))
            if vbar: vbar.setValue(min(max(0, target_y), hymax))
            self.view.viewport().update()
            pdebug(f"scroll_to_panel[{mode}]: target=({target_x},{target_y}) max=({hxmax},{hymax}) z={z:.2f}")
        except Exception:
            pdebug("scroll_to_panel error:\n" + traceback.format_exc())

    def _update_status(self):
        """Update the status bar with current page and zoom info."""
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

    def _on_page_changed(self):
        """Handle page change events."""
        try:
            if self._panel_mode:
                self._panel_index = -1  # Reset panel navigation
                self._ensure_panels()
                cur = self.view.pageNavigator().currentPage()
                rects = self._panel_cache.get(cur, [])
                self.view.set_panel_overlay(rects, True)
        except Exception:
            pdebug("page_changed error:\n" + traceback.format_exc())


# -----------------------------
# Main entry point
# -----------------------------
def main():
    app = QApplication(sys.argv)
    window = ComicsView()
    window.show()
    return app.exec()


if __name__ == "__main__":
    main()
