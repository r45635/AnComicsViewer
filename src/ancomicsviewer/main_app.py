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

# Import AR integration if available
try:
    from .ar_integration import ARIntegrationMixin
    AR_AVAILABLE = True
except ImportError:
    class ARIntegrationMixin:
        pass
    AR_AVAILABLE = False

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

from PySide6.QtCore import Qt, QPoint, QPointF, QSize, QSizeF, QMimeData, QRectF, QTimer, QSettings

# Import du cache amélioré
try:
    from .utils.enhanced_cache import PanelCacheManager
    ENHANCED_CACHE_AVAILABLE = True
except ImportError:
    ENHANCED_CACHE_AVAILABLE = False
    PanelCacheManager = None

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
        
        # --- reading order parameters ---
        self.row_band_frac = 0.06  # 6% of page height for row grouping tolerance

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
                f"row_band={self.row_band_frac:.3f}",
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

            # Ordre de lecture avec groupement par rangées
            rects = self._sort_reading_order(rects, page_point_size)
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

    def _sort_reading_order(self, rects: List[QRectF], page_point_size: QSizeF) -> List[QRectF]:
        """Sort panels by reading order: group into rows, then sort by horizontal position."""
        if not rects: 
            return rects
        H = float(page_point_size.height()) if page_point_size.height() > 0 else 1.0
        band = max(8.0, H * float(getattr(self, "row_band_frac", 0.06)))
        rows = []  # [{ "y": float, "items": List[QRectF] }]
        for r in rects:
            y = r.top()
            placed = False
            for row in rows:
                if abs(y - row["y"]) <= band:
                    row["items"].append(r)
                    row["y"] = min(row["y"], y)  # ancre sur le plus haut
                    placed = True
                    break
            if not placed:
                rows.append({"y": y, "items": [r]})
        rows.sort(key=lambda rr: rr["y"])
        ordered: List[QRectF] = []
        if self.reading_rtl:
            for row in rows:
                row["items"].sort(key=lambda rr: (-rr.left(), rr.top()))
                ordered.extend(row["items"])
        else:
            for row in rows:
                row["items"].sort(key=lambda rr: (rr.left(), rr.top()))
                ordered.extend(row["items"])
        return ordered


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
        self._overlay_rects: List[QRectF] = []  # panels in page points for CURRENT page only
        self._overlay_balloons: List[QRectF] = []  # balloons in page points for CURRENT page only

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
    def set_panel_overlay(self, panels: List[QRectF], balloons: List[QRectF], enabled: bool):
        self._overlay_rects = panels or []
        self._overlay_balloons = balloons or []
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
        
        # 🎯 APPROCHE CORRIGÉE : Utiliser le padding comme QPdfView le fait réellement
        # Mais seulement quand la page est plus petite que le viewport
        pad_x = max(0.0, (vw - content_w) / 2.0) if content_w < vw else 0.0
        pad_y = max(0.0, (vh - content_h) / 2.0) if content_h < vh else 0.0
        
        sx = self.horizontalScrollBar().value()
        sy = self.verticalScrollBar().value()
        
        # 🔧 CALCUL HYBRIDE : padding + zoom - scroll
        x = pad_x + (x_pt * z) - sx
        y = pad_y + (y_pt * z) - sy
        
        # 🔍 DEBUG TRANSFORMATION CIBLE pour BALLOONS uniquement page 3
        cur = self.pageNavigator().currentPage()
        if cur == 2:  # Page 3 (index 2)
            # Diagnostic uniquement pour les balloons
            if (abs(x_pt - 268.1) < 0.1 and abs(y_pt - 0.3) < 0.1) or \
               (abs(x_pt - 287.0) < 0.1 and abs(y_pt - 69.7) < 0.1):
                object_type = "BALLOON B1" if abs(x_pt - 268.1) < 0.1 else "BALLOON B2"
                print(f"🎯 {object_type} TRANSFORM DIRECT:")
                print(f"   PDF({x_pt:.1f},{y_pt:.1f}) → Z{z:.3f} → Pad({pad_x:.1f},{pad_y:.1f}) → Scr({sx},{sy}) → VIEW({x:.1f},{y:.1f})")
                
                # 🔧 VÉRIFICATION: recalculer avec le zoom EXACT du paintEvent
                import time
                current_time = time.time()
                if not hasattr(self, '_last_paint_zoom_check') or (current_time - self._last_paint_zoom_check) > 0.1:
                    self._last_paint_zoom_check = current_time
                    paint_zoom = self.zoomFactor()
                    if abs(paint_zoom - z) > 0.001:
                        corrected_x = pad_x + (x_pt * paint_zoom) - sx
                        corrected_y = pad_y + (y_pt * paint_zoom) - sy
                        print(f"   🚨 CORRECTION ZOOM: Paint={paint_zoom:.3f} vs Calc={z:.3f}")
                        print(f"   🎯 COORDONNÉES CORRIGÉES: ({corrected_x:.1f},{corrected_y:.1f})")
        
        return (x, y)

    def _page_rect_to_view(self, r: QRectF) -> QRectF:
        """Convertir un rectangle PDF vers VIEW en utilisant le zoom EXACT du moment"""
        # 🎯 UTILISER LE ZOOM EXACT du moment de l'affichage
        current_zoom = self.zoomFactor()
        
        # Utiliser la fonction originale pour les coordonnées avec le zoom exact
        x, y = self._page_to_view_xy(r.left(), r.top())
        
        # 🔧 Calculer les dimensions avec le zoom EXACT du moment
        w = r.width() * current_zoom
        h = r.height() * current_zoom
        
        return QRectF(x, y, w, h)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._overlay_enabled or (not self._overlay_rects and not self._overlay_balloons):
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

            # 🔄 VÉRIFIER changement de zoom pour invalider cache VIEW
            current_zoom = self.zoomFactor()
            main = self.window()
            if (hasattr(main, '_last_cached_zoom') and 
                abs(current_zoom - main._last_cached_zoom) > 0.001):
                print(f"🔄 ZOOM CHANGÉ EN PAINT: {main._last_cached_zoom:.3f} → {current_zoom:.3f}")
                print("🗑️ Invalidation cache VIEW (zoom en temps réel)")
                if hasattr(main, '_panel_view_cache'):
                    main._panel_view_cache.clear()
                if hasattr(main, '_balloon_view_cache'):
                    main._balloon_view_cache.clear()

            # --- Cadre de page (debug) ---
            page_rect_view = self._page_rect_to_view(QRectF(0, 0, page_pts.width(), page_pts.height()))
            p2 = QPen(QColor(0, 120, 255, 200), 2)  # bleu pour le cadre de page
            painter.setPen(p2); painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(page_rect_view)

            # --- Panels ---
            pen = QPen(QColor(0, 200, 0, 220), 2)
            fill = QColor(0, 200, 0, 55)
            painter.setPen(pen); painter.setBrush(fill)
            
            # 🔍 LOGS DÉTAILLÉS SPÉCIFIQUES PAGE 3
            is_page_3 = (cur == 2)  # page 3 = index 2
            if is_page_3:
                print(f"\n🎯 === DIAGNOSTIC PAGE 3 === PAINT EVENT ===")
                print(f"   📄 Page courante: {cur} (Page 3)")
                print(f"   📏 Page size: {page_pts.width():.1f}x{page_pts.height():.1f} pts")
                print(f"   🔍 Zoom actuel: {current_zoom:.3f}")
                print(f"   📐 Viewport size: {self.viewport().size().width()}x{self.viewport().size().height()}")
                
                # Informations de positionnement du viewport
                scroll_x = self.horizontalScrollBar().value()
                scroll_y = self.verticalScrollBar().value()
                print(f"   📜 Scroll position: ({scroll_x}, {scroll_y})")
                
                # État du cache zoom
                if hasattr(main, '_last_cached_zoom'):
                    print(f"   🔄 Cache zoom: {main._last_cached_zoom:.3f} vs actuel: {current_zoom:.3f}")
                    print(f"   🔄 Différence zoom: {abs(current_zoom - main._last_cached_zoom):.6f}")
            else:
                print(f"🎨 PAINT EVENT - Page {cur}")
                print(f"   📏 Page size: {page_pts.width():.1f}x{page_pts.height():.1f} pts")
                print(f"   🔍 Zoom: {current_zoom:.3f}")
            
            # 🎯 UTILISER les coordonnées VIEW pré-calculées si disponibles
            cur_page = self.pageNavigator().currentPage()
            
            # Vérifier si on a des coordonnées VIEW en cache
            # 🎯 DÉSACTIVER TEMPORAIREMENT le cache VIEW pour diagnostiquer le problème
            # if (hasattr(main, '_panel_view_cache') and cur_page in main._panel_view_cache and 
            #     hasattr(main, '_balloon_view_cache') and cur_page in main._balloon_view_cache):
            if False:  # 🔧 FORCER le recalcul en temps réel
                # UTILISER les coordonnées VIEW pré-calculées (plus précises)
                view_panels = main._panel_view_cache[cur_page]
                view_balloons = main._balloon_view_cache[cur_page]
                
                print(f"   🟢 Panels à dessiner: {len(view_panels)} (coordonnées VIEW pré-calculées)")
                for idx, vr in enumerate(view_panels):
                    if vr.isEmpty(): 
                        continue
                    print(f"      P{idx+1}: VIEW({vr.x():.1f},{vr.y():.1f},{vr.width():.1f}x{vr.height():.1f}) [PRE-CALC]")
                    painter.drawRect(vr)
                    # index en surimpression
                    painter.setPen(QPen(QColor(0,0,0,255), 1))
                    painter.drawText(vr.adjusted(3, 3, -3, -3), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, f"P{idx+1}")
                    painter.setPen(pen)
                
                # --- Balloons (ROUGE) ---
                pen_balloon = QPen(QColor(200, 0, 0, 220), 2)
                fill_balloon = QColor(200, 0, 0, 55)
                painter.setPen(pen_balloon); painter.setBrush(fill_balloon)
                print(f"   🔴 Balloons à dessiner: {len(view_balloons)} (coordonnées VIEW pré-calculées)")
                for idx, vr in enumerate(view_balloons):
                    if vr.isEmpty(): 
                        continue
                    print(f"      B{idx+1}: VIEW({vr.x():.1f},{vr.y():.1f},{vr.width():.1f}x{vr.height():.1f}) [PRE-CALC]")
                    painter.drawRect(vr)
                    # index en surimpression
                    painter.setPen(QPen(QColor(255,255,255,255), 1))
                    painter.drawText(vr.adjusted(3, 3, -3, -3), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, f"B{idx+1}")
                    painter.setPen(pen_balloon)
            else:
                # FALLBACK: utiliser les vraies coordonnées PDF depuis le cache du main
                main = self.window()
                cur_page = self.pageNavigator().currentPage()
                
                # Récupérer les vraies coordonnées PDF depuis le cache principal
                if (hasattr(main, '_panel_cache') and cur_page in main._panel_cache):
                    pdf_panels = main._panel_cache[cur_page]
                    print(f"   🟢 Panels à dessiner: {len(pdf_panels)} (conversion PDF→VIEW temps réel)")
                    for idx, r in enumerate(pdf_panels):
                        if r.isEmpty(): 
                            continue
                        vr = self._page_rect_to_view(r)
                        print(f"      P{idx+1}: PDF({r.x():.1f},{r.y():.1f},{r.width():.1f}x{r.height():.1f}) → View({vr.x():.1f},{vr.y():.1f},{vr.width():.1f}x{vr.height():.1f}) [REALTIME]")
                        painter.drawRect(vr)
                        # index en surimpression
                        painter.setPen(QPen(QColor(0,0,0,255), 1))
                        painter.drawText(vr.adjusted(3, 3, -3, -3), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, f"P{idx+1}")
                        painter.setPen(pen)
                else:
                    print("   🟢 Panels à dessiner: 0 (pas de cache PDF disponible)")
                
                # --- Balloons (ROUGE) ---
                pen_balloon = QPen(QColor(200, 0, 0, 220), 2)
                fill_balloon = QColor(200, 0, 0, 55)
                painter.setPen(pen_balloon); painter.setBrush(fill_balloon)
                
                if (hasattr(main, '_balloon_cache') and cur_page in main._balloon_cache):
                    pdf_balloons = main._balloon_cache[cur_page]
                    
                    # 🔍 LOGS DÉTAILLÉS SPÉCIFIQUES PAGE 3 pour BALLOONS
                    if is_page_3:
                        print(f"\n🔴 === BALLOONS PAGE 3 - DIAGNOSTIC CONVERSION ===")
                        print(f"   📊 Nombre de balloons trouvés: {len(pdf_balloons)}")
                        print(f"   🔍 Zoom exact au moment du paint: {current_zoom:.6f}")
                        
                        # État du viewport au moment du calcul
                        vw = self.viewport().width()
                        vh = self.viewport().height()
                        sx = self.horizontalScrollBar().value()
                        sy = self.verticalScrollBar().value()
                        print(f"   📐 Viewport: {vw}x{vh}, Scroll: ({sx},{sy})")
                        
                        # Test de la page pour le padding
                        content_w = page_pts.width() * current_zoom
                        content_h = page_pts.height() * current_zoom
                        pad_x = max(0.0, (vw - content_w) / 2.0) if content_w < vw else 0.0
                        pad_y = max(0.0, (vh - content_h) / 2.0) if content_h < vh else 0.0
                        print(f"   � Content: {content_w:.1f}x{content_h:.1f}, Padding: ({pad_x:.1f},{pad_y:.1f})")
                    else:
                        print(f"   �🔴 Balloons à dessiner: {len(pdf_balloons)} (conversion PDF→VIEW temps réel)")
                        
                    for idx, r in enumerate(pdf_balloons):
                        if r.isEmpty(): 
                            continue
                        vr = self._page_rect_to_view(r)
                        
                        # 🔍 LOGS DÉTAILLÉS SPÉCIFIQUES PAGE 3
                        if is_page_3:
                            print(f"   B{idx+1}: PDF({r.x():.1f},{r.y():.1f},{r.width():.1f}x{r.height():.1f}) → VIEW({vr.x():.1f},{vr.y():.1f},{vr.width():.1f}x{vr.height():.1f}) [REALTIME-P3]")
                            
                            # 🎯 Analyse spéciale pour B1 et B2
                            if idx == 0:  # B1
                                print(f"      🎯 B1 ANALYSE: Center PDF({r.center().x():.1f},{r.center().y():.1f}) → VIEW({vr.center().x():.1f},{vr.center().y():.1f})")
                                print(f"      🎯 B1 VISIBLE: Dans viewport? x∈[0,{vw}]={0 <= vr.x() <= vw}, y∈[0,{vh}]={0 <= vr.y() <= vh}")
                            elif idx == 1:  # B2  
                                print(f"      🎯 B2 ANALYSE: Center PDF({r.center().x():.1f},{r.center().y():.1f}) → VIEW({vr.center().x():.1f},{vr.center().y():.1f})")
                                print(f"      🎯 B2 VISIBLE: Dans viewport? x∈[0,{vw}]={0 <= vr.x() <= vw}, y∈[0,{vh}]={0 <= vr.y() <= vh}")
                        else:
                            print(f"      B{idx+1}: PDF({r.x():.1f},{r.y():.1f},{r.width():.1f}x{r.height():.1f}) → View({vr.x():.1f},{vr.y():.1f},{vr.width():.1f}x{vr.height():.1f}) [REALTIME]")
                            
                        painter.drawRect(vr)
                        # index en surimpression
                        painter.setPen(QPen(QColor(255,255,255,255), 1))
                        painter.drawText(vr.adjusted(3, 3, -3, -3), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, f"B{idx+1}")
                        painter.setPen(pen_balloon)
                else:
                    print("   🔴 Balloons à dessiner: 0 (pas de cache PDF disponible)")

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
        self.w_rowband = dsb(getattr(det, "row_band_frac", 0.06), 0.01, 0.20, 0.005)
        form.addRow("Row band frac", self.w_rowband)
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
        d.row_band_frac = float(self.w_rowband.value())
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
                if parent and hasattr(parent, "_balloon_cache"):
                    parent._balloon_cache.clear()
                if parent and hasattr(parent, "_ensure_panels"):
                    parent._ensure_panels(force=True)
                if parent and hasattr(parent, "view"):
                    cur = parent.view.pageNavigator().currentPage()
                    rects = parent._panel_cache.get(cur, []) if hasattr(parent, "_panel_cache") else []
                    balloons = parent._balloon_cache.get(cur, []) if hasattr(parent, "_balloon_cache") else []
                    parent.view.set_panel_overlay(rects, balloons, parent._panel_mode if hasattr(parent, "_panel_mode") else False)
                    parent.view.viewport().update()
                if parent and hasattr(parent, "statusBar"):
                    parent.statusBar().showMessage("Panel tuning applied", 1500)
            except Exception:
                pdebug("Fallback apply failed:\n" + traceback.format_exc())
        # UI refresh is handled by the parent in _apply_panel_tuning()


# -----------------------------
# Main Window
# -----------------------------
class ComicsView(QMainWindow, ARIntegrationMixin):
    def __init__(self):
        super().__init__()
        
        # Initialiser AR Integration si disponible
        ar_available = globals().get('AR_AVAILABLE', False)
        if ar_available:
            try:
                # Initialisation manuelle des attributs AR
                self.ar_page_view = None
                self.ar_detector = None
                self.ar_mode_enabled = False
                self.ar_current_qimage = None
                self.ar_pdf_document = None
                self.ar_current_page = 0
                self.ar_pdf_path = ""
                pdebug("🔬 AR Integration initialisé")
            except Exception as e:
                pdebug(f"⚠️ Erreur init AR: {e}")
        else:
            # Fallback si AR non disponible
            self.ar_mode_enabled = False
        
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
        
        # Vérifier si le mode AR est activé
        self._ar_mode_enabled = os.environ.get("ANCOMICS_AR_MODE") == "1"
        if self._ar_mode_enabled and AR_AVAILABLE:
            pdebug("🔬 Mode AR ACTIVÉ - Overlays parfaitement alignés!")
        
        # Utiliser le détecteur YOLO ROBUSTE AR-COMPLIANT
        try:
            from .detectors.robust_yolo_detector import RobustYoloDetector
            self._panel_detector = RobustYoloDetector()
            print("🔥 DÉTECTEUR YOLO ROBUSTE AR-COMPLIANT ACTIVÉ !")
        except Exception as e:
            print(f"❌ ERREUR détecteur yolo robuste: {e}")
            # Fallback vers ultra-robuste
            try:
                from .detectors.ultra_robust_detector import UltraRobustDetector
                self._panel_detector = UltraRobustDetector()
                print("🔄 Fallback vers UltraRobustDetector")
            except Exception as e2:
                print(f"❌ ERREUR fallback ultra: {e2}")
                # Fallback final vers YOLO28H
                try:
                    from .detectors.yolo_28h_detector import YOLO28HDetector
                    self._panel_detector = YOLO28HDetector(device='cpu')
                    print("🔄 Fallback final vers YOLO28HDetector")
                except Exception as e3:
                    print(f"❌ ERREUR fallback final: {e3}")
                    raise e3
            
        self._panel_mode = False
        self._panel_framing = "fit"  # "fit" | "fill" | "center"
        self._panel_cache: dict[int, List[QRectF]] = {}       # PDF coordinates
        self._balloon_cache: dict[int, List[QRectF]] = {}     # PDF coordinates  
        self._panel_view_cache: dict[int, List[QRectF]] = {}  # VIEW coordinates (final)
        self._balloon_view_cache: dict[int, List[QRectF]] = {} # VIEW coordinates (final)
        self._page_overview_mode = False  # True when showing full page before panels
        self._panel_index = -1
        self._det_dpi = 130.0  # detection render DPI (130 pour éviter fragmentation)
        
        # Post-processing settings
        self._snap_to_gutters = True  # Enable border snapping to gutters
        self._split_by_gutters = True  # Enable splitting large panels by internal gutters
        
        # Cache amélioré si disponible
        if ENHANCED_CACHE_AVAILABLE and PanelCacheManager is not None:
            self._enhanced_cache = PanelCacheManager()
            pdebug("✅ Cache amélioré activé")
        else:
            self._enhanced_cache = None
            pdebug("ℹ️  Cache amélioré non disponible, utilisation cache simple")
        
        # Auto-load trained model if available
        trained_model = "../../runs/detect/overfit_small/weights/best.pt"
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
        
        # Raccourcis clavier pour navigation AR
        self._setup_ar_shortcuts()

    def _setup_ar_shortcuts(self):
        """Configure les raccourcis clavier pour la navigation AR."""
        # Flèches et Page Up/Down pour navigation
        shortcuts = [
            (Qt.Key.Key_Left, self.nav_prev),
            (Qt.Key.Key_Right, self.nav_next),
            (Qt.Key.Key_PageUp, self.nav_prev),
            (Qt.Key.Key_PageDown, self.nav_next),
        ]
        
        for key, slot in shortcuts:
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(slot)
            self.addAction(action)

    def _setup_window_icon(self):
        """Configure l'icône de la fenêtre."""
        try:
            # Essayer d'utiliser l'icône configurée par main.py
            icon_path = os.environ.get('ANCOMICSVIEWER_ICON')
            if not icon_path:
                # Fallback vers l'icône locale
                icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/icon.ico")
            
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

        # ======= MENU ULTRA-SIMPLIFIÉ POUR YOLO 28H =======
        
        # Informations du modèle
        info = self._panel_detector.get_model_info()
        model_info = menu.addAction(f"🔥 Modèle: {info['name']}")
        model_info.setEnabled(False)  # Juste informatif
        
        conf_info = menu.addAction(f"⚙️ Confidence: {info['confidence']}")
        conf_info.setEnabled(False)  # Juste informatif
        
        menu.addSeparator()
        
        # DPI de détection (les seules options utiles)
        dpi150 = menu.addAction("Detection DPI: 150")
        dpi200 = menu.addAction("Detection DPI: 200")
        dpi150.triggered.connect(lambda: self._set_det_dpi(150.0))
        dpi200.triggered.connect(lambda: self._set_det_dpi(200.0))

        menu.addSeparator()
        
        # Actions de re-détection
        rerun = menu.addAction("Re-run detection (this page)")
        rerun.triggered.connect(self._rerun_detection_current_page)
        rerun_all = menu.addAction("Re-run detection (all pages)")
        rerun_all.triggered.connect(self._rerun_detection_all)
        
        menu.addSeparator()
        
        # DEBUG: Force clear cache and re-detect
        debug_redetect = menu.addAction("🔧 DEBUG: Force clear cache & re-detect (Ctrl+Shift+R)")
        debug_redetect.triggered.connect(self.debug_force_redetect_current_page)
        debug_redetect.setShortcut(QKeySequence("Ctrl+Shift+R"))

        # ======= FINI - PLUS DE COMPLEXITÉ ! =======
        # 🔥 SYSTÈME ULTRA-SIMPLIFIÉ : YOLO 28H UNIQUEMENT
        # Tous les anciens menus détecteur ont été supprimés

        menu.addSeparator()

        # 🔥 MÉNAGE TERMINÉ - SYSTÈME ULTRA-SIMPLIFIÉ 
        # Suppression complète de tous les anciens détecteurs et menus
        
        # Advanced tuning dialog (désactivé pour simplification)
        # menu.addSeparator()
        # adv = menu.addAction("Advanced tuning…")
        # def _open_tuning():
        #     dlg = PanelTuningDialog(self, self._panel_detector, self._det_dpi)
        #     dlg.exec()
        # adv.triggered.connect(_open_tuning)

        # 🔥 MÉNAGE TERMINÉ - SYSTÈME ULTRA-SIMPLIFIÉ 
        # Suppression complète de tous les anciens détecteurs et menus

        # ======= MÉTRIQUES SIMPLIFIÉES =======

        # ======= MÉTRIQUES SIMPLIFIÉES =======

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

        # ---- Cache submenu ----
        menu.addSeparator()
        cache_menu = menu.addMenu("Cache")
        
        act_cache_stats = cache_menu.addAction("📊 Statistiques")
        act_cache_clear = cache_menu.addAction("🧹 Vider cache fichier")
        act_cache_clear_all = cache_menu.addAction("🗑️ Vider tout le cache")
        
        def _show_cache_stats():
            if self._enhanced_cache:
                info = self._enhanced_cache.get_cache_info()
                msg = f"""📊 Statistiques du Cache Amélioré

💾 Mémoire:
  • {info['memory']['files']} fichiers
  • {info['memory']['pages']} pages

💿 Disque:
  • {info['disk']['files']} fichiers  
  • {info['disk']['size_mb']:.1f} MB
  • {info['disk']['path']}

📈 Performance:
  • Hits mémoire: {info['stats']['memory_hits']}
  • Hits disque: {info['stats']['disk_hits']}
  • Misses: {info['stats']['misses']}
  • Sauvegardes: {info['stats']['saves']}"""
            else:
                msg = "Cache amélioré non disponible\nUtilisation du cache mémoire simple"
            
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Statistiques Cache", msg)
        
        def _clear_file_cache():
            if self._enhanced_cache and self._current_path:
                self._enhanced_cache.clear_file_cache(self._current_path)
                self._panel_cache.clear()
                self.statusBar().showMessage("Cache fichier vidé", 2000)
            else:
                self._panel_cache.clear()
                self.statusBar().showMessage("Cache mémoire vidé", 2000)
        
        def _clear_all_cache():
            if self._enhanced_cache:
                self._enhanced_cache.clear_all_cache()
            self._panel_cache.clear()
            self.statusBar().showMessage("Tout le cache vidé", 2000)
        
        act_cache_stats.triggered.connect(_show_cache_stats)
        act_cache_clear.triggered.connect(_clear_file_cache)
        act_cache_clear_all.triggered.connect(_clear_all_cache)

        tb.addWidget(settings_btn)

    def _add_sep(self, tb: QToolBar):
        sep = QWidget()
        sep.setFixedWidth(6)
        tb.addWidget(sep)

    # ---------- Settings handlers (simplifiés) ----------
    def _on_toggle_debug(self, checked: bool):
        self._debug_panels = checked
        # 🔥 YOLO28HDetector n'a pas de propriété debug - simplifié
        self.statusBar().showMessage(f"Debug logs {'ON' if checked else 'OFF'}", 1500)

    def _on_toggle_canny(self, checked: bool):
        # 🔥 YOLO28HDetector n'utilise pas Canny - simplifié
        self.statusBar().showMessage(f"Canny fallback {'ON' if checked else 'OFF'}", 1500)

    def _on_toggle_rtl(self, checked: bool):
        # 🔥 YOLO28HDetector n'a pas de mode RTL - simplifié
        self.statusBar().showMessage(f"Reading {'RTL' if checked else 'LTR'}", 1500)

    def _on_toggle_snap_gutters(self, checked: bool):
        """Toggle border snapping to gutters."""
        self._snap_to_gutters = checked
        # Clear cache to force re-detection with new settings
        self._panel_cache.clear()
        self._balloon_cache.clear()
        self._ensure_panels(force=True)
        self.statusBar().showMessage(f"Snap to gutters {'ON' if checked else 'OFF'}", 1500)

    def _on_toggle_split_panels(self, checked: bool):
        """Toggle splitting large panels by internal gutters.""" 
        self._split_by_gutters = checked
        # Clear cache to force re-detection with new settings
        self._panel_cache.clear()
        self._balloon_cache.clear()
        self._ensure_panels(force=True)
        self.statusBar().showMessage(f"Split large panels {'ON' if checked else 'OFF'}", 1500)

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
            self._balloon_cache.clear()
            self._ensure_panels(force=True)
            cur = self.view.pageNavigator().currentPage()
            rects = self._panel_cache.get(cur, [])
            balloons = self._balloon_cache.get(cur, [])
            self.view.set_panel_overlay(rects, balloons, self._panel_mode)
            self.view.viewport().update()
            self.statusBar().showMessage("Panel tuning applied", 1500)
        except Exception:
            pdebug("_apply_panel_tuning error:\n" + traceback.format_exc())

    # ---------- Public API for command-line control (simplifié) ----------
    def apply_preset(self, name: str):
        """Apply a detection preset by name. Public API for command-line usage."""
        # 🔥 SYSTÈME SIMPLIFIÉ - YOLO28HDetector n'a pas de presets configurables
        name_lower = name.lower()
        
        if name_lower in ("fb", "franco-belge", "francobelge"):
            self._det_dpi = 200
        elif name_lower in ("manga", "jp"):
            self._det_dpi = 150
        elif name_lower in ("newspaper", "news", "us"):
            self._det_dpi = 200
        
        # Le YOLO28HDetector fonctionne avec ses paramètres fixes optimaux
        self._apply_panel_tuning(self._det_dpi)
        pdebug(f"🔥 Preset '{name}' appliqué - DPI: {self._det_dpi} (YOLO28HDetector)")

    def set_detector(self, name: str):
        """Switch detector by name. Public API for command-line usage."""
        name_lower = name.lower()
        
        if name_lower in ("heur", "heuristic"):
            from .main_app import PanelDetector
            self._panel_detector = PanelDetector(debug=self._debug_panels)
        elif name_lower in ("yolo", "yolov8"):
            if not self._ml_weights:
                pdebug("Warning: No ML weights loaded, cannot switch to YOLO detector")
                return
            try:
                from .detectors.yolo_seg import YoloSegPanelDetector
                self._panel_detector = YoloSegPanelDetector(weights=self._ml_weights, rtl=False)
            except Exception as e:
                pdebug(f"Error loading YOLO detector: {e}")
                return
        elif name_lower in ("multibd", "multibd+"):
            try:
                from .detectors.multibd_detector import MultiBDPanelDetector
                self._panel_detector = MultiBDPanelDetector()
            except Exception as e:
                pdebug(f"Error loading Multi-BD detector: {e}")
                return
        
        self._apply_panel_tuning(self._det_dpi)

    def open_on_start(self, pdf_path: str, page: int = 0):
        """Open a PDF file on startup. Public API for command-line usage."""
        if pdf_path and os.path.exists(pdf_path):
            if self.load_pdf(pdf_path):
                self.view.pageNavigator().jump(max(0, page), QPointF(0, 0))
                return True
        return False

    def _rerun_detection_all(self):
        """Clear the detector cache for all pages and re-run detection for the current page."""
        try:
            self._panel_cache.clear()
            self._balloon_cache.clear()
            self._panel_index = -1
            self._ensure_panels(force=True)
            cur = self.view.pageNavigator().currentPage()
            rects = self._panel_cache.get(cur, [])
            balloons = self._balloon_cache.get(cur, [])
            self.view.set_panel_overlay(rects, balloons, self._panel_mode)
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
    
    def _save_last_file(self, path: str):
        """Save the path of the last opened file to settings."""
        try:
            settings = QSettings("AnComicsViewer", "AnComicsViewer")
            settings.setValue("lastFile", path)
            pdebug(f"💾 Saved last file: {path}")
        except Exception as e:
            pdebug(f"⚠️ Failed to save last file setting: {e}")
    
    # ---------- Navigation AR ----------
    def enable_ar_mode(self):
        """Active le mode AR avec PageView."""
        ar_available = globals().get('AR_AVAILABLE', False)
        if not ar_available:
            return
            
        try:
            from .ui.page_view import PageView
            from .detectors.adaptive_ultra_robust_detector import AdaptiveUltraRobustDetector
            
            pdebug("🔄 Activation du mode AR...")
            
            # Créer le PageView
            self.ar_page_view = PageView()
            
            # Créer le détecteur adaptatif
            self.ar_detector = AdaptiveUltraRobustDetector()
            
            # Remplacer la vue centrale
            if hasattr(self, 'view') and self.view:
                self.traditional_view = self.view
                
            if hasattr(self, 'setCentralWidget'):
                self.setCentralWidget(self.ar_page_view)
            
            self.ar_mode_enabled = True
            pdebug("✅ Mode AR activé - PageView opérationnel")
            
        except Exception as e:
            pdebug(f"❌ Erreur activation AR: {e}")
            import traceback
            traceback.print_exc()

    def ar_load_and_render_pdf(self, pdf_path: str, page_num: int = 0):
        """Charge un PDF et rend une page avec détection AR."""
        ar_available = globals().get('AR_AVAILABLE', False)
        if not ar_available or not hasattr(self, 'ar_page_view') or not self.ar_page_view:
            pdebug("❌ Mode AR non activé ou PageView manquant")
            return None
            
        try:
            from PySide6.QtPdf import QPdfDocument
            
            pdebug(f"📖 Chargement PDF: {pdf_path}")
            
            # Fermer le document précédent s'il existe
            if hasattr(self, 'ar_pdf_document') and self.ar_pdf_document:
                self.ar_pdf_document.close()
            
            # Créer et charger le nouveau document
            self.ar_pdf_document = QPdfDocument()
            self.ar_pdf_document.load(pdf_path)
            
            if self.ar_pdf_document.status() != QPdfDocument.Status.Ready:
                pdebug(f"❌ Impossible de charger le PDF: {pdf_path}")
                return None
                
            # Conserver les infos pour la navigation
            self.ar_pdf_path = pdf_path
            self.ar_current_page = page_num
                
            pdebug(f"✅ PDF chargé - {self.ar_pdf_document.pageCount()} pages")
            
            # Rendre la page demandée
            return self.ar_render_page(page_num)
            
        except Exception as e:
            pdebug(f"❌ Erreur rendu AR: {e}")
            import traceback
            traceback.print_exc()
            return None

    def ar_render_page(self, page_num: int):
        """Rend une page spécifique du PDF AR."""
        if not hasattr(self, 'ar_pdf_document') or not self.ar_pdf_document:
            pdebug("❌ Pas de document PDF AR")
            return None
            
        if page_num < 0 or page_num >= self.ar_pdf_document.pageCount():
            pdebug(f"❌ Page {page_num} inexistante (max: {self.ar_pdf_document.pageCount()-1})")
            return None
        
        try:
            # Rendre la page
            page_size = self.ar_pdf_document.pagePointSize(page_num)
            dpi = 200
            qimg = self.ar_pdf_document.render(page_num, QSize(int(page_size.width() * dpi / 72), int(page_size.height() * dpi / 72)))
            
            if qimg.isNull():
                pdebug(f"❌ Échec rendu page {page_num}")
                return None
                
            pdebug(f"✅ Page {page_num} rendue: {qimg.width()}x{qimg.height()}")
            
            # Mettre à jour l'état
            self.ar_current_page = page_num
            self.ar_current_qimage = qimg
            
            # Afficher dans PageView
            self.ar_page_view.show_qimage(qimg)
            
            # Lancer la détection
            if hasattr(self, 'ar_detector') and self.ar_detector:
                dets = self.ar_detector.detect_on_qimage(qimg)
                pdebug(f"🔍 Détections: {len(dets)} panels")
                
                # Dessiner les overlays
                self.ar_page_view.draw_detections(dets, show_fullframe_debug=True)
            
            return qimg
            
        except Exception as e:
            pdebug(f"❌ Erreur rendu page {page_num}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def ar_next_page(self) -> bool:
        """Aller à la page suivante."""
        if not hasattr(self, 'ar_pdf_document') or not self.ar_pdf_document:
            return False
            
        next_page = getattr(self, 'ar_current_page', 0) + 1
        if next_page < self.ar_pdf_document.pageCount():
            return self.ar_render_page(next_page) is not None
        return False
    
    def ar_prev_page(self) -> bool:
        """Aller à la page précédente."""
        if not hasattr(self, 'ar_pdf_document') or not self.ar_pdf_document:
            return False
            
        prev_page = getattr(self, 'ar_current_page', 0) - 1
        if prev_page >= 0:
            return self.ar_render_page(prev_page) is not None
        return False

    def _load_last_file(self) -> Optional[str]:
        """Load the path of the last opened file from settings."""
        try:
            settings = QSettings("AnComicsViewer", "AnComicsViewer")
            path = settings.value("lastFile", "")
            if path and isinstance(path, str) and os.path.exists(path):
                pdebug(f"📂 Found last file: {path}")
                return path
            elif path:
                pdebug(f"⚠️ Last file no longer exists: {path}")
        except Exception as e:
            pdebug(f"⚠️ Failed to load last file setting: {e}")
        return None
    
    def _try_reopen_last_file(self):
        """Attempt to reopen the last file if it still exists."""
        last_file = self._load_last_file()
        if last_file:
            pdebug(f"🔄 Attempting to reopen last file: {os.path.basename(last_file)}")
            return self.load_pdf(last_file)
        return False

    def _load_pdf_ar_mode(self, path: str) -> bool:
        """Charge un PDF en mode AR avec PageView."""
        try:
            pdebug(f"🔬 Chargement PDF en mode AR: {path}")
            
            # Activer le mode AR via l'intégration
            self.enable_ar_mode()
            
            # Afficher la page 0 avec le PDF chargé
            qimage = self.ar_load_and_render_pdf(path, 0)
            
            if qimage is not None:
                pdebug(f"✅ PDF AR chargé avec succès!")
                self._current_path = path
                self.setWindowTitle(f"AnComicsViewer AR — {os.path.basename(path)}")
                return True
            else:
                pdebug(f"❌ Échec du rendu AR")
                return False
                
        except Exception as e:
            pdebug(f"❌ Erreur mode AR: {e}")
            traceback.print_exc()
            return False

    def load_pdf(self, path: str) -> bool:
        pdebug(f"🔄 load_pdf() called with: {path}")
        
        if not path or not os.path.exists(path) or not path.lower().endswith(".pdf"):
            pdebug(f"❌ Invalid PDF path: {path}")
            QMessageBox.warning(self, "Invalid File", "Please select a valid PDF file.")
            return False

        pdebug(f"✅ Path validation OK")

        # Si mode AR activé, utiliser la vue AR
        if self._ar_mode_enabled and AR_AVAILABLE:
            pdebug(f"🔬 Mode AR détecté - Basculement vers PageView...")
            return self._load_pdf_ar_mode(path)

        # Release previous doc fully (Windows-safe)
        if self.document is not None:
            pdebug(f"🧹 Releasing previous document...")
            try:
                self.view.setDocument(QPdfDocument())  # clear ref from view
                pdebug(f"✅ View cleared")
                self.document.close()
                pdebug(f"✅ Document closed")
                self.document.deleteLater()
                pdebug(f"✅ Document deleted")
            except Exception as e:
                pdebug(f"⚠️ Error releasing previous doc: {e}")
            self.document = None
            pdebug(f"✅ Document set to None")

        pdebug(f"📖 Creating new QPdfDocument...")
        doc = QPdfDocument(self)
        pdebug(f"✅ QPdfDocument created")
        
        pdebug(f"📂 Loading PDF file...")
        err = doc.load(path)
        pdebug(f"✅ doc.load() returned: {err}")
        
        success = False
        try:
            if hasattr(QPdfDocument.Error, "None_") and err == QPdfDocument.Error.None_:
                success = True
                pdebug(f"✅ Success via None_ check")
        except Exception as e:
            pdebug(f"⚠️ Error checking None_: {e}")
            
        try:
            # Simple check: if err is 0 or has value 0, consider success
            if err == 0 or str(err) == "0" or getattr(err, "value", None) == 0:
                success = True
                pdebug(f"✅ Success via zero check")
        except Exception as e:
            pdebug(f"⚠️ Error checking zero: {e}")

        pdebug(f"📊 Load success: {success}")
        
        if not success or doc.pageCount() <= 0:
            pdebug(f"❌ Load failed or no pages: success={success}, pages={doc.pageCount()}")
            QMessageBox.critical(self, "Load Error", "Failed to load PDF (corrupted/unsupported).")
            doc.deleteLater()
            return False

        pdebug(f"✅ PDF loaded successfully, pages: {doc.pageCount()}")
        
        pdebug(f"🔗 Setting document references...")
        self.document = doc
        pdebug(f"✅ self.document set")
        
        self._current_path = path
        pdebug(f"✅ _current_path set")
        
        pdebug(f"🖼️ Setting view document...")
        self.view.setDocument(self.document)
        pdebug(f"✅ view.setDocument() done")
        
        pdebug(f"🏷️ Setting window title...")
        self.setWindowTitle(f"ComicsView — {os.path.basename(path)}")
        pdebug(f"✅ Window title set")

        # Save this file as the last opened file
        pdebug(f"💾 Saving as last file...")
        self._save_last_file(path)
        pdebug(f"✅ Last file saved")

        pdebug(f"📐 Calling fit_page()...")
        self.fit_page()  # sensible default for comics
        pdebug(f"✅ fit_page() done")
        
        pdebug(f"🧹 Clearing panel cache...")
        self._panel_cache.clear()
        self._balloon_cache.clear()
        pdebug(f"✅ Panel cache cleared")
        
        self._panel_index = -1
        pdebug(f"✅ Panel index reset")
        
        pdebug(f"📊 Updating status...")
        self._update_status()
        pdebug(f"✅ Status updated")
        
        pdebug(f"🎉 load_pdf() completed successfully!")
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
        # Mode AR : utiliser la navigation AR
        ar_available = globals().get('AR_AVAILABLE', False)
        if self._ar_mode_enabled and ar_available:
            pdebug("🔬 Navigation AR : page précédente")
            self.ar_prev_page()
        elif self.document:
            cur = self.view.pageNavigator().currentPage()
            if cur > 0:
                self.view.pageNavigator().jump(cur - 1, QPointF(0, 0))

    def nav_next(self):
        # Mode AR : utiliser la navigation AR
        ar_available = globals().get('AR_AVAILABLE', False)
        if self._ar_mode_enabled and ar_available:
            pdebug("🔬 Navigation AR : page suivante")
            self.ar_next_page()
        elif self.document:
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
        pdebug(f"📐 fit_page() called")
        if self.document:
            pdebug(f"📐 Setting zoom mode to FitInView...")
            self.view.setZoomMode(QPdfView.ZoomMode.FitInView)
            pdebug(f"✅ Zoom mode set")
            pdebug(f"📊 Calling _update_status from fit_page...")
            self._update_status()
            pdebug(f"✅ _update_status done from fit_page")
        else:
            pdebug(f"⚠️ fit_page() called but no document")

    # ---------- Panels ----------
    def toggle_panels(self):
        if not self.document:
            return
        self._panel_mode = not self._panel_mode
        self.view.setPageMode(QPdfView.PageMode.SinglePage if self._panel_mode else QPdfView.PageMode.MultiPage)
        self._panel_index = -1
        self._page_overview_mode = False  # Reset overview mode
        self._ensure_panels(force=True)
        cur = self.view.pageNavigator().currentPage()
        rects = self._panel_cache.get(cur, [])
        balloons = self._balloon_cache.get(cur, [])
        pdebug(f"toggle: page={cur} rects={len(rects)}")
        self.view.set_panel_overlay(rects, balloons, self._panel_mode)
        self.statusBar().showMessage("Panel mode ON" if self._panel_mode else "Panel mode OFF", 2000)

    def panel_next(self):
        """Navigation vers la case suivante avec saut de page automatique."""
        if not (self.document and self._panel_mode):
            return
        try:
            cur = self.view.pageNavigator().currentPage()
            self._ensure_panels_for(cur)
            rects = self._panel_cache.get(cur, [])
            
            # Si on est en mode overview, passer au premier panel de la page
            if self._page_overview_mode:
                if rects:
                    self._page_overview_mode = False
                    self._panel_index = 0
                    self._focus_panel(rects[0])
                    pdebug(f"panel_next (overview->panel) -> page {cur}, panel 1/{len(rects)}")
                    self.statusBar().showMessage(f"Page {cur + 1}: panel 1/{len(rects)}", 3000)
                    return
                else:
                    # Pas de panels sur cette page, continuer vers la suivante
                    self._page_overview_mode = False
                    self._panel_index = -1
            
            # Cas spécial : gestion de l'état initial _panel_index == -1
            if self._panel_index == -1:
                # Chercher la première page avec des panels à partir de la page courante
                for page in range(cur, self.document.pageCount()):
                    self._ensure_panels_for(page)
                    page_rects = self._panel_cache.get(page, [])
                    if page_rects:
                        if page != cur:
                            # Aller à la nouvelle page et montrer l'overview
                            self._show_page_overview(page)
                            self._panel_index = -1  # Sera géré au prochain panel_next
                            pdebug(f"panel_next (init) -> page {page} overview")
                        else:
                            # Même page, commencer directement avec les panels
                            self._panel_index = 0
                            self._focus_panel(page_rects[0])
                            pdebug(f"panel_next (init) -> page {page}, panel 1/{len(page_rects)}")
                            self.statusBar().showMessage(f"Page {page + 1}: panel 1/{len(page_rects)}", 3000)
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
                        # Afficher d'abord la page overview
                        self._show_page_overview(page)
                        self._panel_index = -1  # Sera géré au prochain panel_next
                        pdebug(f"panel_next (no panels) -> page {page} overview")
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
                        # D'abord montrer la page entière comme overview
                        self._show_page_overview(page)
                        self._panel_index = -1  # Sera incrémenté au prochain panel_next
                        pdebug(f"panel_next (page jump) -> page {page} overview")
                        return
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
            
            # Si on est en mode overview, aller à la dernière case de la page
            if self._page_overview_mode:
                if rects:
                    self._page_overview_mode = False
                    self._panel_index = len(rects) - 1
                    self._focus_panel(rects[self._panel_index])
                    pdebug(f"panel_prev (overview->panel) -> page {cur}, panel {len(rects)}/{len(rects)}")
                    self.statusBar().showMessage(f"Page {cur + 1}: panel {len(rects)}/{len(rects)}", 3000)
                    return
                else:
                    # Pas de panels sur cette page, continuer vers la précédente
                    self._page_overview_mode = False
                    self._panel_index = -1
            
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
        balloons = self._balloon_cache.get(page, [])
        self.view.set_panel_overlay(rects, balloons, self._panel_mode)

    def _ensure_panels_for(self, page: int, force: bool = False):
        """Helper pour détecter les cases d'une page donnée avec cache amélioré."""
        if not self.document:
            pdebug("ensure_panels_for: no document")
            return
        
        # Vérifier le cache amélioré d'abord
        if self._enhanced_cache and self._current_path and not force:
            cached_panels = self._enhanced_cache.get_panels(
                self._current_path, page, self._panel_detector
            )
            if cached_panels is not None:
                self._panel_cache[page] = cached_panels
                pdebug(f"ensure_panels_for: page={page}, panels={len(cached_panels)} (cache hit)")
                return
        
        # Vérifier le cache mémoire classique
        if (not force) and page in self._panel_cache:
            return
            
        try:
            pt = self.document.pagePointSize(page)
            dpi = self._det_dpi
            scale = dpi / 72.0
            qsize = QSizeF(pt.width() * scale, pt.height() * scale).toSize()
            qimg = self.document.render(page, qsize)
            
            # Détecter panels ET balloons avec les vraies dimensions
            if hasattr(self._panel_detector, 'detect_panels_and_balloons'):
                # Passer à la fois la taille de page PDF (pt) ET la taille image (qsize)
                panels, balloons = self._panel_detector.detect_panels_and_balloons(qimg, pt, qsize)
                print(f"🔍 DÉTECTION Page {page}: {len(panels)} panels + {len(balloons)} balloons")
                print(f"   📐 Page size: {pt.width():.1f}x{pt.height():.1f} pts")
                print(f"   🖼️ Image size: {qsize.width()}x{qsize.height()} px")
                print(f"   🟢 PANELS détectés:")
                for i, panel in enumerate(panels):
                    print(f"      P{i+1}: ({panel.x():.1f},{panel.y():.1f},{panel.width():.1f}x{panel.height():.1f})")
                print(f"   🔴 BALLOONS détectés:")
                for i, balloon in enumerate(balloons):
                    print(f"      B{i+1}: ({balloon.x():.1f},{balloon.y():.1f},{balloon.width():.1f}x{balloon.height():.1f})")
            else:
                panels = self._panel_detector.detect_panels(qimg, pt)
                balloons = []
                print(f"🔍 DÉTECTION Page {page}: {len(panels)} panels (pas de balloons)")
            
            self._panel_cache[page] = panels
            self._balloon_cache[page] = balloons
            
            # Sauvegarder dans le cache amélioré
            if self._enhanced_cache and self._current_path:
                self._enhanced_cache.save_panels(
                    self._current_path, page, panels, self._panel_detector
                )
            
            pdebug(f"ensure_panels_for: page={page}, panels={len(panels)} balloons={len(balloons)} @ {int(dpi)} DPI")
        except Exception:
            pdebug("ensure_panels_for error:\n" + traceback.format_exc())
            self._panel_cache[page] = []

    def _ensure_panels(self, force: bool=False):
        """Assure la détection des cases pour la page courante."""
        if not self.document:
            pdebug("ensure_panels: no document")
            return
        cur = self.view.pageNavigator().currentPage()
        
        # 🔍 DEBUG ÉTAT VIEWPORT APRÈS DÉLAI (si force=True)
        if force:
            doc = self.view.document()
            if doc:
                page_pts = doc.pagePointSize(cur)
                z = self.view.zoomFactor()
                vw = self.view.viewport().width()
                vh = self.view.viewport().height()
                content_w = page_pts.width() * z
                content_h = page_pts.height() * z
                pad_x = max(0.0, (vw - content_w) / 2.0)
                pad_y = max(0.0, (vh - content_h) / 2.0)
                sx = self.view.horizontalScrollBar().value()
                sy = self.view.verticalScrollBar().value()
                print(f"🔧 VIEWPORT STATE APRÈS DÉLAI Page {cur}:")
                print(f"   🔍 Zoom: {z:.3f}")
                print(f"   🖼️ Viewport: {vw}x{vh}")
                print(f"   📄 Content: {content_w:.1f}x{content_h:.1f}")
                print(f"   🅿️ Padding: ({pad_x:.1f},{pad_y:.1f})")
                print(f"   📜 Scroll: ({sx},{sy})")
        
        self._ensure_panels_for(cur, force)

    def _delayed_ensure_panels(self, attempt: int = 1, max_attempts: int = 5):
        """Re-détection avec vérification de stabilité du viewport."""
        if not self.document:
            return
            
        cur = self.view.pageNavigator().currentPage()
        doc = self.view.document()
        
        if doc:
            page_pts = doc.pagePointSize(cur)
            z = self.view.zoomFactor()
            vw = self.view.viewport().width()
            vh = self.view.viewport().height()
            content_w = page_pts.width() * z
            content_h = page_pts.height() * z
            pad_x = max(0.0, (vw - content_w) / 2.0)
            pad_y = max(0.0, (vh - content_h) / 2.0)
            sx = self.view.horizontalScrollBar().value()
            sy = self.view.verticalScrollBar().value()
            
            print(f"🔧 VIEWPORT APRÈS DÉLAI {attempt} Page {cur}:")
            print(f"   🔍 Zoom: {z:.3f}")
            print(f"   🖼️ Viewport: {vw}x{vh}")
            print(f"   📄 Content: {content_w:.1f}x{content_h:.1f}")
            print(f"   🅿️ Padding: ({pad_x:.1f},{pad_y:.1f})")
            print(f"   📜 Scroll: ({sx},{sy})")
            
            # 🔄 VÉRIFIER si le zoom OU le scroll ont changé depuis la dernière conversion VIEW
            zoom_changed = (hasattr(self, '_last_cached_zoom') and 
                           abs(z - self._last_cached_zoom) > 0.001)
            
            scroll_changed = (hasattr(self, '_last_cached_scroll') and 
                             (abs(sx - self._last_cached_scroll[0]) > 5 or 
                              abs(sy - self._last_cached_scroll[1]) > 5))
            
            # 🎯 VÉRIFIER STABILITÉ DU ZOOM entre deux appels successifs
            if hasattr(self, '_last_attempt_zoom'):
                zoom_stabilizing = abs(z - self._last_attempt_zoom) < 0.001
                print(f"🔄 ZOOM: {self._last_attempt_zoom:.3f} → {z:.3f} (stable: {zoom_stabilizing})")
            else:
                zoom_stabilizing = False
                print(f"🔄 ZOOM: Premier appel à {z:.3f}")
            
            # Mémoriser le zoom de cette tentative
            self._last_attempt_zoom = z
            
            if zoom_changed:
                print(f"🔄 ZOOM CHANGÉ: {getattr(self, '_last_cached_zoom', 'N/A')} → {z:.3f}")
                print("🗑️ Invalidation cache VIEW (zoom différent)")
                self._panel_view_cache.clear()
                self._balloon_view_cache.clear()
            
            if scroll_changed:
                print(f"🔄 SCROLL CHANGÉ: {getattr(self, '_last_cached_scroll', 'N/A')} → ({sx},{sy})")
                print("🗑️ Invalidation cache VIEW (scroll différent)")
                self._panel_view_cache.clear()
                self._balloon_view_cache.clear()
            
            # 🎯 VÉRIFIER STABILITÉ : zoom valide ET scroll proche de zéro (page overview) ET zoom stabilisé
            viewport_stable = (z > 0.1 and content_w > 0 and content_h > 0)
            scroll_stable = (abs(sx) < 50 and abs(sy) < 50)  # Tolérance pour scroll proche de 0
            
            # 🔧 CONDITION SUPPLÉMENTAIRE : le zoom doit être stable entre deux tentatives
            # OU on a atteint le nombre max de tentatives
            zoom_stable = zoom_stabilizing or attempt >= max_attempts
            
            # Pour le mode overview, on s'attend à un scroll proche de (0,0) ET un zoom stable
            all_stable = viewport_stable and scroll_stable and zoom_stable
            
            # 🔍 LOGS SPÉCIAUX POUR PAGE 3
            if cur == 2:  # Page 3 (index 2)
                print(f"   🎯 PAGE 3 STABILITÉ: viewport={viewport_stable}, scroll={scroll_stable}, zoom_stable={zoom_stable} -> global={all_stable}")
            else:
                print(f"   📊 Stabilité: viewport={viewport_stable}, scroll={scroll_stable}, zoom_stable={zoom_stable} -> global={all_stable}")
            
            if all_stable or attempt >= max_attempts:
                print(f"✅ VIEWPORT {'STABLE' if all_stable else 'TIMEOUT'} - Lancement re-détection")
                
                # 🎯 TOUJOURS forcer la re-détection pour s'assurer d'avoir les bonnes coordonnées PDF
                self._ensure_panels(force=True)
                
                # 🎯 APRÈS la re-détection, récupérer les résultats et TOUJOURS recalculer les coordonnées VIEW
                cur = self.view.pageNavigator().currentPage()
                rects = self._panel_cache.get(cur, [])
                balloons = self._balloon_cache.get(cur, [])
                
                # 🔄 TOUJOURS CONVERTIR les coordonnées PDF vers VIEW (même si on a un cache)
                # Car le viewport peut avoir changé depuis la dernière fois
                view_panels = []
                view_balloons = []
                
                for panel in rects:
                    view_rect = self.view._page_rect_to_view(panel)
                    view_panels.append(view_rect)
                    
                for balloon in balloons:
                    view_rect = self.view._page_rect_to_view(balloon)
                    view_balloons.append(view_rect)
                
                # 💾 TOUJOURS mettre à jour le cache VIEW avec les nouvelles coordonnées
                self._panel_view_cache[cur] = view_panels
                self._balloon_view_cache[cur] = view_balloons
                
                # 🎯 MÉMORISER le zoom ET le scroll pour lesquels ces coordonnées VIEW sont valides
                self._last_cached_zoom = z
                self._last_cached_scroll = (sx, sy)
                
                print(f"📋 CACHE MISE À JOUR Page {cur} (zoom {z:.3f}, scroll ({sx},{sy})):")
                print(f"   🟢 PANELS: {len(rects)} (PDF) → {len(view_panels)} (VIEW) [RECALCUL FORCÉ]")
                for i, (panel, view_panel) in enumerate(zip(rects, view_panels)):
                    print(f"      P{i+1}: PDF({panel.x():.1f},{panel.y():.1f},{panel.width():.1f}x{panel.height():.1f}) → VIEW({view_panel.x():.1f},{view_panel.y():.1f},{view_panel.width():.1f}x{view_panel.height():.1f})")
                print(f"   🔴 BALLOONS: {len(balloons)} (PDF) → {len(view_balloons)} (VIEW) [RECALCUL FORCÉ]")
                for i, (balloon, view_balloon) in enumerate(zip(balloons, view_balloons)):
                    print(f"      B{i+1}: PDF({balloon.x():.1f},{balloon.y():.1f},{balloon.width():.1f}x{balloon.height():.1f}) → VIEW({view_balloon.x():.1f},{view_balloon.y():.1f},{view_balloon.width():.1f}x{view_balloon.height():.1f})")
                
                # ✅ RÉACTIVER les overlays avec les coordonnées VIEW pré-calculées
                self.view.set_panel_overlay(view_panels, view_balloons, self._panel_mode)
                print(f"✅ Overlays réactivés: {len(view_panels)} panels + {len(view_balloons)} balloons (coordonnées VIEW FRAÎCHES pour zoom {z:.3f})")
                
                # 🔄 FORCER le re-calcul des coordonnées d'affichage MAINTENANT
                QTimer.singleShot(10, lambda: self.view.viewport().update())
                print(f"🔄 Force refresh de l'affichage pour coordonnées stables")
            else:
                # 🔍 LOGS SPÉCIAUX POUR PAGE 3
                if cur == 2:  # Page 3 (index 2)
                    print(f"   ⏳ PAGE 3 VIEWPORT INSTABLE - Tentative {attempt+1}/{max_attempts} dans 150ms (attendre stabilisation zoom)")
                    QTimer.singleShot(150, lambda: self._delayed_ensure_panels(attempt + 1, max_attempts))
                else:
                    print(f"⏳ VIEWPORT INSTABLE - Tentative {attempt+1}/{max_attempts} dans 50ms")
                    QTimer.singleShot(50, lambda: self._delayed_ensure_panels(attempt + 1, max_attempts))

    def _show_page_overview(self, page: int):
        """Affiche la page entière comme une 'grande case' avant de naviguer case par case."""
        try:
            # Aller à la page
            self.view.pageNavigator().jump(page, QPointF(0, 0))
            
            # Ajuster pour voir la page entière
            self.view.setZoomMode(QPdfView.ZoomMode.FitInView)
            
            # Activer le mode overview
            self._page_overview_mode = True
            
            # Afficher l'overlay avec toutes les cases de la page
            self._ensure_panels_for(page)
            rects = self._panel_cache.get(page, [])
            balloons = self._balloon_cache.get(page, [])
            self.view.set_panel_overlay(rects, balloons, self._panel_mode)
            
            pdebug(f"page_overview -> page {page} ({len(rects)} panels)")
            self.statusBar().showMessage(f"Page {page + 1} overview - Next: panel navigation", 3000)
            
        except Exception:
            pdebug("_show_page_overview error:\n" + traceback.format_exc())

    def _force_clear_all_caches(self):
        """DEBUG: Force la suppression de tous les caches pour test."""
        pdebug("🔧 DEBUG: Forcing complete cache clear...")
        
        # Vider les caches mémoire
        self._panel_cache.clear()
        self._balloon_cache.clear()
        pdebug("✅ Memory caches cleared")
        
        # Vider le cache amélioré
        if self._enhanced_cache:
            try:
                # Vider tout le cache disque
                import shutil
                import os
                cache_dir = os.path.expanduser("~/.ancomicsviewer/cache")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                    pdebug("✅ Disk cache cleared")
            except Exception as e:
                pdebug(f"⚠️ Could not clear disk cache: {e}")
        
        # Vider le cache du détecteur YOLO
        try:
            detector_cache = getattr(self._panel_detector, '_cache', None)
            if detector_cache and hasattr(detector_cache, 'clear'):
                detector_cache.clear()
                pdebug("✅ YOLO detector cache cleared")
            else:
                clear_cache_method = getattr(self._panel_detector, 'clear_cache', None)
                if clear_cache_method:
                    clear_cache_method()
                    pdebug("✅ Detector cache cleared")
                else:
                    pdebug("⚠️ No detector cache to clear")
        except Exception as e:
            pdebug(f"⚠️ Could not clear detector cache: {e}")
        
        pdebug("🔧 All caches force-cleared - next detection will be fresh!")

    def debug_force_redetect_current_page(self):
        """DEBUG: Force la re-détection de la page courante."""
        if not self.document:
            return
        
        pdebug("🔧 DEBUG: Force re-detecting current page...")
        self._force_clear_all_caches()
        
        cur = self.view.pageNavigator().currentPage()
        self._ensure_panels_for(cur, force=True)
        
        # Mettre à jour l'affichage
        rects = self._panel_cache.get(cur, [])
        balloons = self._balloon_cache.get(cur, [])
        self.view.set_panel_overlay(rects, balloons, self._panel_mode)
        
        pdebug(f"✅ Re-detected: {len(rects)} panels + {len(balloons)} balloons on page {cur+1}")

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
        pdebug(f"📊 _update_status() called")
        try:
            if not self.document:
                pdebug(f"⚠️ No document in _update_status")
                self.statusBar().showMessage("No document loaded")
                return
            
            pdebug(f"📄 Getting page navigator...")
            nav = self.view.pageNavigator()
            pdebug(f"✅ Page navigator obtained")
            
            pdebug(f"📍 Getting current page...")
            current = nav.currentPage() + 1
            pdebug(f"✅ Current page: {current}")
            
            pdebug(f"📚 Getting page count...")
            total = self.document.pageCount()
            pdebug(f"✅ Total pages: {total}")
            
            pdebug(f"🔍 Getting zoom factor...")
            zoom = int(self.view.zoomFactor() * 100)
            pdebug(f"✅ Zoom factor: {zoom}%")
            
            pdebug(f"📝 Setting status message...")
            self.statusBar().showMessage(f"Page {current}/{total} | Zoom {zoom}%")
            pdebug(f"✅ Status message set")
            
        except Exception as e:
            pdebug(f"❌ Exception in _update_status: {e}")
            import traceback
            pdebug(f"🔍 Traceback: {traceback.format_exc()}")
            self.statusBar().showMessage("Status update error")

    def _on_page_changed(self):
        """Handle page change events."""
        cur = self.view.pageNavigator().currentPage()
        
        # 🔍 LOGS SPÉCIAUX POUR L'ARRIVÉE SUR PAGE 3
        if cur == 2:  # Page 3 (index 2)
            print(f"\n🎯 === ARRIVÉE SUR PAGE 3 === _on_page_changed ===")
            print(f"   📄 Page changée vers: {cur} (Page 3)")
            print(f"   🔍 Zoom actuel: {self.view.zoomFactor():.6f}")
            print(f"   📐 Viewport: {self.view.viewport().width()}x{self.view.viewport().height()}")
            
            # État des scrollbars
            hscroll = self.view.horizontalScrollBar()
            vscroll = self.view.verticalScrollBar()
            print(f"   📜 HScroll: value={hscroll.value()}, min={hscroll.minimum()}, max={hscroll.maximum()}")
            print(f"   📜 VScroll: value={vscroll.value()}, min={vscroll.minimum()}, max={vscroll.maximum()}")
            
            # État du document
            doc = self.document
            if doc:
                page_pts = doc.pagePointSize(cur)
                print(f"   📏 Page size: {page_pts.width():.1f}x{page_pts.height():.1f} pts")
            
            print(f"   🎯 Mode panel: {self._panel_mode}")
            print(f"   ⏰ ATTENDRE STABILISATION DU ZOOM AVANT AFFICHAGE")
        else:
            pdebug(f"📄 _on_page_changed() called")
            
        try:
            if self._panel_mode:
                if cur == 2:
                    print(f"   🎯 Panel mode is ON - invalidating caches")
                else:
                    pdebug(f"🎯 Panel mode is ON")
                self._panel_index = -1  # Reset panel navigation
                if cur == 2:
                    print(f"   ✅ Panel index reset")
                else:
                    pdebug(f"✅ Panel index reset")
                
                # 🔥 INVALIDATION COMPLÈTE DU CACHE pour éviter décalage
                if cur in self._panel_cache:
                    del self._panel_cache[cur]
                    if cur == 2:
                        print(f"   🗑️ Panel cache invalidé pour page {cur}")
                    else:
                        pdebug(f"🗑️ Panel cache invalidé pour page {cur}")
                if cur in self._balloon_cache:
                    del self._balloon_cache[cur]
                    if cur == 2:
                        print(f"   🗑️ Balloon cache invalidé pour page {cur}")
                    else:
                        pdebug(f"🗑️ Balloon cache invalidé pour page {cur}")
                # 🔥 INVALIDATION AUSSI DES COORDONNÉES VIEW PRÉ-CALCULÉES
                if cur in self._panel_view_cache:
                    del self._panel_view_cache[cur]
                    if cur == 2:
                        print(f"   🗑️ Panel VIEW cache invalidé pour page {cur}")
                    else:
                        pdebug(f"🗑️ Panel VIEW cache invalidé pour page {cur}")
                if cur in self._balloon_view_cache:
                    del self._balloon_view_cache[cur]
                    if cur == 2:
                        print(f"   🗑️ Balloon VIEW cache invalidé pour page {cur}")
                    else:
                        pdebug(f"🗑️ Balloon VIEW cache invalidé pour page {cur}")
                
                # Nettoyer aussi le cache du détecteur (safely)
                try:
                    cache = getattr(self._panel_detector, '_cache', None)
                    if cache is not None:
                        cache.clear()
                        if cur == 2:
                            print(f"   🗑️ Détecteur _cache cleared")
                        else:
                            pdebug(f"🗑️ Détecteur _cache cleared")
                    else:
                        cache = getattr(self._panel_detector, 'cache', None)
                        if cache is not None:
                            cache.clear()
                            pdebug(f"🗑️ Détecteur cache cleared")
                except Exception:
                    pass  # Ignore if detector doesn't have cache
                
                pdebug(f"🔄 ATTENDRE STABILISATION VIEWPORT puis re-détecter...")
                
                # 🔍 DEBUG ÉTAT VIEWPORT AVANT RE-DÉTECTION
                cur_page = self.view.pageNavigator().currentPage() 
                doc = self.view.document()
                if doc:
                    page_pts = doc.pagePointSize(cur_page)
                    z = self.view.zoomFactor()
                    vw = self.view.viewport().width()
                    vh = self.view.viewport().height()
                    content_w = page_pts.width() * z
                    content_h = page_pts.height() * z
                    pad_x = max(0.0, (vw - content_w) / 2.0)
                    pad_y = max(0.0, (vh - content_h) / 2.0)
                    sx = self.view.horizontalScrollBar().value()
                    sy = self.view.verticalScrollBar().value()
                    print(f"🔧 VIEWPORT INITIAL Page {cur_page}:")
                    print(f"   🔍 Zoom: {z:.3f}")
                    print(f"   🖼️ Viewport: {vw}x{vh}")
                    print(f"   📄 Content: {content_w:.1f}x{content_h:.1f}")
                    print(f"   🅿️ Padding: ({pad_x:.1f},{pad_y:.1f})")
                    print(f"   📜 Scroll: ({sx},{sy})")
                
                # ⏱️ DIFFÉRER avec délai plus long ET vérification multiple
                QTimer.singleShot(100, lambda: self._delayed_ensure_panels(attempt=1))
                pdebug(f"✅ _ensure_panels programmé avec délai 100ms + vérifications")
                
                # ❌ NE PAS lire du cache maintenant - il est vide !
                # ❌ L'affichage sera fait APRÈS la re-détection dans _delayed_ensure_panels
                pdebug(f"⏳ Affichage des overlays sera fait APRÈS la re-détection")
                
                # � DÉSACTIVER temporairement les overlays pour éviter affichage du cache vide
                self.view.set_panel_overlay([], [], False)
                pdebug(f"🚫 Overlays temporairement désactivés")
            else:
                pdebug(f"ℹ️ Panel mode is OFF, skipping panel logic")
        except Exception as e:
            pdebug(f"❌ Exception in _on_page_changed: {e}")
            pdebug("page_changed error:\n" + traceback.format_exc())


# -----------------------------
# Main entry point
# -----------------------------
def main():
    app = QApplication(sys.argv)
    window = ComicsView()
    window.show()
    
    # Process environment variables for command-line control
    preset = os.getenv("ANCOMICS_PRESET")
    if preset:
        window.apply_preset(preset)
    
    detector = os.getenv("ANCOMICS_DETECTOR")
    if detector:
        window.set_detector(detector)
    
    dpi_str = os.getenv("ANCOMICS_DPI")
    if dpi_str:
        try:
            window._det_dpi = int(dpi_str)
            window._apply_panel_tuning(window._det_dpi)
        except ValueError:
            pass
    
    pdf_path = os.getenv("ANCOMICS_PDF")
    if pdf_path:
        page = 0
        page_str = os.getenv("ANCOMICS_PAGE")
        if page_str:
            try:
                page = int(page_str)
            except ValueError:
                pass
        window.open_on_start(pdf_path, page)
    else:
        # No explicit PDF specified, try to reopen the last file
        window._try_reopen_last_file()
    
    return app.exec()


if __name__ == "__main__":
    main()
