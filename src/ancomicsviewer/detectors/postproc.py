# detectors/postproc.py
from typing import List, Tuple
import numpy as np, cv2
from PySide6.QtCore import QRectF, QSizeF

def lab_L(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)

def find_runs(bool_arr, min_len: int):
    runs, start = [], None
    for i, v in enumerate(list(bool_arr) + [False]):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    return runs

def _closest_center(runs, target: int):
    if not runs:
        return target
    return min(((abs(((a+b)//2) - target), (a+b)//2) for a, b in runs))[1]

def snap_rect_to_gutters(
    L_img: np.ndarray,
    rect: QRectF,
    page_size: QSizeF,
    px_per_pt: float,
    min_gutter_px: int = 8,
    cov: float = 0.90,
    max_expand_frac: float = 0.04,
    smooth_k: int = 17,
) -> QRectF:
    H, W = L_img.shape[:2]
    x0 = max(0, int(round(rect.left()   * px_per_pt)))
    y0 = max(0, int(round(rect.top()    * px_per_pt)))
    ww = max(1, int(round(rect.width()  * px_per_pt)))
    hh = max(1, int(round(rect.height() * px_per_pt)))

    pad = int(max_expand_frac * min(W, H))
    xa = max(0, x0 - pad); ya = max(0, y0 - pad)
    xb = min(W, x0 + ww + pad); yb = min(H, y0 + hh + pad)
    roi = L_img[ya:yb, xa:xb].astype(np.float32) / 255.0
    if roi.size == 0:
        return rect

    col = roi.mean(axis=0); row = roi.mean(axis=1)
    k = max(9, smooth_k)
    if col.size >= k: col = np.convolve(col, np.ones(k) / k, mode="same")
    if row.size >= k: row = np.convolve(row, np.ones(k) / k, mode="same")

    ct = (col.max() - col.min()) * 0.15 + col.mean()
    rt = (row.max() - row.min()) * 0.15 + row.mean()
    vbin = col >= ct; hbin = row >= rt

    v_runs = find_runs(vbin, min_gutter_px)
    h_runs = find_runs(hbin, min_gutter_px)

    Lx = x0 - xa; Rx = x0 + ww - xa
    Ty = y0 - ya; By = y0 + hh - ya

    newL = _closest_center(v_runs, Lx)
    newR = _closest_center(v_runs, Rx)
    newT = _closest_center(h_runs, Ty)
    newB = _closest_center(h_runs, By)

    nx = (xa + min(newL, newR)) / px_per_pt
    ny = (ya + min(newT, newB)) / px_per_pt
    nw = max(1.0 / px_per_pt, abs((xa + newR) - (xa + newL)) / px_per_pt)
    nh = max(1.0 / px_per_pt, abs((ya + newB) - (ya + newT)) / px_per_pt)
    return QRectF(nx, ny, nw, nh)

def snap_rect_to_gutters_rgb(
    rgb_img: np.ndarray,
    rect: QRectF,
    page_size: QSizeF,
    px_per_pt: float,
    min_gutter_px: int = 8,
    cov: float = 0.90,
    max_expand_frac: float = 0.04,
    smooth_k: int = 17,
) -> QRectF:
    """Version qui accepte une image RGB et fait la conversion LAB en interne."""
    L_img = lab_L(rgb_img)
    return snap_rect_to_gutters(L_img, rect, page_size, px_per_pt, min_gutter_px, cov, max_expand_frac, smooth_k)

# ============ NEW POST-PROCESSING FUNCTIONS ============

def _to_gray01(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to normalized grayscale."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return g.astype(np.float32) / 255.0

def _find_gutter_pos(profile: np.ndarray, center_idx: int, max_shift: int, look_for='bright'):
    """Scan left/right (or up/down) from a starting index and return the index of the
    strongest local maximum (bright gutter) within a limited radius."""
    lo = max(0, center_idx - max_shift)
    hi = min(len(profile)-1, center_idx + max_shift)
    window = profile[lo:hi+1]
    k = window.argmax() if look_for == 'bright' else window.argmin()
    return lo + int(k)

def snap_panels_to_gutters(rgb: np.ndarray, rect: QRectF, pad: int = 6, max_shift: int = 10) -> QRectF:
    """
    Pull each edge of 'rect' to the nearest bright gutter line using intensity projections.
    Works best on color pages with pale gutters.
    """
    h, w = rgb.shape[:2]
    x1, y1 = int(rect.left()), int(rect.top())
    x2, y2 = int(rect.right()), int(rect.bottom())
    x1 = np.clip(x1, 0, w-1); x2 = np.clip(x2, 0, w-1)
    y1 = np.clip(y1, 0, h-1); y2 = np.clip(y2, 0, h-1)

    g = _to_gray01(rgb)

    # vertical projections near each vertical edge
    vx_left  = g[y1:y2, max(0, x1-pad):min(w, x1+pad)].mean(axis=1) if x1 < x2 else None
    vx_right = g[y1:y2, max(0, x2-pad):min(w, x2+pad)].mean(axis=1) if x1 < x2 else None

    # horizontal projections near each horizontal edge
    hx_top    = g[max(0, y1-pad):min(h, y1+pad), x1:x2].mean(axis=0) if y1 < y2 else None
    hx_bottom = g[max(0, y2-pad):min(h, y2+pad), x1:x2].mean(axis=0) if y1 < y2 else None

    # snap y1 / y2 to the brightest rows
    if vx_left is not None and len(vx_left) > 0:
        j = _find_gutter_pos(vx_left, 0, min(max_shift, len(vx_left)//3), 'bright')
        y1 = y1 + j
        j = _find_gutter_pos(vx_left, len(vx_left)-1, min(max_shift, len(vx_left)//3), 'bright')
        y2 = y1 + j

    # snap x1 / x2 to the brightest cols
    if hx_top is not None and len(hx_top) > 0:
        i = _find_gutter_pos(hx_top, 0, min(max_shift, len(hx_top)//3), 'bright')
        x1 = x1 + i
    if hx_bottom is not None and len(hx_bottom) > 0:
        i = _find_gutter_pos(hx_bottom, len(hx_bottom)-1, min(max_shift, len(hx_bottom)//3), 'bright')
        x2 = x1 + i

    if x2 <= x1 or y2 <= y1:
        return rect  # fallback
    return QRectF(float(x1), float(y1), float(x2-x1), float(y2-y1))

def split_by_internal_gutters(rgb: np.ndarray, rect: QRectF) -> list[QRectF]:
    """
    If a large rect likely contains multiple panels, split along vertical bright bands.
    More conservative approach with harder conditions.
    """
    h, w = rgb.shape[:2]
    x1, y1 = int(rect.left()), int(rect.top())
    x2, y2 = int(rect.right()), int(rect.bottom())
    
    # Safety checks
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return [rect]
        
    roi = rgb[y1:y2, x1:x2]
    L = lab_L(roi)  # Use LAB L channel with CLAHE
    H, W = L.shape[:2]
    
    # Bandes/gouttières doivent être franchement larges
    min_band_ratio = 0.04
    min_gap_px = max(20, int(0.02 * W))

    vproj = L.mean(axis=0)
    # Exiger des bandes très claires (évite neige/ciel)
    thr = max(215, vproj.mean() + 0.60 * vproj.std())
    mask = vproj > thr
    spans = find_runs(mask, int(min_band_ratio * W))

    # Filtrer par faible densité d'arêtes (vraies gouttières = peu d'arêtes)
    edges = cv2.Canny(L, 60, 180)
    def edge_density(x1_span, x2_span):
        sl = edges[:, max(0, x1_span):min(W, x2_span)]
        return float((sl > 0).mean())
    
    spans = [(x1_span, x2_span) for (x1_span, x2_span) in spans if edge_density(x1_span, x2_span) < 0.05]

    if not spans:
        return [rect]

    # Créer les découpes
    splits = [0]
    for x1_span, x2_span in spans:
        # Couper au milieu de la bande claire
        splits.append((x1_span + x2_span) // 2)
    splits.append(W)
    splits = sorted(set(splits))
    
    pieces = []
    for a, b in zip(splits, splits[1:]):
        if b - a > 0.15 * W:  # Ignorer les tranches trop fines
            pieces.append(QRectF(float(x1 + a), float(y1), float(b - a), float(y2 - y1)))

    return pieces if len(pieces) > 1 else [rect]

def lab_L(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)

def find_runs(bool_arr, min_len: int):
    runs, start = [], None
    for i, v in enumerate(list(bool_arr) + [False]):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    return runs

def _closest_center(runs, target: int):
    if not runs:
        return target
    return min(((abs(((a+b)//2) - target), (a+b)//2) for a, b in runs))[1]

def snap_rect_to_gutters(
    L_img: np.ndarray,
    rect: QRectF,
    page_size: QSizeF,
    px_per_pt: float,
    min_gutter_px: int = 8,
    cov: float = 0.90,
    max_expand_frac: float = 0.04,
    smooth_k: int = 17,
) -> QRectF:
    H, W = L_img.shape[:2]
    x0 = max(0, int(round(rect.left()   * px_per_pt)))
    y0 = max(0, int(round(rect.top()    * px_per_pt)))
    ww = max(1, int(round(rect.width()  * px_per_pt)))
    hh = max(1, int(round(rect.height() * px_per_pt)))

    pad = int(max_expand_frac * min(W, H))
    xa = max(0, x0 - pad); ya = max(0, y0 - pad)
    xb = min(W, x0 + ww + pad); yb = min(H, y0 + hh + pad)
    roi = L_img[ya:yb, xa:xb].astype(np.float32) / 255.0
    if roi.size == 0:
        return rect

    col = roi.mean(axis=0); row = roi.mean(axis=1)
    k = max(9, smooth_k)
    if col.size >= k: col = np.convolve(col, np.ones(k) / k, mode="same")
    if row.size >= k: row = np.convolve(row, np.ones(k) / k, mode="same")

    ct = (col.max() - col.min()) * 0.15 + col.mean()
    rt = (row.max() - row.min()) * 0.15 + row.mean()
    vbin = col >= ct; hbin = row >= rt

    v_runs = find_runs(vbin, min_gutter_px)
    h_runs = find_runs(hbin, min_gutter_px)

    Lx = x0 - xa; Rx = x0 + ww - xa
    Ty = y0 - ya; By = y0 + hh - ya

    newL = _closest_center(v_runs, Lx)
    newR = _closest_center(v_runs, Rx)
    newT = _closest_center(h_runs, Ty)
    newB = _closest_center(h_runs, By)

    nx = (xa + min(newL, newR)) / px_per_pt
    ny = (ya + min(newT, newB)) / px_per_pt
    nw = max(1.0 / px_per_pt, abs((xa + newR) - (xa + newL)) / px_per_pt)
    nh = max(1.0 / px_per_pt, abs((ya + newB) - (ya + newT)) / px_per_pt)
    return QRectF(nx, ny, nw, nh)

def snap_rect_to_gutters_rgb(
    rgb_img: np.ndarray,
    rect: QRectF,
    page_size: QSizeF,
    px_per_pt: float,
    min_gutter_px: int = 8,
    cov: float = 0.90,
    max_expand_frac: float = 0.04,
    smooth_k: int = 17,
) -> QRectF:
    """Version qui accepte une image RGB et fait la conversion LAB en interne."""
    L_img = lab_L(rgb_img)
    return snap_rect_to_gutters(L_img, rect, page_size, px_per_pt, min_gutter_px, cov, max_expand_frac, smooth_k)
