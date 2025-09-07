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

# DEBUG RECIPE:
# Example runs:
# python main.py --config detect_strict.yaml --debug-detect --save-debug-overlays debug
# If some rows stay merged, try the gutter version:
# python main.py --config detect_grid_gutters.yaml --debug-detect --save-debug-overlays debug
# During tests, temporarily disable any 'full page before 1st' UI option.
# What we expect on a Tintin-like 3x3 grid page:
# panels_raw around 9-12 (a few extra small boxes are ok)
# panels_merged around 9
# balloons_raw > 0 (often 10-25)
# full_page_triggered=False

from __future__ import annotations
import sys, os, json, yaml, argparse, time, math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QAction, QImage, QPixmap, QPen, QColor, QKeySequence, QPainter, QShortcut
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
try:
    import cv2
except Exception:
    cv2 = None

# ---------------- global config ----------------
GLOBAL_CONFIG = {}
DEBUG_DETECT = False
DEBUG_OVERLAY_DIR = "debug"

def debug_detection_stats(step_name: str, panels: list, balloons: list, page_area: float = 0):
    """Affiche les statistiques de détection pour le debug"""
    global DEBUG_DETECT
    
    if not DEBUG_DETECT:
        return
        
    print(f"\n🔍 {step_name}:")
    print(f"   📦 Panels: {len(panels)} | 💬 Balloons: {len(balloons)}")
    
    if panels:
        areas = [_area(r) for (_, _, r) in panels]
        confs = [p for (_, p, _) in panels]
        print(f"   📏 Panel areas: min={min(areas):.0f}, max={max(areas):.0f}, avg={sum(areas)/len(areas):.0f}")
        print(f"   🎯 Panel confs: min={min(confs):.3f}, max={max(confs):.3f}, avg={sum(confs)/len(confs):.3f}")
        print(f"   📊 Panel area %: min={min(areas)/page_area*100:.2f}%, max={max(areas)/page_area*100:.2f}%")
    
    if balloons:
        areas = [_area(r) for (_, _, r) in balloons]
        confs = [p for (_, p, _) in balloons]
        print(f"   📏 Balloon areas: min={min(areas):.0f}, max={max(areas):.0f}, avg={sum(areas)/len(areas):.0f}")
        print(f"   🎯 Balloon confs: min={min(confs):.3f}, max={max(confs):.3f}, avg={sum(confs)/len(confs):.3f}")
        print(f"   📊 Balloon area %: min={min(areas)/page_area*100:.4f}%, max={max(areas)/page_area*100:.4f}%")


def save_debug_overlay(image, panels, balloons, filename, step_name):
    """Sauvegarde une image avec les overlays de debug"""
    global DEBUG_OVERLAY_DIR
    
    if not DEBUG_DETECT:
        return
        
    try:
        import cv2
        import numpy as np
        
        # Créer le dossier de debug s'il n'existe pas
        os.makedirs(DEBUG_OVERLAY_DIR, exist_ok=True)
        
        # Convertir QImage en numpy array
        if hasattr(image, 'width'):
            # QImage
            w, h = image.width(), image.height()
            ptr = image.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            # Numpy array
            img = image.copy()
        
        # Dessiner les panels en rouge
        for _, _, rect in panels:
            x1, y1 = int(rect.left()), int(rect.top())
            x2, y2 = int(rect.right()), int(rect.bottom())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Dessiner les balloons en bleu
        for _, _, rect in balloons:
            x1, y1 = int(rect.left()), int(rect.top())
            x2, y2 = int(rect.right()), int(rect.bottom())
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Sauvegarder l'image
        filepath = os.path.join(DEBUG_OVERLAY_DIR, f"{filename}_{step_name.replace(' ', '_').lower()}.png")
        cv2.imwrite(filepath, img)
        print(f"💾 Overlay sauvegardé: {filepath}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde de l'overlay {step_name}: {e}")


def run_rule_based_panel_detection(qimage):
    """Détection de panels basée sur des règles simples utilisant OpenCV - Version optimisée pour Golden City"""
    if cv2 is None:
        print("⚠️ OpenCV non disponible pour la détection par règles")
        return []

    try:
        # Convertir QImage en numpy array de manière robuste
        w, h = qimage.width(), qimage.height()
        format_type = qimage.format()

        if format_type == QImage.Format.Format_RGBA8888:
            ptr = qimage.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif format_type == QImage.Format.Format_RGB888:
            ptr = qimage.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 3)
            img = arr.copy()
        else:
            # Convertir vers RGB888 si nécessaire
            rgb_qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
            ptr = rgb_qimage.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 3)
            img = arr.copy()

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculer le contraste pour adapter les seuils de Canny
        contrast = gray.std()
        print(f"🔆 Contraste détecté: {contrast:.2f}")

        # Adapter les seuils de Canny selon le contraste
        if contrast < 65:  # Contraste faible (comme page 6)
            canny_min = 30
            canny_max = 90
            print("🎯 Utilisation seuils Canny faibles pour contraste faible")
        else:  # Contraste normal
            canny_min = 50
            canny_max = 150
            print("🎯 Utilisation seuils Canny normaux")

        # Détection des contours pour identifier les panels
        edges = cv2.Canny(gray, canny_min, canny_max)

        # Dilatation pour connecter les contours
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Trouver les contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrer les contours par taille - PARAMÈTRES EXACTS DE main_test_clean.py
        page_area = w * h
        min_area = int(page_area * 0.02)  # 2% de la page - MÊME QUE main_test_clean.py
        max_area = int(page_area * 0.8)   # 80% de la page - MÊME QUE main_test_clean.py

        potential_panels = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w_contour, h_contour = cv2.boundingRect(contour)
                aspect_ratio = w_contour / h_contour if h_contour > 0 else 0

                # Filtrer par ratio d'aspect - AJUSTÉ POUR LES PANELS TRÈS VERTICAUX
                if 0.2 < aspect_ratio < 5.0:  # Étendu pour les pages comme la page 6
                    potential_panels.append({
                        'x': x,
                        'y': y,
                        'w': w_contour,
                        'h': h_contour,
                        'area': area,
                        'center_x': x + w_contour/2,
                        'center_y': y + h_contour/2,
                        'aspect_ratio': aspect_ratio,
                        'source': 'rules'
                    })

        print(f"📦 Panels potentiels détectés par règles: {len(potential_panels)}")

        # Debug: Afficher les dimensions des panels détectés
        if potential_panels:
            print("   📏 Dimensions détectées:")
            for i, panel in enumerate(potential_panels):  # Afficher TOUS les panels
                print(f"      {i+1}: ({panel['x']:.0f},{panel['y']:.0f}) {panel['w']:.0f}x{panel['h']:.0f}")
                if i >= 10:  # Limiter à 10 pour éviter un log trop long
                    print(f"      ... et {len(potential_panels) - i - 1} autres")
                    break

        return potential_panels

    except Exception as e:
        print(f"❌ Erreur dans la détection par règles: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_rule_based_balloon_detection(qimage):
    """Détection de ballons basée sur des règles simples"""
    if cv2 is None:
        print("⚠️ OpenCV non disponible pour la détection par règles")
        return []

    try:
        # Convertir QImage en numpy array de manière robuste
        w, h = qimage.width(), qimage.height()
        format_type = qimage.format()

        if format_type == QImage.Format.Format_RGBA8888:
            ptr = qimage.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif format_type == QImage.Format.Format_RGB888:
            ptr = qimage.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 3)
            img = arr.copy()
        else:
            # Convertir vers RGB888 si nécessaire
            rgb_qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
            ptr = rgb_qimage.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 3)
            img = arr.copy()

        # Améliorer le contraste pour mieux détecter les ballons
        enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # Détection des contours avec paramètres adaptés aux ballons
        edges = cv2.Canny(gray, 30, 100)  # Paramètres plus sensibles

        # Dilatation plus légère pour les ballons
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Trouver les contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"🔍 Contours trouvés: {len(contours)} (avant filtrage ballons)")

        # Filtrer pour les ballons (plus petits que les panels)
        page_area = w * h
        min_area = int(page_area * 0.0005)  # 0.05% de la page (encore plus petit)
        max_area = int(page_area * 0.08)    # 8% de la page (plus permissif)

        potential_balloons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w_contour, h_contour = cv2.boundingRect(contour)
                aspect_ratio = w_contour / h_contour if h_contour > 0 else 0

                # Critères spécifiques aux ballons - plus permissifs
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                # Filtrer par circularité (ballons souvent plus circulaires) - plus permissif
                if 0.05 < aspect_ratio < 8.0 and circularity > 0.05:
                    potential_balloons.append({
                        'x': x,
                        'y': y,
                        'w': w_contour,
                        'h': h_contour,
                        'area': area,
                        'center_x': x + w_contour/2,
                        'center_y': y + h_contour/2,
                        'aspect_ratio': aspect_ratio,
                        'circularity': circularity,
                        'source': 'rules'
                    })

        print(f"💬 Ballons potentiels détectés par règles: {len(potential_balloons)}")
        return potential_balloons

    except Exception as e:
        print(f"❌ Erreur dans la détection de ballons par règles: {e}")
        return []


def merge_yolo_and_rules(yolo_results, rule_results, element_type):
    """Fusionner les résultats YOLO et par règles de manière intelligente"""
    merged_results = []

    # STRATÉGIE CORRIGÉE: Privilégier les panels par règles (plus précis)
    if element_type == "panel" and rule_results:
        print(f"🎯 FUSION HYBRIDE: YOLO={len(yolo_results)}, Règles={len(rule_results)}")

        # Étape 1: Ajouter TOUS les panels par règles (ils sont plus précis)
        for rule_panel in rule_results:
            rule_rect = QRectF(rule_panel['x'], rule_panel['y'], rule_panel['w'], rule_panel['h'])
            merged_results.append((0, 0.8, rule_rect))  # Confiance plus élevée pour les règles
            print(f"   ✅ Ajouté panel règles: ({rule_panel['x']:.0f},{rule_panel['y']:.0f}) {rule_panel['w']:.0f}x{rule_panel['h']:.0f}")

        # Étape 2: Ajouter SEULEMENT les panels YOLO qui ne se chevauchent PAS avec les règles
        # et qui pourraient être des panels supplémentaires
        yolo_added = 0
        for yolo_panel in yolo_results:
            yolo_rect = yolo_panel[2]
            overlap_with_rules = False

            for rule_panel in rule_results:
                rule_rect = QRectF(rule_panel['x'], rule_panel['y'], rule_panel['w'], rule_panel['h'])
                iou = calculate_iou_hybrid(yolo_panel, rule_panel)
                if iou > 0.1:  # Seuil plus strict pour éviter les faux positifs YOLO
                    overlap_with_rules = True
                    break

            if not overlap_with_rules:
                merged_results.append(yolo_panel)
                yolo_added += 1
                print(f"   ➕ Ajouté panel YOLO supplémentaire: ({yolo_rect.left():.0f},{yolo_rect.top():.0f}) {yolo_rect.width():.0f}x{yolo_rect.height():.0f}")

        print(f"   📊 Résultat fusion: {len(merged_results)} panels ({len(rule_results)} règles + {yolo_added} YOLO)")

    else:
        # Pour les ballons ou si pas de règles, garder l'ancienne logique
        merged_results.extend(yolo_results)

    return merged_results


def calculate_iou_hybrid(panel1, panel2):
    """Calculer l'IoU pour la fusion hybride"""
    # Gérer les deux formats possibles
    if isinstance(panel1, tuple) and len(panel1) == 3:
        # Format YOLO: (class, conf, QRectF)
        r1 = panel1[2]
        r2 = QRectF(panel2['x'], panel2['y'], panel2['w'], panel2['h'])
    elif isinstance(panel2, tuple) and len(panel2) == 3:
        # Format YOLO: (class, conf, QRectF)
        r1 = QRectF(panel1['x'], panel1['y'], panel1['w'], panel1['h'])
        r2 = panel2[2]
    else:
        # Les deux sont au format règles
        r1 = QRectF(panel1['x'], panel1['y'], panel1['w'], panel1['h'])
        r2 = QRectF(panel2['x'], panel2['y'], panel2['w'], panel2['h'])

    # Calculer l'intersection
    inter = r1.intersected(r2)
    if inter.isEmpty():
        return 0.0

    inter_area = inter.width() * inter.height()
    union_area = r1.width()*r1.height() + r2.width()*r2.height() - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def save_detection_data(data, filename, step_name):
    """Sauvegarde les données de détection au format JSON"""
    global DEBUG_OVERLAY_DIR
    
    if not DEBUG_DETECT:
        return
        
    try:
        import json
        
        # Créer le dossier de debug s'il n'existe pas
        os.makedirs(DEBUG_OVERLAY_DIR, exist_ok=True)
        
        # Convertir les données en format sérialisable
        serializable_data = []
        for item in data:
            c, p, r = item
            serializable_data.append({
                'class': int(c),
                'confidence': float(p),
                'bbox': {
                    'x1': float(r.left()),
                    'y1': float(r.top()),
                    'x2': float(r.right()),
                    'y2': float(r.bottom()),
                    'width': float(r.width()),
                    'height': float(r.height())
                }
            })
        
        # Sauvegarder en JSON
        filepath = os.path.join(DEBUG_OVERLAY_DIR, f"{filename}_{step_name.replace(' ', '_').lower()}.json")
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        print(f"💾 Données sauvegardées: {filepath}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde des données {step_name}: {e}")

def load_config(path: str) -> dict:
    """Charge la configuration depuis un fichier YAML"""
    config = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"⚠️  Erreur lors du chargement de la config {path}: {e}")
    return config

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
        self.setPen(pen); self.setBrush(Qt.BrushStyle.NoBrush); self.setZValue(10)
        if label:
            t = QGraphicsSimpleTextItem(label, self)
            t.setBrush(color)
            t.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            t.setPos(rect.left()+2, rect.top()+2)

class ClickableGraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load reading defaults from config/detect.yaml if present
        try:
            import yaml, os
            cfg_path = None
            for c in (os.path.join('config','detect.yaml'), 'detect.yaml'):
                if os.path.exists(c):
                    cfg_path = c; break
            if cfg_path:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    _cfgyaml = yaml.safe_load(f) or {}
                # Note: cfg will be set by parent window, this is just for reference
        except Exception:
            pass
    
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
    def __init__(self, initial_page=0):
        super().__init__()
        self.setWindowTitle("AnComicsViewer MINI")
        self.resize(1100, 800)

        self.scene = QGraphicsScene(self)
        self.view = ClickableGraphicsView(self.scene, self)
        self.view.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)  # keep content centered

        central = QWidget(self); lay = QVBoxLayout(central); lay.setContentsMargins(0,0,0,0); lay.addWidget(self.view)
        self.setCentralWidget(central)
        self.status = QStatusBar(self); self.setStatusBar(self.status)

        self.pdf = None; self.page_index = initial_page
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.qimage_current: Optional[QImage] = None
        self.model = None
        self.class_names = ["panel", "balloon"]
        self.conf_thres, self.iou_thres, self.max_det = 0.15, 0.6, 200
        self.show_panels, self.show_balloons = True, True
        self.cfg = AppConfig()
        
        # Balloon detection state
        self.balloon_detection_disabled = False
        self.balloon_class_index = 1  # Default to class index 1 for balloons

        # Detection mode selection
        self.detection_mode = "hybrid"  # Options: "yolo_only", "rules_only", "hybrid"

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
        self.a_bal = QAction("Balloons", self); self.a_bal.setCheckable(True); self.a_bal.setChecked(True); self.a_bal.toggled.connect(self._toggle_balloons); tb.addAction(self.a_bal)

        # Detection mode selector
        tb.addSeparator()
        from PySide6.QtWidgets import QComboBox
        self.detection_mode_combo = QComboBox(self)
        self.detection_mode_combo.addItem("🤖 YOLO seul", "yolo_only")
        self.detection_mode_combo.addItem("📏 Règles seules", "rules_only")
        self.detection_mode_combo.addItem("🎯 Hybride (Recommandé)", "hybrid")
        self.detection_mode_combo.setCurrentText("🎯 Hybride (Recommandé)")
        self.detection_mode_combo.currentTextChanged.connect(self._change_detection_mode)
        tb.addWidget(self.detection_mode_combo)

        # Reading mode shortcuts
        QShortcut(QKeySequence("Space"), self).activated.connect(self.next_step)
        QShortcut(QKeySequence("Shift+Space"), self).activated.connect(self.prev_step)
        QShortcut(QKeySequence("Return"), self).activated.connect(self.next_step)
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
        print(f"➡️ NEXT_PAGE: current_page={self.page_index}, pdf_pages={len(self.pdf) if self.pdf else 0}")
        if self.pdf and self.page_index + 1 < len(self.pdf):
            print(f"   ✅ Navigation vers page {self.page_index + 1}")
            self.load_page(self.page_index + 1)
        else:
            print(f"   ❌ Navigation impossible: {'pas de PDF' if not self.pdf else f'page_index={self.page_index}, total_pages={len(self.pdf)}'}")

    def prev_page(self):
        print(f"⬅️ PREV_PAGE: current_page={self.page_index}, pdf_pages={len(self.pdf) if self.pdf else 0}")
        if self.pdf and self.page_index - 1 >= 0:
            print(f"   ✅ Navigation vers page {self.page_index - 1}")
            self.load_page(self.page_index - 1)
        else:
            print(f"   ❌ Navigation impossible: {'pas de PDF' if not self.pdf else f'page_index={self.page_index}'}")

    # ---------- image & overlays ----------
    def _set_page_image(self, qimg: QImage):
        self.scene.clear(); self.qimage_current = qimg
        pix = QPixmap.fromImage(qimg); pix.setDevicePixelRatio(1.0)
        self.pixmap_item = QGraphicsPixmapItem(pix); self.pixmap_item.setZValue(0); self.scene.addItem(self.pixmap_item)
        self.reset_view()

    def _draw_detections(self, debug_tiles=None):
        if not hasattr(self, 'pixmap_item') or not self.pixmap_item: return
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

        # Priority 1: Try our trained model first
        trained_model = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/runs/detect/ancomics_final_optimized_v2/weights/best.pt"
        if os.path.exists(trained_model):
            try:
                self._load_model(trained_model);
                self.status.showMessage("✅ Trained model loaded automatically")
                return
            except Exception as e:
                print(f"⚠️  Failed to load trained model: {e}")

        # Priority 2: Try the best performing model: YOLOv8s (from experiment results)
        best_model = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/yolov8s.pt"
        if os.path.exists(best_model):
            try:
                self._load_model(best_model);
                self.status.showMessage("✅ Best model (YOLOv8s) loaded automatically")
                return
            except Exception: pass

        # Fallback: Try the original model with good panel detection
        original_model = "anComicsViewer_v01.pt"
        if os.path.exists(original_model):
            try:
                self._load_model(original_model)
                self.status.showMessage("✅ Original model loaded automatically")
                return
            except Exception:
                pass

        # Last fallback: Try enhanced model
        enhanced_model = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/multibd_enhanced_v2.pt"
        if os.path.exists(enhanced_model):
            try:
                self._load_model(enhanced_model)
                self.status.showMessage("✅ Enhanced model loaded automatically")
                return
            except Exception:
                pass

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
            print(f"MODEL_LOADED: weights={path} names={repr(getattr(self.model, 'names', None))}")
            
            # Check if "balloon" class is present in model names
            if hasattr(self.model, 'names') and self.model.names:
                if "balloon" not in self.model.names.values():
                    print("⚠️  WARNING: balloon class not found in model.names -> balloon detection disabled")
                    self.balloon_detection_disabled = True
                else:
                    self.balloon_detection_disabled = False
                    # Find the class index for balloons
                    self.balloon_class_index = None
                    for idx, name in self.model.names.items():
                        if name == "balloon":
                            self.balloon_class_index = idx
                            break
            else:
                # If no names available, assume balloon detection is possible
                self.balloon_detection_disabled = False
                self.balloon_class_index = 1  # Default assumption
                
        finally:
            torch.load = orig
        
        # Update model status with balloon detection info
        status_text = f"🟢 {os.path.basename(path)}"
        if self.balloon_detection_disabled:
            status_text += " (balloons disabled)"
            # Disable the balloon toggle in the UI
            if hasattr(self, 'a_bal'):
                self.a_bal.setEnabled(False)
                self.a_bal.setChecked(False)
                self.show_balloons = False
        else:
            # Re-enable balloon toggle if it was disabled
            if hasattr(self, 'a_bal'):
                self.a_bal.setEnabled(True)
                self.a_bal.setChecked(True)
                self.show_balloons = True
                
        self.model_status.setText(status_text)

    def _split_by_gutters(self, rect: QRectF) -> List[QRectF]:
        """
        Heuristic: if a rect covers multiple grid panels, split along bright gutters.
        - Works on grayscale integral image for speed.
        - Returns [rect] if no confident gutters are found.
        Tunables from YAML (with defaults):
          gutter_split_enable (bool), gutter_min_gap_px (int), gutter_min_contrast (int),
          gutter_min_coverage (float), gutter_dir ("both"|"h"|"v")
        """
        cfg = getattr(self, "_detect_cfg", {}) or {}
        if not bool(cfg.get("gutter_split_enable", True)): return [rect]
        if self.qimage_current is None: return [rect]
        W, H = self.qimage_current.width(), self.qimage_current.height()
        # Extract ROI as numpy (gray)
        qimg = self.qimage_current
        img = np.frombuffer(bytes(qimg.constBits()), dtype=np.uint8)
        img = img.reshape(qimg.height(), qimg.bytesPerLine())[:, : qimg.width()*3]
        img = img.reshape(H, W, 3).mean(axis=2).astype(np.float32)  # gray
        x0, y0 = int(rect.left()), int(rect.top())
        x1, y1 = int(rect.right()), int(rect.bottom())
        x0 = max(0, min(W-1, x0)); x1 = max(0, min(W-1, x1))
        y0 = max(0, min(H-1, y0)); y1 = max(0, min(H-1, y1))
        roi = img[y0:y1, x0:x1]
        if roi.size == 0: return [rect]
        # Params
        min_gap = int(cfg.get("gutter_min_gap_px", 6))
        min_contrast = float(cfg.get("gutter_min_contrast", 20))
        min_cov = float(cfg.get("gutter_min_coverage", 0.70))
        direction = str(cfg.get("gutter_dir", "both"))
        # find vertical gutters
        cuts_v = []
        if direction in ("both","v"):
            col = roi.mean(axis=0)
            thr = min(255.0, roi.mean() + min_contrast)
            bright = (col >= thr).astype(np.uint8)
            # contiguous bright runs
            s = 0
            for i in range(1, bright.size+1):
                if i==bright.size or bright[i]==0:
                    if bright[i-1]==1:
                        e=i
                        if (e-s)>=min_gap and (roi[:, s:e] > thr).mean() >= min_cov:
                            cuts_v.append((s, e))
                    s=i+1 if i<bright.size else i
                elif bright[i-1]==0:
                    s=i
        # find horizontal gutters
        cuts_h = []
        if direction in ("both","h"):
            row = roi.mean(axis=1)
            thr = min(255.0, roi.mean() + min_contrast)
            bright = (row >= thr).astype(np.uint8)
           # contiguous runs
            s = 0
            for i in range(1, bright.size+1):
                if i==bright.size or bright[i]==0:
                    if bright[i-1]==1:
                        e=i
                        if (e-s)>=min_gap and (roi[s:e, :] > thr).mean() >= min_cov:
                            cuts_h.append((s, e))
                    s=i+1 if i<bright.size else i
                elif bright[i-1]==0:
                    s=i
        if not cuts_v and not cuts_h:
            return [rect]
        # build grid cells from cuts
        xs = [0] + [c[0] for c in cuts_v] + [c[1] for c in cuts_v] + [roi.shape[1]]
        xs = sorted(set(xs))
        ys = [0] + [c[0] for c in cuts_h] + [c[1] for c in cuts_h] + [roi.shape[0]]
        ys = sorted(set(ys))
        cells = []
        for i in range(len(ys)-1):
            for j in range(len(xs)-1):
                rx0 = x0 + xs[j]; rx1 = x0 + xs[j+1]
                ry0 = y0 + ys[i]; ry1 = y0 + ys[i+1]
                w = max(1, rx1-rx0); h = max(1, ry1-ry0)
                if w*h >= 0.004 * (W*H):  # drop tiny slivers
                    cells.append(QRectF(rx0, ry0, w, h))
        return cells or [rect]

    def merge_boxes_iou(self, rects: List[QRectF], iou_thresh: float, dist_thresh: float, page_size: Tuple[float, float]) -> List[QRectF]:
        """
        Merge overlapping rectangles using IoU and distance thresholds.
        Returns a list of merged bounding rectangles.
        """
        if not rects:
            return []
        
        W, H = page_size
        max_dim = max(W, H)
        dist_thresh_px = dist_thresh * max_dim
        
        merged = []
        used = [False] * len(rects)
        
        for i, rect1 in enumerate(rects):
            if used[i]:
                continue
                
            # Start with the current rectangle
            current_rect = QRectF(rect1)
            used[i] = True
            
            # Check all other rectangles for merging
            changed = True
            while changed:
                changed = False
                for j, rect2 in enumerate(rects):
                    if used[j]:
                        continue
                    
                    # Calculate IoU
                    iou = _iou(current_rect, rect2)
                    
                    # Calculate center distance
                    dx = abs(current_rect.center().x() - rect2.center().x())
                    dy = abs(current_rect.center().y() - rect2.center().y())
                    dist = (dx**2 + dy**2)**0.5
                    
                    # Merge if IoU or distance criteria are met
                    if iou > iou_thresh or dist < dist_thresh_px:
                        current_rect = current_rect.united(rect2)
                        used[j] = True
                        changed = True
            
            merged.append(current_rect)
        
        return merged

    def _run_detection(self):
        """Improved detection pipeline with intelligent post-processing."""
        global DEBUG_DETECT, DEBUG_OVERLAY_DIR
        import numpy as np
        
        if self.qimage_current is None or self.model is None:
            self.dets = []
            self._draw_detections()
            return

        # --- Load YAML configuration ---
        config = GLOBAL_CONFIG.copy()  # Use global config loaded from CLI

        # Store config for gutter splitting
        self._detect_cfg = config

        # --- Extract configuration parameters ---
        PANEL_CONF = float(config.get('panel_conf', 0.18))
        PANEL_AREA_MIN_PCT = float(config.get('panel_area_min_pct', 0.02))
        BALLOON_CONF = float(config.get('balloon_conf', 0.22))
        BALLOON_AREA_MIN_PCT = float(config.get('balloon_area_min_pct', 0.0006))
        BALLOON_MIN_W = int(config.get('balloon_min_w', 30))
        BALLOON_MIN_H = int(config.get('balloon_min_h', 22))
        MAX_PANELS = int(config.get('max_panels', 20))
        MAX_BALLOONS = int(config.get('max_balloons', 12))
        MAX_DET = int(config.get('max_det', 400))
        IMGSZ_MAX = int(config.get('imgsz_max', 1536))
        TILE_TARGET = int(config.get('tile_target', 1024))
        TILE_OVERLAP = float(config.get('tile_overlap', 0.20))
        FORCE_TILING = bool(config.get('force_tiling', False))
        PAGE_MARGIN_INSET_PCT = float(config.get('page_margin_inset_pct', 0.02))

        # Merging parameters
        PANEL_MERGE_IOU = float(config.get('panel_merge_iou', 0.25))
        PANEL_MERGE_DIST = float(config.get('panel_merge_dist', 0.02))
        PANEL_CONTAINMENT_MERGE = float(config.get('panel_containment_merge', 0.55))
        ENABLE_PANEL_MERGE = bool(config.get('enable_panel_merge', True))
        PANEL_ROW_OVERLAP = float(config.get('panel_row_overlap', 0.35))
        PANEL_ROW_GAP_PCT = float(config.get('panel_row_gap_pct', 0.02))
        ENABLE_ROW_MERGE = bool(config.get('enable_row_merge', True))
        IOU_MERGE = float(config.get('iou_merge', 0.25))

        # Full-page detection
        FULL_PAGE_PANEL_PCT = float(config.get('full_page_panel_pct', 0.93))
        FULL_PAGE_KEEP_BALLOONS = bool(config.get('full_page_keep_balloons', True))
        FULL_PAGE_BALLOON_OVERLAP_PCT = float(config.get('full_page_balloon_overlap_pct', 0.12))

        # DEBUG: Afficher les paramètres utilisés
        print("\n🔧 PARAMÈTRES DE CONFIGURATION:")
        print(f"   🎯 Panel conf: {PANEL_CONF}, Balloon conf: {BALLOON_CONF}")
        print(f"   📏 Panel area min: {PANEL_AREA_MIN_PCT*100:.1f}%, Balloon area min: {BALLOON_AREA_MIN_PCT*100:.4f}%")
        print(f"   🔄 Merging: IoU={IOU_MERGE}, Dist={PANEL_MERGE_DIST}, Containment={PANEL_CONTAINMENT_MERGE}")
        print(f"   📊 Max: Panels={MAX_PANELS}, Balloons={MAX_BALLOONS}, Det={MAX_DET}")
        print(f"   🖼️  Image: imgsz_max={IMGSZ_MAX}, tile_target={TILE_TARGET}")

        # --- Prepare image ---
        qimg = self.qimage_current
        W, H = qimg.width(), qimg.height()
        PAGE_AREA = float(W * H)
        imgsz = min(IMGSZ_MAX, max(W, H))
        # Round imgsz to multiple of 32 to avoid stride warnings
        imgsz = ((imgsz + 31) // 32) * 32

        # --- Decide tiling strategy ---
        nx = ny = 1
        if (max(W, H) > TILE_TARGET * 1.15) or FORCE_TILING:
            nx = ny = 2 if max(W, H) < TILE_TARGET * 2.2 else 3

        # --- Helper functions for prediction ---
        def _to_rgb(q: QImage) -> np.ndarray:
            if q.format() != QImage.Format.Format_RGBA8888:
                q_ = q.convertToFormat(QImage.Format.Format_RGBA8888)
            else:
                q_ = q
            h, w = q_.height(), q_.width()
            buf = bytes(q_.constBits())
            arr = np.frombuffer(buf, dtype=np.uint8)[: h * q_.bytesPerLine()].reshape(h, q_.bytesPerLine())[:, : w * 4]
            return arr.reshape(h, w, 4)[:, :, :3].copy()

        def _predict_rgb(np_rgb):
            res = self.model.predict(
                source=np_rgb, imgsz=imgsz, conf=min(PANEL_CONF, BALLOON_CONF) if not self.balloon_detection_disabled else PANEL_CONF,
                iou=0.6, max_det=MAX_DET, augment=False, verbose=False,
                device="mps" if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "1") else None
            )[0]
            out = []
            if res and getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                    # Skip balloons if balloon detection is disabled
                    if self.balloon_detection_disabled and c == self.balloon_class_index:
                        continue
                    out.append((int(c), float(p), QRectF(float(x1), float(y1), float(x2-x1), float(y2-y1))))
            return out

        # --- Run detection (single pass or tiled) ---
        dets_raw = []
        if nx == 1 and ny == 1:
            dets_raw = _predict_rgb(_to_rgb(qimg))
            tiles_str = "1x1"
        else:
            dx, dy = W / nx, H / ny
            ox, oy = dx * TILE_OVERLAP, dy * TILE_OVERLAP
            for iy in range(ny):
                for ix in range(nx):
                    x1 = max(0, int(ix * dx - ox))
                    y1 = max(0, int(iy * dy - oy))
                    x2 = min(W, int((ix + 1) * dx + ox))
                    y2 = min(H, int((iy + 1) * dy + oy))
                    sub = qimg.copy(x1, y1, x2 - x1, y2 - y1)
                    preds = _predict_rgb(_to_rgb(sub))
                    # Remap to page coordinates
                    for (c, p, r) in preds:
                        r.translate(x1, y1)
                        dets_raw.append((c, p, r))
            tiles_str = f"{nx}x{ny}"

        # DEBUG: Statistiques après détection brute
        all_panels_raw = [(c, p, r) for (c, p, r) in dets_raw if c == 0]
        all_balloons_raw = [(c, p, r) for (c, p, r) in dets_raw if c == self.balloon_class_index] if not self.balloon_detection_disabled else []
        debug_detection_stats("DÉTECTION BRUTE", all_panels_raw, all_balloons_raw, PAGE_AREA)
        
        # Capture raw detections for debug summary
        raw_panels = [[r.left(), r.top(), r.right(), r.bottom(), p] for (_, p, r) in all_panels_raw]
        raw_balloons = [[r.left(), r.top(), r.right(), r.bottom(), p] for (_, p, r) in all_balloons_raw] if not self.balloon_detection_disabled else []
        
        # Sauvegarder les données brutes si debug activé
        page_name = f"page_{int(time.time())}"
        
        save_detection_data(all_panels_raw + all_balloons_raw, page_name, "raw_detection")
        save_debug_overlay(qimg, all_panels_raw, all_balloons_raw, page_name, "raw_detection")

        # --- 1. Initial cleaning and filtering ---
        # Separate panels and balloons
        all_panels = [(c, p, r) for (c, p, r) in dets_raw if c == 0]
        all_balloons = [(c, p, r) for (c, p, r) in dets_raw if c == self.balloon_class_index] if not self.balloon_detection_disabled else []

        # --- Detection mode selection ---
        print(f"\n🔄 MODE DE DÉTECTION: {self.detection_mode.upper()}")
        print("=" * 50)

        if self.detection_mode == "yolo_only":
            # Mode YOLO seul - utiliser seulement les résultats YOLO
            print("🤖 UTILISATION DU MODE YOLO SEUL")
            all_panels = [(c, p, r) for (c, p, r) in dets_raw if c == 0]
            all_balloons = [(c, p, r) for (c, p, r) in dets_raw if c == self.balloon_class_index] if not self.balloon_detection_disabled else []

        elif self.detection_mode == "rules_only":
            # Mode règles seules - utiliser seulement la détection par règles
            print("📏 UTILISATION DU MODE RÈGLES SEULES")
            all_panels = []
            all_balloons = []

            # Obtenir les panels par règles
            try:
                rule_panels = run_rule_based_panel_detection(qimg)
                if rule_panels:
                    # Convertir le format règles vers le format YOLO
                    for rule_panel in rule_panels:
                        rule_rect = QRectF(rule_panel['x'], rule_panel['y'], rule_panel['w'], rule_panel['h'])
                        all_panels.append((0, 0.5, rule_rect))  # Classe 0 pour panel, confiance moyenne
                    print(f"   📦 Panels par règles: {len(all_panels)}")
            except Exception as e:
                print(f"⚠️ Erreur dans la détection par règles: {e}")

            # Obtenir les ballons par règles si nécessaire
            if not self.balloon_detection_disabled:
                try:
                    rule_balloons = run_rule_based_balloon_detection(qimg)
                    if rule_balloons:
                        # Convertir le format règles vers le format YOLO
                        for rule_balloon in rule_balloons:
                            balloon_rect = QRectF(rule_balloon['x'], rule_balloon['y'], rule_balloon['w'], rule_balloon['h'])
                            all_balloons.append((self.balloon_class_index, 0.5, balloon_rect))
                        print(f"   💬 Ballons par règles: {len(all_balloons)}")
                except Exception as e:
                    print(f"⚠️ Erreur dans la détection de ballons par règles: {e}")

        else:  # hybrid (default)
            # --- HYBRID APPROACH: Combine YOLO with rule-based detection ---
            print("🎯 UTILISATION DU MODE HYBRIDE (YOLO + RÈGLES)")

            # DEBUG: Vérifier l'image utilisée
            print(f"📸 Image pour détection par règles: {qimg.width()}x{qimg.height()}, format={qimg.format()}")

            # Étape 1: Obtenir les résultats par règles pour les panels
            rule_panels = []
            try:
                rule_panels = run_rule_based_panel_detection(qimg)
                if rule_panels is None:
                    rule_panels = []
            except Exception as e:
                print(f"⚠️ Erreur dans la détection par règles: {e}")
                import traceback
                traceback.print_exc()
                rule_panels = []

            # DÉTECTION DE PAGE PLEINE (sans gouttières blanches)
            rule_panels_count = len(rule_panels)
            yolo_panels_count = len([(c,p,r) for c,p,r in dets_raw if c==0])
            is_full_page_layout = rule_panels_count == 0 and yolo_panels_count >= 3

            if is_full_page_layout:
                print(f"🖼️  PAGE PLEINE DÉTECTÉE - Règles: {rule_panels_count}, YOLO: {yolo_panels_count}")
                print("   Utilisation YOLO uniquement pour cette page pleine")
                # Pour les pages pleines, on garde tous les panels YOLO sans fusion
                all_panels = [(c, p, r) for (c, p, r) in dets_raw if c == 0]
            else:
                # Étape 2: Fusionner les panels YOLO avec ceux détectés par règles
                if rule_panels:
                    all_panels = merge_yolo_and_rules(all_panels, rule_panels, "panel")
                    print(f"\n🎯 RÉSULTATS HYBRIDES PANELS:")
                    print(f"   📦 Panels: YOLO={yolo_panels_count}, Règles={rule_panels_count}, Fusion={len(all_panels)}")

            # Étape 3: Détection par règles pour les ballons si nécessaire
            if not self.balloon_detection_disabled:
                # Vérifier le rappel des ballons YOLO
                balloon_matches = 0
                if all_balloons:
                    # Pour l'instant, on garde les ballons YOLO
                    # TODO: Implémenter la logique de vérification du rappel
                    pass

                if len(all_balloons) < 5:  # Si peu de ballons détectés, essayer les règles
                    print(f"\n🔍 TENTATIVE DE DÉTECTION BALLONS PAR RÈGLES (YOLO: {len(all_balloons)})")
                    rule_balloons = run_rule_based_balloon_detection(qimg)
                    if rule_balloons:
                        # Fusionner les ballons YOLO avec ceux détectés par règles
                        all_balloons = merge_yolo_and_rules(all_balloons, rule_balloons, "balloon")
                        print(f"   💬 Ballons: YOLO={len([(c,p,r) for c,p,r in dets_raw if c==self.balloon_class_index])}, Règles={len(rule_balloons)}, Fusion={len(all_balloons)}")

        # DEBUG: Statistiques après sélection du mode
        debug_detection_stats(f"APRÈS MODE {self.detection_mode.upper()}", all_panels, all_balloons, PAGE_AREA)

        # --- POST-PROCESSING SPÉCIFIQUE PAR MODE ---
        print(f"🔧 POST-PROCESSING POUR MODE {self.detection_mode.upper()}")

        # Vérifier si c'est une page pleine détectée
        is_detected_full_page = False
        if self.detection_mode == "hybrid":
            rule_panels_count = len([p for p in all_panels if p[1] == 0.5])  # Panels avec confiance 0.5 viennent des règles
            yolo_panels_count = len([p for p in all_panels if p[1] != 0.5])  # Panels avec autres confiances viennent de YOLO
            is_detected_full_page = rule_panels_count == 0 and yolo_panels_count >= 3

        if is_detected_full_page:
            # Pour les pages pleines détectées, garder tous les panels YOLO sans filtrage strict
            print("🖼️ POST-PROCESSING LAXISTE POUR PAGE PLEINE DÉTECTÉE")
            panels = all_panels
            balloons = all_balloons

        elif self.detection_mode == "rules_only":
            # Mode règles : post-processing minimal, les règles sont déjà fiables
            print("📏 POST-PROCESSING MINIMAL POUR RÈGLES")
            panels = all_panels  # Garder tous les panels détectés par règles
            balloons = all_balloons  # Garder tous les ballons détectés par règles

        elif self.detection_mode == "yolo_only":
            # Mode YOLO seul : filtrage permissif pour éviter de perdre des détections
            print("🤖 POST-PROCESSING PERMISSIF POUR YOLO")
            filtered_panels = []
            filtered_balloons = []

            for c, p, r in all_panels:
                area = _area(r)
                area_ratio = area / PAGE_AREA
                aspect_ratio = r.width() / r.height() if r.height() > 0 else 0

                # Seuils permissifs pour YOLO
                if (p >= 0.1 and  # Seuil confiance bas
                    area_ratio >= 0.005 and  # 0.5% minimum
                    area_ratio <= 0.95 and   # 95% maximum
                    0.1 <= aspect_ratio <= 10.0):  # Ratio d'aspect très permissif
                    filtered_panels.append((c, p, r))

            for c, p, r in all_balloons:
                area = _area(r)
                area_ratio = area / PAGE_AREA
                if p >= 0.1 and area_ratio >= 0.001 and area_ratio <= 0.5:  # Seuils permissifs
                    filtered_balloons.append((c, p, r))

            panels = filtered_panels
            balloons = filtered_balloons

        else:  # hybrid (default) - pages normales
            # --- POST-PROCESSING OPTIMISÉ POUR HYBRIDE ---
            print("🎯 POST-PROCESSING OPTIMISÉ POUR HYBRIDE")

            # Étape 1: Filtrage STRICT (mêmes seuils que le mode règles)
            filtered_panels = []
            filtered_balloons = []

            for c, p, r in all_panels:
                area = _area(r)
                area_ratio = area / PAGE_AREA
                aspect_ratio = r.width() / r.height() if r.height() > 0 else 0

                # MÊMES SEUILS QUE LE MODE RÈGLES pour garantir 7 panels
                if (p >= 0.25 and  # Seuil confiance strict comme config
                    area_ratio >= 0.02 and  # 2% minimum comme règles
                    area_ratio <= 0.8 and   # 80% maximum comme règles
                    0.3 <= aspect_ratio <= 3.0):  # Ratio d'aspect comme règles
                    filtered_panels.append((c, p, r))

            for c, p, r in all_balloons:
                area = _area(r)
                area_ratio = area / PAGE_AREA
                if p >= 0.20 and area_ratio >= 0.005 and area_ratio <= 0.3:  # Seuils de config
                    filtered_balloons.append((c, p, r))

            panels = filtered_panels
            balloons = filtered_balloons

        debug_detection_stats("APRÈS FILTRE STRICT", panels, balloons, PAGE_AREA)

        # Étape 2: Merging léger (seuils plus élevés pour éviter la fusion excessive)
        if ENABLE_PANEL_MERGE and len(panels) > 1:
            panel_rects = [r for (_, _, r) in panels]

            # IoU merging avec seuil plus élevé (moins de fusions)
            merged_rects = self.merge_boxes_iou(panel_rects, 0.7, PANEL_MERGE_DIST, (W, H))  # 0.7 au lieu de 0.5

            # Reconstruire les panels avec les rectangles mergés
            best_conf = max([p for (_, p, _) in panels], default=0.5)
            panels = [(0, best_conf, rr) for rr in merged_rects]

        debug_detection_stats("APRÈS MERGING LÉGER", panels, balloons, PAGE_AREA)

        # Étape 3: PAS de merging par rangées (trop destructeur pour hybride)
        # Étape 4: PAS de filtrage containment (trop destructeur pour hybride)
        # Étape 5: PAS de gutter splitting (trop destructeur pour hybride)

        # Étape 6: Détection page complète avec seuil très élevé
        if panels and len(panels) > 1:  # Seulement si plusieurs panels
            largest_panel = max(panels, key=lambda t: _area(t[2]))
            largest_area = _area(largest_panel[2])

            if largest_area / max(1e-6, PAGE_AREA) >= 0.95:  # Seuil très élevé
                print("🖼️ PANEL PAGE COMPLÈTE DÉTECTÉ - CONSERVATION")
                # Garder tous les panels mais marquer le plus grand
                pass
            else:
                print("📦 PANELS NORMAUX - AUCUNE FUSION")

        # Étape 7: Limitation douce du nombre de panels
        if len(panels) > 30:  # Limite plus haute
            panels.sort(key=lambda t: _area(t[2]), reverse=True)
            panels = panels[:30]
            print(f"✂️ LIMITÉ À 30 PANELS (était {len(panels) + (len(panels) - 30)})")

        debug_detection_stats("APRÈS LIMITATION DOUCE", panels, balloons, PAGE_AREA)
        
        # Sauvegarder les données finales après tous les traitements
        save_detection_data(panels + balloons, page_name, "final_result")
        save_debug_overlay(qimg, panels, balloons, page_name, "final_result")

        # --- 6. Final output ---
        self.dets = [Detection(c, r, p) for c, p, r in panels + balloons]
        
        # FORCE AFFICHAGE STATISTIQUES FINALES
        print(f"\n🎯 STATISTIQUES FINALES FORCÉES:")
        print(f"   📦 Panels finaux: {len(panels)}")
        print(f"   💬 Ballons finaux: {len(balloons)}")
        print(f"   🔢 Total détections: {len(self.dets)}")
        
        if panels:
            print("   📏 Panels détaillés:")
            for i, (c, p, r) in enumerate(panels[:5]):
                print(f"      {i+1}: ({r.left():.0f},{r.top():.0f}) {r.width():.0f}x{r.height():.0f} conf={p:.3f}")
        
        # Capture merged detections for debug summary
        merged_panels = [[r.left(), r.top(), r.right(), r.bottom()] for (_, _, r) in panels]
        merged_balloons = [[r.left(), r.top(), r.right(), r.bottom()] for (_, _, r) in balloons]
        
        # Debug summary if enabled
        if DEBUG_DETECT:
            full_page_flag = False
            if panels:
                largest_panel = max(panels, key=lambda t: _area(t[2]))
                largest_area = _area(largest_panel[2])
                if largest_area / max(1e-6, PAGE_AREA) >= FULL_PAGE_PANEL_PCT:
                    full_page_flag = True
            
            print("=== DEBUG SUMMARY ===")
            print(f"page={self.page_index} size={W}x{H} tiles={nx}x{ny} tile_target={TILE_TARGET} overlap={TILE_OVERLAP}")
            print(f"model.names={repr(getattr(self.model, 'names', None))}")
            print(f"counts: panels_raw={len(raw_panels)} panels_merged={len(merged_panels)} "
                  f"balloons_raw={len(raw_balloons)} balloons_merged={len(merged_balloons)}")
            print(f"full_page_triggered={full_page_flag}")
            print(f"first3_panels_raw={raw_panels[:3]}")
            print(f"first3_panels_merged={merged_panels[:3]}")
        
        # Save debug overlays if requested
        if DEBUG_OVERLAY_DIR and DEBUG_OVERLAY_DIR != "debug":
            try:
                import cv2
                import numpy as np
                os.makedirs(DEBUG_OVERLAY_DIR, exist_ok=True)
                
                # Convert QImage to numpy for OpenCV
                if hasattr(qimg, 'width'):
                    w, h = qimg.width(), qimg.height()
                    ptr = qimg.constBits()
                    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
                    img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                else:
                    img = qimg.copy()
                
                # Raw detection overlay
                img_raw = img.copy()
                for bbox in raw_panels:
                    x1, y1, x2, y2, conf = bbox
                    cv2.rectangle(img_raw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(img_raw, f"{conf:.2f}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                for bbox in raw_balloons:
                    x1, y1, x2, y2, conf = bbox
                    cv2.rectangle(img_raw, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(img_raw, f"{conf:.2f}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.imwrite(os.path.join(DEBUG_OVERLAY_DIR, f"page_{self.page_index}_raw.png"), img_raw)
                
                # Merged detection overlay
                img_merged = img.copy()
                for bbox in merged_panels:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(img_merged, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                for bbox in merged_balloons:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(img_merged, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                cv2.imwrite(os.path.join(DEBUG_OVERLAY_DIR, f"page_{self.page_index}_merged.png"), img_merged)
                
            except Exception as e:
                print(f"Warning: Could not save debug overlays: {e}")
        
        self._draw_detections()

        # --- 7. Update status bar ---
        cache_status = "hit"  # Could be enhanced to track actual caching
        balloon_status = f"balloons={len(balloons)}" if not self.balloon_detection_disabled else "balloons=DISABLED"
        if hasattr(self, "status") and self.status:
            self.status.showMessage(
                f"Page {self.page_index+1}: panels={len(panels)}, {balloon_status} "
                f"(imgsz={imgsz}, tiles={tiles_str}, cache={cache_status})"
            )

    # ---------- reading mode ----------
    def _prepare_reading_units(self):
        """Une unité = panel ∪ (balloons qui chevauchent le panel). Ballons restants = unités seules."""
        print("📖 PRÉPARATION DES UNITÉS DE LECTURE")
        self.read_units = []
        if not getattr(self, "dets", None):
            print("⚠️  Aucune détection trouvée")
            # Créer une unité de lecture pour toute la page si aucune détection
            if self.qimage_current is not None:
                full_page_rect = QRectF(0, 0, self.qimage_current.width(), self.qimage_current.height())
                self.read_units.append(full_page_rect)
                print(f"🖼️  Unité pleine page créée: {full_page_rect.width():.0f}x{full_page_rect.height():.0f}")
            self.read_index = -1
            return

        # séparer
        panels   = [r for (c,p,r) in [(d.cls, d.conf, d.rect) for d in self.dets] if c==0]
        balloons = [r for (c,p,r) in [(d.cls, d.conf, d.rect) for d in self.dets] if c==1]

        print(f"📦 Panels détectés: {len(panels)}")
        print(f"💬 Ballons détectés: {len(balloons)}")

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

        print(f"📚 Unités de lecture créées: {len(self.read_units)} (panels: {len(panels)}, ballons seuls: {len(balloons) - len(attached)})")

        
        # ordre de lecture (banded to keep near-aligned items together)
        if self.qimage_current is not None:
            if getattr(self, "cfg", None) and getattr(self.cfg, "direction", "LR_TB") == "LR_TB":
                # RÉDUCTION DU ROW_BAND pour mieux séparer les lignes
                ROW_BAND = max(8.0, 0.008 * self.qimage_current.height())  # De 0.02 → 0.008 (3x plus petit)
                print(f"📏 ROW_BAND: {ROW_BAND:.1f} pixels pour hauteur {self.qimage_current.height()}")

                rows = {}
                for i, r in enumerate(self.read_units):
                    key = round(r.top() / ROW_BAND)
                    rows.setdefault(key, []).append((i, r))
                    print(f"   Unité {i}: Y={r.top():.1f}, key={key} (ROW_BAND={ROW_BAND:.1f})")

                print(f"📊 Répartition par lignes: {len(rows)} lignes")
                for key, rects in sorted(rows.items()):
                    print(f"   Ligne {key}: {len(rects)} unités")
                    for idx, rect in rects:
                        print(f"      Unité {idx}: ({rect.left():.1f}, {rect.top():.1f}) {rect.width():.1f}x{rect.height():.1f}")

                ordered = []
                for key in sorted(rows.keys()):
                    ordered.extend(sorted([r for _, r in rows[key]], key=lambda rr: rr.left()))
                self.read_units = ordered

                print(f"📚 Total unités de lecture: {len(self.read_units)}")
                print("📚 Ordre final des unités:")
                for i, r in enumerate(self.read_units):
                    print(f"   {i}: ({r.left():.1f}, {r.top():.1f}) {r.width():.1f}x{r.height():.1f}")
            else:
                COL_BAND = max(12.0, 0.02 * self.qimage_current.width())
                cols = {}
                for r in self.read_units:
                    key = round(r.left() / COL_BAND)
                    cols.setdefault(key, []).append(r)
                ordered = []
                for key in sorted(cols.keys()):
                    ordered.extend(sorted(cols[key], key=lambda rr: rr.top()))
                self.read_units = ordered

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
        print(f"⬅️ PREV_STEP: reading_mode={self.cfg.reading_mode}, read_units={len(self.read_units) if hasattr(self, 'read_units') else 'N/A'}, read_index={self.read_index if hasattr(self, 'read_index') else 'N/A'}")
        if not self.cfg.reading_mode:
            print("   📖 Mode lecture désactivé, appel prev_page()")
            self.prev_page(); return
        if not self.read_units:
            print("   📖 Aucune unité de lecture, appel prev_page()")
            self.prev_page(); return
        if self.read_index <= 0:
            print(f"   📖 Début de page atteint (read_index={self.read_index})")
            self.read_index = -1
            # Si la page complète a déjà été montrée ou si on ne veut pas la montrer, aller à la page précédente
            if not self.cfg.show_full_page_before_first_panel or self.fullpage_shown_on_page:
                print("   📖 Navigation vers page précédente")
                self.prev_page()
            else:
                print("   📖 Affichage page complète avant premier panel")
                self.fit_full_page(); self.fullpage_shown_on_page = True
            return
        self.read_index -= 1
        print(f"   📖 Navigation vers unité {self.read_index}")
        self._focus_rect(self.read_units[self.read_index])

    def _focus_rect(self, r: QRectF):
        if not self.pixmap_item: return

        # DEBUG: Vérifier les coordonnées
        print(f"🔍 FOCUS_RECT - Rectangle demandé: ({r.left():.1f}, {r.top():.1f}) {r.width():.1f}x{r.height():.1f}")
        print(f"🔍 FOCUS_RECT - Pixmap bounds: ({self.pixmap_item.boundingRect().left():.1f}, {self.pixmap_item.boundingRect().top():.1f}) {self.pixmap_item.boundingRect().width():.1f}x{self.pixmap_item.boundingRect().height():.1f}")

        pad=0.08
        rr=QRectF(r); rr.adjust(-r.width()*pad,-r.height()*pad,r.width()*pad,r.height()*pad)
        rr = rr.intersected(self.pixmap_item.boundingRect())

        print(f"🔍 FOCUS_RECT - Rectangle après padding: ({rr.left():.1f}, {rr.top():.1f}) {rr.width():.1f}x{rr.height():.1f}")

        rect_scene = self.pixmap_item.mapRectToScene(rr)
        print(f"🔍 FOCUS_RECT - Rectangle dans la scène: ({rect_scene.left():.1f}, {rect_scene.top():.1f}) {rect_scene.width():.1f}x{rect_scene.height():.1f}")

        self.view.fitInView(rect_scene, Qt.AspectRatioMode.KeepAspectRatio)
        self.view.centerOn(rect_scene.center())

    def fit_full_page(self):
        if not self.pixmap_item: return
        self.view.fitInView(self.pixmap_item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.view.centerOn(self.pixmap_item.sceneBoundingRect().center())

    # ---------- zoom/reset ----------
    def reset_view(self):
        self.view.resetTransform()
        if self.pixmap_item:
            self.view.centerOn(self.pixmap_item.sceneBoundingRect().center())

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
            self.view.centerOn(self.pixmap_item.sceneBoundingRect().center())

    # ---------- toggles ----------
    def _toggle_panels(self, checked: bool):
        self.show_panels = checked; self._run_detection(); self._prepare_reading_units()

    def _toggle_balloons(self, checked: bool):
        if self.balloon_detection_disabled:
            # If balloon detection is disabled, don't allow toggling
            if hasattr(self, 'a_bal'):
                self.a_bal.setChecked(False)
            return
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

    def _change_detection_mode(self, text: str):
        """Change le mode de détection selon la sélection utilisateur"""
        mode_map = {
            "🤖 YOLO seul": "yolo_only",
            "📏 Règles seules": "rules_only",
            "🎯 Hybride (Recommandé)": "hybrid"
        }
        self.detection_mode = mode_map.get(text, "hybrid")
        print(f"🔄 Mode de détection changé: {self.detection_mode}")
        
        # Relancer la détection avec le nouveau mode
        if self.qimage_current is not None:
            self._run_detection()

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


def run_app():
    """Point d'entrée principal avec support CLI"""
    global GLOBAL_CONFIG
    
    parser = argparse.ArgumentParser(description="AnComicsViewer - Comic Book Reader")
    parser.add_argument('--config', type=str, default='detect_with_merge.yaml',
                       help='Path to detection configuration YAML file')
    parser.add_argument('--debug-detect', action='store_true',
                       help='Enable detection debug mode')
    parser.add_argument('--save-debug-overlays', type=str, default='debug',
                       help='Directory to save debug overlay images')
    parser.add_argument('--page', type=int, default=0,
                       help='Page number to load (0-based)')
    
    args = parser.parse_args()
    
    # Charger la configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        # Chercher dans le répertoire config/ si chemin relatif
        config_dir = os.path.join(os.path.dirname(__file__), 'config')
        candidate_path = os.path.join(config_dir, config_path)
        if os.path.exists(candidate_path):
            config_path = candidate_path
        else:
            # Chercher dans le répertoire courant
            if os.path.exists(config_path):
                config_path = os.path.abspath(config_path)
            else:
                print(f"⚠️  Configuration file not found: {args.config}")
                config_path = None
    
    if config_path:
        config = load_config(config_path)
        abs_path = os.path.abspath(config_path)
        keys = sorted(config.keys()) if config else []
        print(f"CONFIG_APPLIED: {abs_path} | keys={keys}")
        
        # Stocker la config globale pour l'utiliser dans l'application
        GLOBAL_CONFIG = config
        
        # Stocker les arguments de debug
        DEBUG_DETECT = args.debug_detect
        DEBUG_OVERLAY_DIR = args.save_debug_overlays if args.save_debug_overlays else "debug"
    else:
        GLOBAL_CONFIG = {}
        
        # Stocker les arguments de debug même sans config
        DEBUG_DETECT = args.debug_detect
        DEBUG_OVERLAY_DIR = args.save_debug_overlays if args.save_debug_overlays else "debug"
    
    # Lancer l'application
    app = QApplication(sys.argv)
    viewer = PdfYoloViewer(initial_page=args.page)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())

# ---------------- main ----------------

def main():
    app = QApplication(sys.argv)
    w = PdfYoloViewer(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
