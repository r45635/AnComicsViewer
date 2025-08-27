
#!/usr/bin/env python3
"""
AnComicsViewer MINI
- PySide6 viewer for PDFs with YOLO panel/balloon overlays
- Robust overlay alignment: detection image == displayed image
- Overlays are children of the pixmap so they follow zoom/scroll perfectly
"""

import sys
import os
import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt, QRectF, QSizeF, QPointF
from PySide6.QtGui import QAction, QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsSimpleTextItem, QToolBar, QWidget, QVBoxLayout, QLabel, QStatusBar
)

# PDF render
try:
    import fitz  # PyMuPDF
except Exception as e:
    fitz = None

# Ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None


# ------------------ utilities ------------------
def qimage_to_rgb(qimg: QImage) -> np.ndarray:
    """Convert QImage -> numpy RGB (H, W, 3)"""
    if qimg.format() != QImage.Format.Format_RGBA8888:
        qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    h, w = qimg.height(), qimg.width()
    bpl = qimg.bytesPerLine()
    # Fix: constBits() returns memoryview in recent PySide6, convert to bytes
    buf = bytes(qimg.constBits())[:h * bpl]
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpl)[:, : w * 4]
    rgba = arr.reshape(h, w, 4)
    rgb = rgba[:, :, :3].copy()
    return rgb


@dataclass
class Detection:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


# ------------------ Graphics Items ------------------
class BBoxItem(QGraphicsRectItem):
    def __init__(self, rect: QRectF, label: str, color: QColor, parent=None):
        super().__init__(rect, parent)
        pen = QPen(color)
        pen.setCosmetic(True)  # constant 1px on screen regardless of zoom
        pen.setWidthF(1.5)
        self.setPen(pen)
        self.setBrush(Qt.NoBrush)
        self.setZValue(10)

        self._label_item = QGraphicsSimpleTextItem(label, self)
        self._label_item.setBrush(color)
        self._label_item.setFlag(QGraphicsSimpleTextItem.ItemIgnoresTransformations, True)
        # Pin label to top-left of rect
        self._label_item.setPos(rect.left() + 2, rect.top() + 2)


# ------------------ Viewer ------------------
class PdfYoloViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnComicsViewer MINI")
        self.resize(1100, 800)

        # Scene & View
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # Root widget
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self.setCentralWidget(central)

        # Status
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)

        # Pixmap & current page image
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.qimage_current: Optional[QImage] = None
        self.pdf = None
        self.page_index = 0
        self.base_scale = 1.0

        # YOLO
        self.model = None
        self.class_names = ["panel", "balloon"]  # default if model has no names
        self.conf_thres = 0.15
        self.iou_thres = 0.6
        self.max_det = 200
        self.show_panels = True
        self.show_balloons = True

        self._build_ui()
        self._auto_load_model()  # Charger automatiquement le modèle
        self._auto_load_last_pdf()  # Charger automatiquement le dernier PDF

    # ---------- Auto-loading ----------
    def _auto_load_model(self):
        """Charge automatiquement le modèle anComicsViewer_v01.pt s'il est présent."""
        if YOLO is None:
            self.status.showMessage("⚠️ Ultralytics YOLO non disponible - Détection désactivée")
            return
            
        # Charger en priorité le modèle anComicsViewer custom
        preferred_model = "anComicsViewer_v01.pt"
        
        if os.path.exists(preferred_model):
            try:
                self._load_model_from_path(preferred_model)
                self.status.showMessage(f"✅ Modèle BD custom chargé: {preferred_model}")
                return
            except Exception as e:
                print(f"Erreur chargement {preferred_model}: {e}")
        
        # Fallback: aucun modèle disponible
        self.status.showMessage("⚠️ Modèle anComicsViewer_v01.pt non trouvé - Utilisez 'Charger Modèle' dans le menu")
        self.model_status_action.setText("🔴 Aucun modèle")
        print("💡 Placez votre modèle 'anComicsViewer_v01.pt' dans ce répertoire pour un chargement automatique")

    # ---------- UI ----------
    def _build_ui(self):
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        act_open_pdf = QAction("Ouvrir PDF…", self)
        act_open_pdf.triggered.connect(self.open_pdf)
        tb.addAction(act_open_pdf)

        act_load_model = QAction("Charger modèle…", self)
        act_load_model.triggered.connect(self.load_model)
        tb.addAction(act_load_model)
        
        # Indicateur de statut du modèle
        self.model_status_action = QAction("🔴 Aucun modèle", self)
        self.model_status_action.setEnabled(False)  # Non cliquable, juste informatif
        tb.addAction(self.model_status_action)

        tb.addSeparator()

        act_prev = QAction("◀", self); act_prev.triggered.connect(self.prev_page); tb.addAction(act_prev)
        act_next = QAction("▶", self); act_next.triggered.connect(self.next_page); tb.addAction(act_next)

        tb.addSeparator()

        act_zoom_in = QAction("Zoom +", self); act_zoom_in.triggered.connect(lambda: self.zoom(1.2)); tb.addAction(act_zoom_in)
        act_zoom_out = QAction("Zoom −", self); act_zoom_out.triggered.connect(lambda: self.zoom(1/1.2)); tb.addAction(act_zoom_out)
        act_reset = QAction("Reset", self); act_reset.triggered.connect(self.reset_zoom); tb.addAction(act_reset)

        tb.addSeparator()

        act_toggle_panels = QAction("Panels", self); act_toggle_panels.setCheckable(True); act_toggle_panels.setChecked(True)
        act_toggle_panels.triggered.connect(self.toggle_panels); tb.addAction(act_toggle_panels)

        act_toggle_balloons = QAction("Balloons", self); act_toggle_balloons.setCheckable(True); act_toggle_balloons.setChecked(True)
        act_toggle_balloons.triggered.connect(self.toggle_balloons); tb.addAction(act_toggle_balloons)

    # ---------- PDF handling ----------
    def open_pdf(self):
        if fitz is None:
            QMessageBox.critical(self, "PyMuPDF manquant", "Installe PyMuPDF (pip install pymupdf)")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choisir un PDF", "", "PDF (*.pdf)")
        if not path:
            return
        try:
            self.pdf = fitz.open(path)
            self.page_index = 0
            self.status.showMessage(f"PDF ouvert: {os.path.basename(path)}  •  {len(self.pdf)} pages")
            self.load_page(self.page_index)
            self._save_session()  # Sauvegarder la session
        except Exception as e:
            QMessageBox.critical(self, "Erreur PDF", str(e))

    def load_page(self, index: int):
        if not self.pdf:
            return
        if index < 0 or index >= len(self.pdf):
            return
        self.page_index = index
        page = self.pdf[index]

        # Render at fixed DPI; zoom is handled by QGraphicsView transforms
        dpi = 200
        zoom = dpi / 72.0  # 72 dpi base
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
        qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888).copy()

        self.set_page_image(qimg)
        self.run_detection_and_overlay()
        self._save_session()  # Sauvegarder la session après changement de page

    def next_page(self):
        if self.pdf and self.page_index + 1 < len(self.pdf):
            self.load_page(self.page_index + 1)

    def prev_page(self):
        if self.pdf and self.page_index - 1 >= 0:
            self.load_page(self.page_index - 1)

    # ---------- Image & overlays ----------
    def set_page_image(self, qimg: QImage):
        """Set the displayed page image; overlays will be rebuilt afterward."""
        self.scene.clear()
        self.qimage_current = qimg

        pix = QPixmap.fromImage(qimg)
        pix.setDevicePixelRatio(1.0)  # neutralize HiDPI
        self.pixmap_item = QGraphicsPixmapItem(pix)
        self.pixmap_item.setZValue(0)
        self.scene.addItem(self.pixmap_item)

        self.reset_zoom()

    def clear_overlays(self):
        if not self.pixmap_item:
            return
        for ch in list(self.pixmap_item.childItems()):
            self.scene.removeItem(ch)

    def draw_detections(self, dets: List[Detection]):
        if not self.pixmap_item:
            return
        self.clear_overlays()

        # Choose colors per class
        def color_for_cls(c: int) -> QColor:
            if c == 0:   # panel
                return QColor(35, 197, 83)   # green
            elif c == 1: # balloon
                return QColor(41, 121, 255)  # blue
            else:
                return QColor(255, 160, 0)   # orange

        for d in dets:
            if d.cls == 0 and not self.show_panels:
                continue
            if d.cls == 1 and not self.show_balloons:
                continue
            x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            label = f"{self.class_names[d.cls] if d.cls < len(self.class_names) else d.cls} {d.conf:.2f}"
            BBoxItem(rect, label, color_for_cls(d.cls), parent=self.pixmap_item)

    # ---------- YOLO ----------
    def _load_model_from_path(self, model_path: str):
        """Charge un modèle YOLO depuis un chemin spécifique."""
        # patch torch.load weights_only if needed
        import torch
        orig = torch.load
        def patched_load(f, map_location=None, pickle_module=None, weights_only=None, **kw):
            if weights_only is None:
                weights_only = False
            return orig(f, map_location=map_location, pickle_module=pickle_module, weights_only=weights_only, **kw)
        torch.load = patched_load
        try:
            self.model = YOLO(model_path)
        finally:
            torch.load = orig

        # read class names if available
        try:
            names = self.model.model.names if hasattr(self.model, 'model') else None
            if names and isinstance(names, dict):
                # convert dict to list
                idx_max = max(names.keys())
                all_class_names = [names.get(i, f"class_{i}") for i in range(idx_max + 1)]
                
                # Pour les modèles COCO standard, on garde seulement les classes pertinentes pour BD
                if len(all_class_names) >= 80:  # Modèle COCO standard
                    # Garder seulement les classes utiles pour les BD
                    useful_classes = ["person", "book", "chair", "couch", "tv", "laptop"]
                    self.class_names = useful_classes
                    print(f"📚 Modèle COCO détecté - Classes filtrées pour BD: {useful_classes}")
                elif len(all_class_names) <= 5 and any(cls in ["panel", "balloon"] for cls in all_class_names):
                    # Votre modèle entraîné spécialement pour les BD !
                    self.class_names = all_class_names
                    print(f"🎯 Modèle BD CUSTOM détecté - Classes: {self.class_names}")
                else:
                    # Autre modèle custom
                    self.class_names = all_class_names
                    print(f"🔧 Modèle custom - Classes: {self.class_names}")
            elif isinstance(names, list):
                self.class_names = names
                print(f"📋 Classes du modèle: {self.class_names}")
        except Exception:
            print("⚠️ Impossible de lire les classes du modèle, utilisation des classes par défaut")
            pass

        # Mettre à jour le titre de la fenêtre pour montrer le modèle chargé
        model_name = os.path.basename(model_path)
        self.setWindowTitle(f"AnComicsViewer MINI - 🤖 {model_name}")
        
        # Mettre à jour l'indicateur dans la toolbar
        self.model_status_action.setText(f"🟢 {model_name}")
        
        # Message de confirmation plus concis
        print(f"✅ Modèle chargé: {model_name}")
        print(f"🎯 Classes utilisées: {len(self.class_names)} classes")

    # ---------- PDF Session Persistence ----------
    def _get_config_file(self) -> str:
        """Retourne le chemin du fichier de configuration."""
        return os.path.expanduser("~/.ancomicsviewer_session.json")
    
    def _save_session(self):
        """Sauvegarde la session actuelle (PDF et page)."""
        if not self.pdf:
            return
        
        try:
            config = {
                "last_pdf": self.pdf.name if hasattr(self.pdf, 'name') else None,
                "last_page": self.page_index,
                "timestamp": __import__('time').time()
            }
            
            with open(self._get_config_file(), 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            print(f"Erreur sauvegarde session: {e}")
    
    def _load_session(self) -> dict:
        """Charge la session sauvegardée."""
        try:
            config_file = self._get_config_file()
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Erreur chargement session: {e}")
        return {}
    
    def _auto_load_last_pdf(self):
        """Charge automatiquement le dernier PDF ouvert s'il existe toujours."""
        if fitz is None:
            return
            
        session = self._load_session()
        last_pdf = session.get('last_pdf')
        
        if not last_pdf or not os.path.exists(last_pdf):
            return
            
        try:
            # Vérifier si le fichier existe toujours et est accessible
            if os.path.isfile(last_pdf) and os.access(last_pdf, os.R_OK):
                print(f"🔄 Rechargement automatique: {os.path.basename(last_pdf)}")
                self.pdf = fitz.open(last_pdf)
                
                # Restaurer la page si valide
                last_page = session.get('last_page', 0)
                if 0 <= last_page < len(self.pdf):
                    self.page_index = last_page
                else:
                    self.page_index = 0
                
                self.status.showMessage(f"📖 PDF rechargé: {os.path.basename(last_pdf)}  •  Page {self.page_index + 1}/{len(self.pdf)}")
                self.load_page(self.page_index)
                
        except Exception as e:
            print(f"Erreur rechargement automatique: {e}")
            # Nettoyer la session si le fichier est corrompu
            try:
                os.remove(self._get_config_file())
            except:
                pass

    def load_model(self):
        if YOLO is None:
            QMessageBox.critical(self, "Ultralytics manquant", "Installe ultralytics (pip install ultralytics)")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choisir un modèle YOLO", "", "PT (*.pt)")
        if not path:
            return
        try:
            self._load_model_from_path(path)
            self.status.showMessage(f"✅ Modèle chargé: {os.path.basename(path)} • classes={self.class_names}")
            # Re-run detection on current page if any
            if self.qimage_current is not None:
                self.run_detection_and_overlay()
        except Exception as e:
            QMessageBox.critical(self, "Erreur modèle", str(e))
            self.setWindowTitle("AnComicsViewer MINI - ❌ Erreur modèle")
            self.model_status_action.setText("🔴 Erreur modèle")

    def run_detection_and_overlay(self):
        """Run detection on exactly the same QImage that is displayed, and draw boxes."""
        if self.qimage_current is None or self.model is None:
            return
        rgb = qimage_to_rgb(self.qimage_current)
        try:
            res = self.model.predict(
                source=rgb,
                imgsz=max(640, max(rgb.shape[0], rgb.shape[1]) // 2),
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=self.max_det,
                verbose=False,
                device="mps" if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "1") else None
            )[0]

            dets: List[Detection] = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                conf = res.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                    dets.append(Detection(cls=c, conf=float(p), x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)))

            self.draw_detections(dets)
            self.status.showMessage(f"Page {self.page_index+1}: {len(dets)} détections")
        except Exception as e:
            QMessageBox.critical(self, "Erreur d'inférence", str(e))

    # ---------- Zoom ----------
    def reset_zoom(self):
        self.view.resetTransform()
        self.base_scale = 1.0

    def zoom(self, factor: float):
        self.base_scale *= factor
        self.view.scale(factor, factor)

    # ---------- Toggles ----------
    def toggle_panels(self, checked: bool):
        self.show_panels = checked
        self.run_detection_and_overlay()

    def toggle_balloons(self, checked: bool):
        self.show_balloons = checked
        self.run_detection_and_overlay()


def main():
    # Neutralize HiDPI surprises (we keep DPR=1.0 on the pixmap)
    app = QApplication(sys.argv)
    w = PdfYoloViewer()
    w.show()
    sys.exit(app.exec())
    

if __name__ == "__main__":
    main()
