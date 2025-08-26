#!/usr/bin/env python3
"""
Test des Architecture Requirements (AR-01 à AR-08)
================================================

Teste le nouveau système PageView + AdaptiveUltraRobustDetector
"""

import sys
import os
from pathlib import Path

# Ajouter le chemin du projet
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6 import QtCore, QtGui, QtWidgets
from src.ancomicsviewer.ui.page_view import PageView
from src.ancomicsviewer.detectors.adaptive_ultra_robust_detector import AdaptiveUltraRobustDetector

class ARTestWindow(QtWidgets.QMainWindow):
    """Fenêtre de test pour les Architecture Requirements."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AR Test - PageView + Detector")
        self.setGeometry(100, 100, 1200, 800)
        
        # AR-01: PageView avec overlays accrochés
        self.page_view = PageView()
        self.setCentralWidget(self.page_view)
        
        # Detector AR-compliant
        self.detector = AdaptiveUltraRobustDetector()
        
        # Toolbar pour tests
        self._create_toolbar()
        
        # Test avec une image simple
        self._create_test_image()
    
    def _create_toolbar(self):
        """Crée une toolbar de test."""
        toolbar = self.addToolBar("Tests")
        
        # Test image simple
        test_action = toolbar.addAction("Test Image")
        test_action.triggered.connect(self._test_simple_image)
        
        # Toggle debug frame
        debug_action = toolbar.addAction("Debug Frame")
        debug_action.triggered.connect(self._toggle_debug_frame)
        
        # Clear
        clear_action = toolbar.addAction("Clear")
        clear_action.triggered.connect(self.page_view.clear_page)
    
    def _create_test_image(self):
        """Crée une image de test avec quelques formes."""
        # Créer une QImage de test
        img = QtGui.QImage(800, 600, QtGui.QImage.Format.Format_RGB888)
        img.fill(QtGui.QColor(240, 240, 240))
        
        # Dessiner quelques rectangles pour simuler des panels
        painter = QtGui.QPainter(img)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 2))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        
        # Panel 1
        painter.drawRect(50, 50, 300, 200)
        # Panel 2  
        painter.drawRect(400, 50, 300, 200)
        # Panel 3 (plus grand)
        painter.drawRect(50, 300, 650, 250)
        
        painter.end()
        
        # AR-07: Logs de diagnostic
        print(f"[Debug] Created test QImage {img.width()}x{img.height()}")
        
        # Affichage et détection
        self._show_image_with_detection(img)
    
    def _show_image_with_detection(self, qimg: QtGui.QImage):
        """AR-04: Affiche l'image et lance la détection."""
        # AR-02: Même QImage pour affichage et détection
        self.page_view.show_qimage(qimg)
        
        # AR-07: Logs de diagnostic
        if self.page_view._page_item:
            pixmap = self.page_view._page_item.pixmap()
            dpr = pixmap.devicePixelRatio()
            print(f"[Debug] QImage {qimg.width()}x{qimg.height()}, pixmap {pixmap.width()}x{pixmap.height()}, DPR={dpr}")
        
        # Détection sur la même QImage
        dets = self.detector.detect_on_qimage(qimg)
        print(f"[Debug] dets: {len(dets)} first: {dets[0] if dets else None}")
        
        # AR-01: Dessiner les overlays accrochés au pixmap
        self.page_view.draw_detections(dets, show_fullframe_debug=False)
    
    def _test_simple_image(self):
        """Test avec l'image simple."""
        self._create_test_image()
    
    def _toggle_debug_frame(self):
        """AR-07: Active/désactive le cadre de debug."""
        # Recréer l'affichage avec debug frame
        if self.page_view._page_item:
            # Récupérer l'image courante (simulation)
            self._create_test_image()  # Pour l'instant, recrée l'image de test
            # TODO: Stocker l'image courante pour la réutiliser

def main():
    """Point d'entrée du test."""
    # AR-05: Configuration HiDPI
    QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, False)
    
    app = QtWidgets.QApplication(sys.argv)
    
    print("🧪 Test des Architecture Requirements (AR-01 à AR-08)")
    print("=" * 60)
    
    window = ARTestWindow()
    window.show()
    
    print("📌 Instructions de test:")
    print("  1. Vérifiez que l'image s'affiche correctement")
    print("  2. Zoomez/dézoomez - les overlays doivent rester collés")
    print("  3. Faites défiler - aucun lag d'alignement")
    print("  4. Testez 'Debug Frame' pour vérifier l'alignement exact")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
