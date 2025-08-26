#!/usr/bin/env python3
"""
Test d'intégration AR avec un PDF réel
=====================================

Teste l'intégration complète des Architecture Requirements avec un PDF.
"""

import sys
from pathlib import Path

# Ajouter le chemin du projet
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PySide6 import QtCore, QtGui, QtWidgets, QtPdf
from src.ancomicsviewer.ar_integration import ARIntegrationMixin

class ARTestPDFViewer(QtWidgets.QMainWindow, ARIntegrationMixin):
    """Viewer PDF avec intégration AR pour tests."""
    
    def __init__(self):
        super().__init__()
        
        # AR-05: Configuration HiDPI
        QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, False)
        
        self.setWindowTitle("AR PDF Viewer Test")
        self.setGeometry(100, 100, 1400, 900)
        
        # Document PDF
        self.document = None
        self.current_page = 0
        
        # Initialiser le système AR
        self.init_ar_system()
        
        # Vue traditionnelle (simple)
        self.traditional_view = QtWidgets.QLabel("Pas de PDF chargé")
        self.traditional_view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.traditional_view)
        
        # Interface
        self._create_menus()
        self._create_toolbar()
        
        # Status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Prêt - Chargez un PDF et activez le mode AR")
    
    def _create_menus(self):
        """Crée les menus."""
        menubar = self.menuBar()
        
        # Menu File
        file_menu = menubar.addMenu("File")
        open_action = file_menu.addAction("Open PDF...")
        open_action.triggered.connect(self._open_pdf)
        
        # Menu AR (via mixin)
        self.ar_add_menu_actions(menubar)
    
    def _create_toolbar(self):
        """Crée la toolbar."""
        toolbar = self.addToolBar("Navigation")
        
        # Navigation pages
        prev_action = toolbar.addAction("← Prev")
        prev_action.triggered.connect(self._prev_page)
        
        self.page_label = QtWidgets.QLabel("Page: -/-")
        toolbar.addWidget(self.page_label)
        
        next_action = toolbar.addAction("Next →")
        next_action.triggered.connect(self._next_page)
        
        toolbar.addSeparator()
        
        # Actions AR
        ar_toggle = toolbar.addAction("AR Mode")
        ar_toggle.triggered.connect(self._ar_toggle_mode)
        
        debug_frame = toolbar.addAction("Debug")
        debug_frame.triggered.connect(self.ar_toggle_debug_frame)
    
    def _open_pdf(self):
        """Ouvre un fichier PDF."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Ouvrir PDF", "", "PDF Files (*.pdf)"
        )
        
        if file_path:
            self._load_pdf(file_path)
    
    def _load_pdf(self, file_path: str):
        """Charge un PDF."""
        try:
            self.document = QtPdf.QPdfDocument()
            self.document.load(file_path)
            
            if self.document.status() == QtPdf.QPdfDocument.Status.Ready:
                self.current_page = 0
                page_count = self.document.pageCount()
                
                self.status.showMessage(f"PDF chargé: {page_count} pages")
                self._update_page_label()
                
                # Si en mode AR, afficher la première page
                if self.ar_mode_enabled:
                    self.ar_render_and_detect_page(self.current_page)
                else:
                    self.traditional_view.setText(f"PDF chargé: {page_count} pages\nActivez le mode AR pour voir les détections")
                
                print(f"✅ PDF chargé: {Path(file_path).name}")
            else:
                self.status.showMessage("Erreur de chargement PDF")
                
        except Exception as e:
            self.status.showMessage(f"Erreur: {e}")
            print(f"❌ Erreur chargement PDF: {e}")
    
    def _update_page_label(self):
        """Met à jour le label de page."""
        if self.document:
            total = self.document.pageCount()
            self.page_label.setText(f"Page: {self.current_page + 1}/{total}")
        else:
            self.page_label.setText("Page: -/-")
    
    def _prev_page(self):
        """Page précédente."""
        if self.document and self.current_page > 0:
            self.current_page -= 1
            self._update_page_label()
            
            if self.ar_mode_enabled:
                self.ar_render_and_detect_page(self.current_page)
    
    def _next_page(self):
        """Page suivante."""
        if self.document and self.current_page < self.document.pageCount() - 1:
            self.current_page += 1
            self._update_page_label()
            
            if self.ar_mode_enabled:
                self.ar_render_and_detect_page(self.current_page)
    
    def enable_ar_mode(self):
        """Override pour mettre à jour l'affichage."""
        super().enable_ar_mode()
        
        if self.document:
            self.ar_render_and_detect_page(self.current_page)
            self.status.showMessage("Mode AR activé - Détection en cours...")
    
    def disable_ar_mode(self):
        """Override pour mettre à jour l'affichage."""
        super().disable_ar_mode()
        
        if self.document:
            total = self.document.pageCount()
            self.traditional_view.setText(f"Mode traditionnel\nPDF: {total} pages\nPage courante: {self.current_page + 1}")
        
        self.status.showMessage("Mode traditionnel restauré")

def main():
    """Point d'entrée du test."""
    app = QtWidgets.QApplication(sys.argv)
    
    print("🧪 Test d'intégration AR avec PDF réel")
    print("=" * 60)
    
    viewer = ARTestPDFViewer()
    viewer.show()
    
    print("📌 Instructions:")
    print("  1. Ouvrez un PDF (File -> Open PDF...)")
    print("  2. Activez le mode AR (AR Mode -> Toggle AR Mode)")
    print("  3. Naviguez entre les pages")
    print("  4. Testez le zoom/scroll - overlays doivent rester collés")
    print("  5. Utilisez 'Debug Frame' pour vérifier l'alignement")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
