#!/usr/bin/env python3
"""
Intégration AR (Architecture Requirements) dans l'application principale
========================================================================

Module pour intégrer progressivement les AR-01 à AR-08 dans main_app.py
"""

from typing import Optional, Dict, Any
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets

# Import des nouveaux composants AR
from .ui.page_view import PageView
from .detectors.adaptive_ultra_robust_detector import AdaptiveUltraRobustDetector
from .utils.qimage_utils import qimage_to_numpy

class ARIntegrationMixin:
    """
    ARIntegrationMixin - Intégration des Architecture Requirements dans l'app existante
    ===================================================================================

    Permet d'intégrer le système AR (PageView + AdaptiveUltraRobustDetector) 
    dans l'application principale ComicsView de manière transparente.

    Usage:
        class MyApp(QMainWindow, ARIntegrationMixin):
            def __init__(self):
                super().__init__()
                ARIntegrationMixin.__init__(self)
                
                # Toggle mode AR
                self.enable_ar_mode()  # ou disable_ar_mode()
    """

from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtPdf import QPdfDocument

try:
    from .ui.page_view import PageView
    from .detectors.adaptive_ultra_robust_detector import AdaptiveUltraRobustDetector
except ImportError:
    # Fallback si imports relatifs échouent
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from ui.page_view import PageView
    from detectors.adaptive_ultra_robust_detector import AdaptiveUltraRobustDetector
    
    def init_ar_system(self):
        """Initialise le système AR-compliant."""
        print("🔧 Initialisation du système AR-compliant...")
        
        # AR-01: PageView avec overlays accrochés
        self.ar_page_view: Optional[PageView] = None
        
        # AR-02/03: Détecteur adaptatif
        self.ar_detector: Optional[AdaptiveUltraRobustDetector] = None
        
        # Mode AR (peut être activé/désactivé)
        self.ar_mode_enabled = False
        
        # Cache pour la dernière QImage
        self.ar_current_qimage: Optional[QtGui.QImage] = None
        
        # Navigation AR
        self.ar_pdf_document: Optional[QPdfDocument] = None
        self.ar_current_page = 0
        self.ar_pdf_path = ""
        
        print("✅ Système AR initialisé")
    
    def _is_ar_enabled(self) -> bool:
        """Vérifie si le mode AR est activé (compatible avec différents noms d'attributs)."""
        return getattr(self, 'ar_mode_enabled', False) or getattr(self, '_ar_mode_enabled', False)
    
    def enable_ar_mode(self):
        """Active le mode AR avec PageView."""
        # Vérifier les différents noms d'attributs possibles
        ar_enabled = self._is_ar_enabled()
        
        if ar_enabled and hasattr(self, 'ar_page_view') and self.ar_page_view is not None:
            return
            
        print("🔄 Activation du mode AR...")
        
        # Créer le PageView
        self.ar_page_view = PageView()
        
        # Créer le détecteur adaptatif
        self.ar_detector = AdaptiveUltraRobustDetector()
        
        # Remplacer la vue actuelle (sauvegarde possible)
        if hasattr(self, 'view') and self.view:
            # Sauvegarder l'ancienne vue
            self.traditional_view = self.view
            
            # Remplacer par PageView
            if hasattr(self, 'setCentralWidget'):
                self.setCentralWidget(self.ar_page_view)
            
        # Marquer comme activé selon l'attribut disponible
        if hasattr(self, 'ar_mode_enabled'):
            self.ar_mode_enabled = True
        else:
            self._ar_mode_enabled = True
            
        print("✅ Mode AR activé - PageView opérationnel")
    
    def ar_load_and_render_pdf(self, pdf_path: str, page_num: int = 0) -> Optional[QtGui.QImage]:
        """Charge un PDF et rend une page avec détection AR."""
        if not self._is_ar_enabled() or not self.ar_page_view:
            print("❌ Mode AR non activé ou PageView manquant")
            return None
            
        try:
            print(f"📖 Chargement PDF: {pdf_path}")
            
            # Fermer le document précédent s'il existe
            if self.ar_pdf_document:
                self.ar_pdf_document.close()
            
            # Créer et charger le nouveau document
            self.ar_pdf_document = QPdfDocument()
            self.ar_pdf_document.load(pdf_path)
            
            if self.ar_pdf_document.status() != QPdfDocument.Status.Ready:
                print(f"❌ Impossible de charger le PDF: {pdf_path}")
                return None
                
            # Conserver les infos pour la navigation
            self.ar_pdf_path = pdf_path
            self.ar_current_page = page_num
                
            print(f"✅ PDF chargé - {self.ar_pdf_document.pageCount()} pages")
            
            # Rendre la page demandée
            return self.ar_render_page(page_num)
            
        except Exception as e:
            print(f"❌ Erreur rendu AR: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def ar_render_page(self, page_num: int) -> Optional[QtGui.QImage]:
        """Rend une page spécifique du PDF AR."""
        if not self._is_ar_enabled() or not self.ar_page_view or not self.ar_pdf_document:
            print("❌ Mode AR non activé ou pas de document")
            return None
            
        if page_num < 0 or page_num >= self.ar_pdf_document.pageCount():
            print(f"❌ Page {page_num} inexistante (max: {self.ar_pdf_document.pageCount()-1})")
            return None
        
        try:
            # AR-02: Rendre la page en QImage
            qimg = self._ar_render_page_to_qimage(self.ar_pdf_document, page_num, 200)
            if qimg.isNull():
                print(f"❌ Échec rendu page {page_num}")
                return None
                
            print(f"✅ Page {page_num} rendue: {qimg.width()}x{qimg.height()}")
            
            # Mettre à jour l'état
            self.ar_current_page = page_num
            self.ar_current_qimage = qimg
            
            # AR-04: Afficher la QImage dans PageView
            self.ar_page_view.show_qimage(qimg)
            
            # AR-04: Lancer la détection
            dets = self.ar_detector.detect_on_qimage(qimg)
            print(f"🔍 Détections: {len(dets)} panels")
            
            # AR-01: Dessiner les overlays (parented au pixmap)
            self.ar_page_view.draw_detections(dets, show_fullframe_debug=True)
            
            return qimg
            
        except Exception as e:
            print(f"❌ Erreur rendu page {page_num}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def ar_next_page(self) -> bool:
        """Aller à la page suivante."""
        if not self._is_ar_enabled() or not self.ar_pdf_document:
            return False
            
        next_page = self.ar_current_page + 1
        if next_page < self.ar_pdf_document.pageCount():
            return self.ar_render_page(next_page) is not None
        return False
    
    def ar_prev_page(self) -> bool:
        """Aller à la page précédente."""
        if not self._is_ar_enabled() or not self.ar_pdf_document:
            return False
            
        prev_page = self.ar_current_page - 1
        if prev_page >= 0:
            return self.ar_render_page(prev_page) is not None
        return False
    
    def ar_goto_page(self, page_num: int) -> bool:
        """Aller à une page spécifique."""
        if not self._is_ar_enabled() or not self.ar_pdf_document:
            return False
            
        return self.ar_render_page(page_num) is not None
    
    def disable_ar_mode(self):
        """Désactive le mode AR et restaure la vue traditionnelle."""
        if not self._is_ar_enabled():
            return
            
        print("🔄 Désactivation du mode AR...")
        
        # Restaurer la vue traditionnelle
        if hasattr(self, 'traditional_view') and self.traditional_view:
            if hasattr(self, 'setCentralWidget'):
                self.setCentralWidget(self.traditional_view)
        
        self.ar_mode_enabled = False
        print("✅ Mode traditionnel restauré")
    
    def ar_render_and_detect_page(self, page_num: int, dpi: float = 150) -> bool:
        """
        AR-02/04: Rend une page PDF en QImage et lance la détection.
        
        Args:
            page_num: Numéro de page (0-based)
            dpi: Résolution de rendu
            
        Returns:
            True si succès
        """
        if not self._is_ar_enabled() or not self.ar_page_view:
            return False
            
        # Récupérer le document
        doc = getattr(self, 'document', None)
        if not doc:
            print("❌ Pas de document PDF chargé")
            return False
            
        try:
            # AR-02: Rendre la page en QImage (même que l'affichage)
            qimg = self._ar_render_page_to_qimage(doc, page_num, dpi)
            if qimg.isNull():
                print(f"❌ Échec rendu page {page_num}")
                return False
                
            # Sauvegarder pour usage ultérieur
            self.ar_current_qimage = qimg
            
            # AR-04: Afficher la QImage dans PageView
            self.ar_page_view.show_qimage(qimg)
            
            # AR-07: Logs de diagnostic
            if self.ar_page_view._page_item:
                pixmap = self.ar_page_view._page_item.pixmap()
                dpr = pixmap.devicePixelRatio()
                print(f"[Debug] QImage {qimg.width()}x{qimg.height()}, pixmap {pixmap.width()}x{pixmap.height()}, DPR={dpr}")
            
            # AR-02/03: Détection sur la même QImage
            if self.ar_detector:
                dets = self.ar_detector.detect_on_qimage(qimg)
                print(f"[Debug] dets: {len(dets)} first: {dets[0] if dets else None}")
                
                # AR-01: Dessiner les overlays accrochés
                self.ar_page_view.draw_detections(dets, show_fullframe_debug=False)
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur AR render/detect: {e}")
            return False
    
    def _ar_render_page_to_qimage(self, doc, page_num: int, dpi: float) -> QtGui.QImage:
        """Rend une page PDF en QImage à la résolution spécifiée."""
        try:
            # Utiliser QPdfDocument.render() pour obtenir une QImage
            from PySide6.QtPdf import QPdfDocument
            
            if isinstance(doc, QPdfDocument):
                # Calculer la taille de rendu
                page_size = doc.pagePointSize(page_num)
                scale = dpi / 72.0  # Conversion points -> pixels
                
                render_size = QtCore.QSize(
                    int(page_size.width() * scale),
                    int(page_size.height() * scale)
                )
                
                # Rendre la page
                qimg = doc.render(page_num, render_size)
                return qimg
            else:
                print(f"❌ Type de document non supporté: {type(doc)}")
                return QtGui.QImage()
                
        except Exception as e:
            print(f"❌ Erreur rendu page: {e}")
            return QtGui.QImage()
    
    def ar_toggle_debug_frame(self):
        """AR-07: Active/désactive le cadre de debug full-frame."""
        if not (self._is_ar_enabled() and self.ar_current_qimage and self.ar_detector):
            return
            
        print("🔄 Toggle debug frame...")
        
        # Relancer la détection avec debug frame
        dets = self.ar_detector.detect_on_qimage(self.ar_current_qimage)
        self.ar_page_view.draw_detections(dets, show_fullframe_debug=True)
    
    def ar_add_menu_actions(self, menubar):
        """Ajoute les actions AR au menu principal."""
        ar_menu = menubar.addMenu("AR Mode")
        
        # Toggle AR mode
        toggle_action = ar_menu.addAction("Toggle AR Mode")
        toggle_action.triggered.connect(self._ar_toggle_mode)
        
        # Debug frame
        debug_action = ar_menu.addAction("Debug Frame")
        debug_action.triggered.connect(self.ar_toggle_debug_frame)
        
        # Reload current page
        reload_action = ar_menu.addAction("Reload Page")
        reload_action.triggered.connect(self._ar_reload_current_page)
    
    def _ar_toggle_mode(self):
        """Toggle entre mode AR et mode traditionnel."""
        if self._is_ar_enabled():
            self.disable_ar_mode()
        else:
            self.enable_ar_mode()
            # Recharger la page courante si possible
            self._ar_reload_current_page()
    
    def _ar_reload_current_page(self):
        """Recharge la page courante en mode AR."""
        if not self._is_ar_enabled():
            return
            
        # Récupérer le numéro de page courant
        current_page = 0
        if hasattr(self, 'view') and hasattr(self.view, 'pageNavigator'):
            try:
                current_page = self.view.pageNavigator().currentPage()
            except:
                pass
        
        print(f"🔄 Rechargement page {current_page} en mode AR...")
        self.ar_render_and_detect_page(current_page)
