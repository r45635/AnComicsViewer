#!/usr/bin/env python3
"""
Test direct de la détection dans l'application en cours
======================================================
Force une nouvelle détection pour voir les vrais résultats
"""

import sys
import os
sys.path.insert(0, '.')

def test_detection_realtime():
    """Test la détection telle qu'utilisée dans l'application"""
    
    print("🔍 TEST DÉTECTION EN TEMPS RÉEL")
    print("=" * 50)
    
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QImage
        from PySide6.QtCore import QSizeF
        
        app = QApplication(sys.argv)
        
        # 1. Test du détecteur directement
        print("📋 1. TEST DÉTECTEUR DIRECT:")
        from src.ancomicsviewer.detectors.robust_yolo_detector import RobustYoloDetector
        detector = RobustYoloDetector()
        
        qimg = QImage('realistic_page.png')
        if qimg.isNull():
            print("❌ Image realistic_page.png non trouvée")
            return
            
        panels_direct = detector.detect_panels(qimg)
        print(f"✅ Détecteur direct: {len(panels_direct)} panels")
        
        # 2. Test avec simulation main_app.py  
        print("\n📋 2. TEST SIMULATION MAIN_APP:")
        
        # Simuler exactement ce que fait main_app.py dans _ensure_panels_for
        try:
            # pt = self.document.pagePointSize(page)
            pt = QSizeF(800.0, 1200.0)  # Points typiques d'une page PDF
            
            # dpi = self._det_dpi (défaut = 150 dans main_app)
            dpi = 150
            
            # scale = dpi / 72.0  
            scale = dpi / 72.0
            
            # qsize = QSizeF(pt.width() * scale, pt.height() * scale).toSize()
            target_width = int(pt.width() * scale)
            target_height = int(pt.height() * scale)
            
            print(f"📏 Simulation main_app:")
            print(f"   Page points: {pt.width()}x{pt.height()}")
            print(f"   DPI: {dpi}")
            print(f"   Scale: {scale}")
            print(f"   Target size: {target_width}x{target_height}")
            
            # Redimensionner l'image comme le fait main_app
            if qimg.width() != target_width or qimg.height() != target_height:
                print(f"🔄 Redimensionnement: {qimg.width()}x{qimg.height()} -> {target_width}x{target_height}")
                qimg_scaled = qimg.scaled(target_width, target_height)
            else:
                qimg_scaled = qimg
                
            # rects = self._panel_detector.detect_panels(qimg, pt)
            panels_scaled = detector.detect_panels(qimg_scaled, pt)
            print(f"✅ Main_app simulation: {len(panels_scaled)} panels")
            
            # Comparaison
            print(f"\n📊 COMPARAISON:")
            print(f"   Direct: {len(panels_direct)} panels")
            print(f"   Main_app: {len(panels_scaled)} panels")
            
            if len(panels_direct) != len(panels_scaled):
                print("⚠️ DIFFÉRENCE! Le redimensionnement affecte la détection")
                
            # 3. Test avec taille réelle PDF
            print(f"\n📋 3. TEST AVEC VRAIE TAILLE PDF:")
            
            # Taille typique Golden City à 150 DPI
            real_pdf_width = int(595 * 150 / 72)  # ~1240px
            real_pdf_height = int(842 * 150 / 72)  # ~1750px
            
            print(f"📄 Taille PDF réelle @ 150 DPI: {real_pdf_width}x{real_pdf_height}")
            
            qimg_pdf = qimg.scaled(real_pdf_width, real_pdf_height)
            pt_pdf = QSizeF(595, 842)  # A4 en points
            
            panels_pdf = detector.detect_panels(qimg_pdf, pt_pdf)
            print(f"✅ Taille PDF réelle: {len(panels_pdf)} panels")
            
            # Analyse détaillée si différent
            if len(panels_direct) != len(panels_pdf):
                print("\n🔍 ANALYSE DÉTAILLÉE:")
                
                print("Panels direct:")
                for i, p in enumerate(panels_direct):
                    print(f"   {i+1}: ({p.x():.0f},{p.y():.0f}) {p.width():.0f}x{p.height():.0f}")
                    
                print("Panels PDF:")
                for i, p in enumerate(panels_pdf):
                    print(f"   {i+1}: ({p.x():.0f},{p.y():.0f}) {p.width():.0f}x{p.height():.0f}")
                    
        except Exception as e:
            print(f"❌ Erreur simulation: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ ERREUR GLOBALE: {e}")
        import traceback  
        traceback.print_exc()

if __name__ == "__main__":
    test_detection_realtime()
