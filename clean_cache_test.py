#!/usr/bin/env python3
"""
Nettoyage du cache et test de détection forcée
==============================================
Force l'application à utiliser le nouveau détecteur
"""

import sys
import os
import shutil
sys.path.insert(0, '.')

def clean_cache_and_test():
    """Nettoie le cache et teste la nouvelle détection"""
    
    print("🧹 NETTOYAGE CACHE & TEST FORCÉ")
    print("=" * 50)
    
    # 1. Nettoyer le cache
    cache_dirs = [
        os.path.expanduser("~/.ancomicsviewer/cache"),
        os.path.join(os.getcwd(), "cache"),
        os.path.join(os.getcwd(), ".cache"),
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            print(f"🗑️ Suppression cache: {cache_dir}")
            try:
                shutil.rmtree(cache_dir)
                print("✅ Cache supprimé")
            except Exception as e:
                print(f"⚠️ Erreur suppression: {e}")
        else:
            print(f"📁 Cache inexistant: {cache_dir}")
    
    # 2. Test détecteur avec DPI 150 (comme main_app)
    print(f"\n🎯 TEST DÉTECTEUR DPI=150 (comme main_app):")
    
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QImage
        from PySide6.QtCore import QSizeF
        from src.ancomicsviewer.detectors.robust_yolo_detector import RobustYoloDetector
        
        app = QApplication(sys.argv)
        
        # Vérifier que le détecteur utilise DPI=150
        detector = RobustYoloDetector()
        print(f"📏 Détecteur DPI: {detector.render_dpi} (doit être 150)")
        
        # Image test
        qimg = QImage('realistic_page.png')
        if qimg.isNull():
            print("❌ Image test non trouvée")
            return
            
        print(f"🖼️ Image: {qimg.width()}x{qimg.height()}")
        
        # Test détection directe
        panels = detector.detect_panels(qimg)
        print(f"✅ Détection directe: {len(panels)} panels")
        
        for i, panel in enumerate(panels):
            x, y, w, h = panel.x(), panel.y(), panel.width(), panel.height()
            print(f"   Panel {i+1}: ({x:.0f},{y:.0f}) {w:.0f}x{h:.0f}")
        
        # 3. Test simulation exacte main_app avec DPI=150
        print(f"\n🔄 SIMULATION MAIN_APP EXACTE (DPI=150):")
        
        # Paramètres réels de main_app
        dpi = 150.0  # self._det_dpi
        pt = QSizeF(595.0, 842.0)  # Page A4 typique en points
        scale = dpi / 72.0  # Facteur d'échelle
        
        target_width = int(pt.width() * scale)
        target_height = int(pt.height() * scale)
        
        print(f"📄 Page points: {pt.width()}x{pt.height()}")
        print(f"🔍 Scale @ {dpi} DPI: {scale}")
        print(f"📐 Target size: {target_width}x{target_height}")
        
        # Redimensionner comme main_app
        qimg_app = qimg.scaled(target_width, target_height)
        print(f"🖼️ Image redimensionnée: {qimg_app.width()}x{qimg_app.height()}")
        
        # Détection comme main_app
        panels_app = detector.detect_panels(qimg_app, pt)
        print(f"✅ Main_app simulation: {len(panels_app)} panels")
        
        for i, panel in enumerate(panels_app):
            x, y, w, h = panel.x(), panel.y(), panel.width(), panel.height()
            print(f"   Panel {i+1}: ({x:.0f},{y:.0f}) {w:.0f}x{h:.0f}")
        
        # 4. Comparaison
        print(f"\n📊 COMPARAISON FINALE:")
        print(f"   Image originale: {len(panels)} panels")
        print(f"   Simulation app: {len(panels_app)} panels")
        
        if len(panels) != len(panels_app):
            print("⚠️ Le redimensionnement change le résultat!")
        else:
            print("✅ Cohérence détection")
            
        print(f"\n💡 PROCHAIN TEST:")
        print(f"   1. Relancer l'application")
        print(f"   2. Ouvrir un PDF") 
        print(f"   3. Appuyer sur P pour voir panels")
        print(f"   4. Le cache est vide, nouvelle détection forcée")
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    clean_cache_and_test()
