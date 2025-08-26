#!/usr/bin/env python3
"""
Test de détection forcée dans le contexte application
"""

import sys
import os
sys.path.insert(0, '.')

def test_detection_in_app():
    """Simule la détection comme dans l'application"""
    
    print("🎯 TEST DÉTECTION DANS CONTEXTE APPLICATION")
    print("=" * 60)
    
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QImage
        from PySide6.QtCore import QSizeF
        from src.ancomicsviewer.detectors.robust_yolo_detector import RobustYoloDetector
        
        app = QApplication(sys.argv)
        
        # Charger image
        qimg = QImage('realistic_page.png')
        if qimg.isNull():
            print("❌ Image non trouvée")
            return
            
        print(f"🖼️ Image chargée: {qimg.width()}x{qimg.height()}")
        
        # Créer détecteur
        detector = RobustYoloDetector()
        
        # Simuler l'appel comme dans main_app.py
        print("\n🔥 SIMULATION _ensure_panels_for:")
        print("-" * 40)
        
        # Équivalent de: rects = self._panel_detector.detect_panels(qimg, pt)
        # où pt est page_point_size (en points)
        pt = QSizeF(800.0, 1200.0)  # Taille en points (comme page PDF)
        
        print(f"📏 Page size (points): {pt.width()}x{pt.height()}")
        
        # Détection avec l'API exacte utilisée dans main_app.py
        panels = detector.detect_panels(qimg, pt)
        
        print(f"\n✅ RÉSULTAT: {len(panels)} panels détectés")
        
        for i, panel in enumerate(panels):
            x, y, w, h = panel.x(), panel.y(), panel.width(), panel.height()
            
            # Vérifications de cohérence (comme dans l'app)
            if x < 0 or y < 0 or x + w > qimg.width() or y + h > qimg.height():
                print(f"⚠️ Panel {i+1}: COORDONNÉES INVALIDES!")
            
            print(f"📍 Panel {i+1}:")
            print(f"   Position: ({x:.1f}, {y:.1f})")
            print(f"   Taille: {w:.1f} x {h:.1f}")
            print(f"   Ratio/image: {w/qimg.width():.3f} x {h/qimg.height():.3f}")
            
            # Test si dans les limites
            if 0 <= x < qimg.width() and 0 <= y < qimg.height():
                print(f"   ✅ Position valide")
            else:
                print(f"   ❌ Position invalide!")
                
        # Test navigation simulation
        if len(panels) > 0:
            print(f"\n🧭 SIMULATION NAVIGATION:")
            print("-" * 30)
            
            for i, panel in enumerate(panels):
                print(f"Panel {i+1}: Focus sur ({panel.x():.0f},{panel.y():.0f}) taille {panel.width():.0f}x{panel.height():.0f}")
                
                # Vérifier si le panel est assez grand pour être utilisable
                min_size = 50  # pixels
                if panel.width() > min_size and panel.height() > min_size:
                    print(f"   ✅ Taille utilisable pour navigation")
                else:
                    print(f"   ⚠️ Panel trop petit pour navigation")
        
        print(f"\n🎉 TEST TERMINÉ - Détection: {'✅ OK' if len(panels) > 0 else '❌ ÉCHEC'}")
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detection_in_app()
