#!/usr/bin/env python3
"""
Test direct de détection pour diagnostiquer le problème
"""

import sys
import os
sys.path.insert(0, '.')

def test_detection_directe():
    """Test direct de la détection sans passer par l'interface"""
    print("🔍 TEST DÉTECTION DIRECTE")
    print("=" * 50)
    
    try:
        # 1. Test import
        print("📦 Import du détecteur...")
        from src.ancomicsviewer.detectors.robust_yolo_detector import RobustYoloDetector
        from PySide6.QtGui import QImage
        print("✅ Imports OK")
        
        # 2. Test création détecteur
        print("🔥 Création détecteur...")
        detector = RobustYoloDetector()
        print("✅ Détecteur créé")
        
        # 3. Test avec image
        if os.path.exists('realistic_page.png'):
            print("🖼️ Test avec realistic_page.png...")
            qimg = QImage('realistic_page.png')
            print(f"📏 Image: {qimg.width()}x{qimg.height()}")
            
            # 4. Détection
            print("🎯 Lancement détection...")
            panels = detector.detect_panels(qimg)
            print(f"✅ Résultat: {len(panels)} panels détectés")
            
            for i, panel in enumerate(panels):
                print(f"   Panel {i+1}: {panel}")
                
        else:
            print("❌ realistic_page.png non trouvé")
            
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detection_directe()
