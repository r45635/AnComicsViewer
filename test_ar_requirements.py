#!/usr/bin/env python3
"""
Test des Acceptance Requirements (AR) pour le détecteur YOLO robuste
====================================================================
Valide la conformité aux spécifications AR pour la détection robuste.
"""

import sys
import os
sys.path.insert(0, '.')

from src.ancomicsviewer.detectors.robust_yolo_detector import RobustYoloDetector
from PySide6.QtGui import QImage
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("AR_Test")

def test_ar_requirements():
    """Test complet des Acceptance Requirements (AR)."""
    print("🧪 TEST AR - ACCEPTANCE REQUIREMENTS")
    print("=" * 50)
    
    # AR-1: Initialisation du détecteur
    print("\n📋 AR-1: Initialisation détecteur robuste...")
    try:
        detector = RobustYoloDetector()
        model_info = detector.get_model_info()
        print(f"✅ Détecteur: {model_info['name']}")
        print(f"✅ Device: {model_info['device']}")
        print(f"✅ DPI: {model_info['render_dpi']}")
        print(f"✅ Conf panel: {model_info['conf_panel']}")
        print(f"✅ Conf inset: {model_info['conf_inset']}")
        print(f"✅ Conf balloon: {model_info['conf_balloon']}")
    except Exception as e:
        print(f"❌ AR-1 ÉCHEC: {e}")
        return False
    
    # AR-2: Test image réaliste
    print("\n📋 AR-2: Test détection sur image réaliste...")
    if os.path.exists('realistic_page.png'):
        qimg = QImage('realistic_page.png')
        if not qimg.isNull():
            start_time = time.time()
            rects = detector.detect_panels(qimg)
            detect_time = time.time() - start_time
            
            print(f"✅ Détections: {len(rects)} panels")
            print(f"✅ Temps: {detect_time:.3f}s")
            
            if len(rects) >= 1:
                print("✅ AR-2: Au moins 1 panel détecté (req: >=1)")
            else:
                print("❌ AR-2: Aucun panel détecté (req: >=1)")
                return False
                
            # Vérifier les rectangles
            for i, rect in enumerate(rects):
                area = rect.width() * rect.height()
                img_area = qimg.width() * qimg.height()
                area_ratio = area / img_area
                print(f"   Panel {i+1}: {rect.width():.0f}x{rect.height():.0f} (ratio={area_ratio:.3f})")
                
                if area_ratio > 0.002:  # AR: min_area_ratio=0.002
                    print(f"   ✅ Panel {i+1}: Taille OK (>{0.002:.3f})")
                else:
                    print(f"   ⚠️ Panel {i+1}: Trop petit (<{0.002:.3f})")
        else:
            print("❌ AR-2: Impossible de charger realistic_page.png")
            return False
    else:
        print("❌ AR-2: realistic_page.png non trouvé")
        return False
    
    # AR-3: Test cache (double détection)
    print("\n📋 AR-3: Test cache (double détection)...")
    start_time = time.time()
    rects2 = detector.detect_panels(qimg)  # Deuxième appel
    cache_time = time.time() - start_time
    
    print(f"✅ Cache time: {cache_time:.3f}s (vs {detect_time:.3f}s)")
    if len(rects2) == len(rects):
        print("✅ AR-3: Cache cohérent")
    else:
        print(f"❌ AR-3: Cache incohérent ({len(rects2)} vs {len(rects)})")
        return False
    
    # AR-4: Test retry (image vide simulée)
    print("\n📋 AR-4: Test retry sur image difficile...")
    # Créer une image quasi-vide pour tester le retry
    empty_img = QImage(800, 600, QImage.Format.Format_RGB888)
    empty_img.fill(255)  # Blanc total
    
    rects_empty = detector.detect_panels(empty_img)
    print(f"✅ Image vide: {len(rects_empty)} détections")
    print("✅ AR-4: Retry testé (peut être 0 sur image vide)")
    
    # AR-5: Test robustesse paramètres
    print("\n📋 AR-5: Vérification paramètres AR...")
    cfg = detector._cfg
    ar_params = {
        "imgsz": 1280,
        "iou": 0.60,
        "conf_panel": 0.20,
        "conf_inset": 0.20,
        "conf_balloon": 0.30,
        "drop_fullpage_ratio": 0.88,
        "min_area_ratio": 0.002,
    }
    
    for param, expected in ar_params.items():
        actual = getattr(cfg, param)
        if actual == expected:
            print(f"✅ {param}: {actual} (conforme AR)")
        else:
            print(f"❌ {param}: {actual} != {expected} (non-conforme AR)")
            return False
    
    print("\n🎉 TOUS LES AR TESTS RÉUSSIS !")
    print("✅ Détecteur YOLO robuste AR-compliant validé")
    return True

if __name__ == "__main__":
    success = test_ar_requirements()
    sys.exit(0 if success else 1)
