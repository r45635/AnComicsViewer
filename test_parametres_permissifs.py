#!/usr/bin/env python3
"""
Test avec paramètres moins restrictifs pour mieux détecter les panels
"""

import sys
import os
sys.path.insert(0, '.')

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage

def test_parametres_permissifs():
    """Test avec paramètres plus permissifs pour detecter plus de panels"""
    
    if not QApplication.instance():
        app = QApplication(sys.argv)
    
    print("🔧 TEST PARAMÈTRES PERMISSIFS")
    print("=" * 50)
    
    try:
        from src.ancomicsviewer.detectors.robust_yolo_detector import RobustYoloDetector
        
        # Image test
        qimg = QImage('realistic_page.png')
        if qimg.isNull():
            print("❌ Impossible de charger realistic_page.png")
            return
            
        print(f"🖼️ Image: {qimg.width()}x{qimg.height()}")
        
        # Test 1: Paramètres originaux (baseline)
        print("\n📊 TEST 1: Paramètres AR originaux")
        print("-" * 30)
        detector1 = RobustYoloDetector()
        panels1 = detector1.detect_panels(qimg)
        print(f"✅ Panels détectés: {len(panels1)}")
        
        # Test 2: Confidence très basse + ratio plus permissif
        print("\n📊 TEST 2: Confidence très basse + ratio permissif")
        print("-" * 30)
        detector2 = RobustYoloDetector()
        detector2._cfg.conf_panel = 0.05      # Très bas
        detector2._cfg.conf_inset = 0.05      # Très bas
        detector2._cfg.conf_balloon = 0.10    # Bas
        detector2._cfg.drop_fullpage_ratio = 0.95  # Plus permissif (au lieu de 0.88)
        detector2._cfg.min_area_ratio = 0.001      # Plus permissif (au lieu de 0.002)
        
        panels2 = detector2.detect_panels(qimg)
        print(f"✅ Panels détectés: {len(panels2)}")
        
        # Test 3: Désactiver les filtres complètement
        print("\n📊 TEST 3: Filtres désactivés")
        print("-" * 30)
        detector3 = RobustYoloDetector()
        detector3._cfg.conf_panel = 0.05
        detector3._cfg.conf_inset = 0.05
        detector3._cfg.conf_balloon = 0.10
        detector3._cfg.drop_fullpage_ratio = 0.99  # Presque désactivé
        detector3._cfg.min_area_ratio = 0.0001     # Presque désactivé
        
        panels3 = detector3.detect_panels(qimg)
        print(f"✅ Panels détectés: {len(panels3)}")
        
        # Test 4: Test YOLO brut (accès direct)
        print("\n📊 TEST 4: YOLO brut sans filtrage")
        print("-" * 30)
        
        # Convertir QImage en array numpy
        from src.ancomicsviewer.detectors.robust_yolo_detector import qimage_to_rgb_array
        img_rgb = qimage_to_rgb_array(qimg)
        
        # Test YOLO direct
        yolo_detector = detector1._detector
        raw_results = yolo_detector._yolo(img_rgb, imgsz=1280, conf=0.05, iou=0.6, verbose=False)
        
        if raw_results and len(raw_results) > 0:
            result = raw_results[0]
            if result.boxes is not None:
                num_raw = len(result.boxes)
                print(f"✅ Détections YOLO brutes: {num_raw}")
                
                # Afficher détails des détections brutes
                for i, box in enumerate(result.boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = yolo_detector._yolo.names[cls]
                    
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    area_ratio = (w * h) / (qimg.width() * qimg.height())
                    
                    print(f"  Détection {i+1}: {class_name} conf={conf:.3f}")
                    print(f"    Box: ({x1:.0f},{y1:.0f}) -> ({x2:.0f},{y2:.0f})")
                    print(f"    Taille: {w:.0f}x{h:.0f} (ratio={area_ratio:.4f})")
                    
                    if area_ratio > 0.88:
                        print(f"    ⚠️ SERAIT SUPPRIMÉ par filtre pleine-page")
                    if area_ratio < 0.002:
                        print(f"    ⚠️ SERAIT SUPPRIMÉ par filtre taille minimale")
            else:
                print("❌ Aucune détection YOLO brute")
        else:
            print("❌ Résultat YOLO vide")
        
        # Résumé comparatif
        print(f"\n📈 RÉSUMÉ COMPARATIF:")
        print(f"  AR originaux: {len(panels1)} panels")
        print(f"  Permissifs: {len(panels2)} panels")
        print(f"  Filtres off: {len(panels3)} panels")
        print(f"  YOLO brut: {num_raw if 'num_raw' in locals() else 'N/A'} détections")
        
        # Diagnostic
        if len(panels2) > len(panels1):
            print("✅ Les paramètres permissifs améliorent la détection")
        if len(panels3) > len(panels2):
            print("✅ Les filtres suppriment des vrais panels")
        if 'num_raw' in locals() and num_raw > len(panels3):
            print("✅ Le post-processing supprime des détections valides")
            
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parametres_permissifs()
