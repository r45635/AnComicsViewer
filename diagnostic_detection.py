#!/usr/bin/env python3
"""
Diagnostic complet des détections de panels
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from src.ancomicsviewer.detectors.multibd_detector import MultiBDPanelDetector
from PySide6.QtGui import QImage
from PySide6.QtCore import QSize

def create_test_comic_image():
    """Crée une image synthétique de BD avec des rectangles nets."""
    # Image 800x1200 (format BD classique)
    img = np.ones((1200, 800, 3), dtype=np.uint8) * 240  # Fond gris clair
    
    # Panel 1 (haut gauche)
    img[50:400, 50:380] = [255, 255, 255]  # Fond blanc
    img[50:55, 50:380] = [0, 0, 0]        # Bordure haut
    img[395:400, 50:380] = [0, 0, 0]      # Bordure bas
    img[50:400, 50:55] = [0, 0, 0]        # Bordure gauche
    img[50:400, 375:380] = [0, 0, 0]      # Bordure droite
    
    # Panel 2 (haut droite)
    img[50:400, 420:750] = [255, 255, 255]
    img[50:55, 420:750] = [0, 0, 0]
    img[395:400, 420:750] = [0, 0, 0]
    img[50:400, 420:425] = [0, 0, 0]
    img[50:400, 745:750] = [0, 0, 0]
    
    # Panel 3 (bas, large)
    img[450:1100, 50:750] = [255, 255, 255]
    img[450:455, 50:750] = [0, 0, 0]
    img[1095:1100, 50:750] = [0, 0, 0]
    img[450:1100, 50:55] = [0, 0, 0]
    img[450:1100, 745:750] = [0, 0, 0]
    
    return img

def test_detection_pipeline():
    print("🔬 DIAGNOSTIC COMPLET DES DÉTECTIONS")
    print("=" * 50)
    
    # 1. Créer le détecteur
    detector = MultiBDPanelDetector(device='cpu')
    print(f"✅ Détecteur créé - Device: {detector.device}")
    
    # 2. Créer une image de test
    test_img = create_test_comic_image()
    print(f"✅ Image de test créée: {test_img.shape}")
    
    # 3. Test du modèle avec différents seuils
    detector._ensure_model_loaded()
    
    # Test 1: Seuil très bas
    print("\n🔍 TEST 1: Seuil de confiance très bas (0.01)")
    try:
        results = detector.model.predict(
            test_img,
            conf=0.01,
            iou=0.5,
            classes=None,
            verbose=False
        )[0]
        
        if hasattr(results, 'boxes') and results.boxes is not None:
            count = int(results.boxes.cls.shape[0]) if results.boxes.cls is not None else 0
            print(f"  📊 Détections: {count}")
            
            if count > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                labels = results.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(min(5, count)):  # Afficher max 5 détections
                    x1, y1, x2, y2 = boxes[i]
                    score = scores[i]
                    cls_name = detector.model.names[labels[i]]
                    print(f"    {i+1}: {cls_name} @ {score:.3f} - Box: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        else:
            print("  ❌ Aucune détection")
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
    
    # Test 2: Pipeline complet
    print("\n🔍 TEST 2: Pipeline _predict_raw complet")
    try:
        dets = detector._predict_raw(test_img)
        print(f"  📊 Résultat _predict_raw: {len(dets)} détections")
        
        if len(dets) > 0:
            for i, det in enumerate(dets[:3]):  # Afficher max 3
                x1, y1, x2, y2, score, cls_idx = det
                print(f"    {i+1}: classe {cls_idx} @ {score:.3f} - Box: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
    except Exception as e:
        print(f"  ❌ Erreur pipeline: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Conversion QImage
    print("\n🔍 TEST 3: Test avec QImage (comme l'application)")
    try:
        # Convertir en QImage
        h, w, ch = test_img.shape
        bytes_per_line = ch * w
        qimage = QImage(test_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Test detect_panels
        from PySide6.QtCore import QSizeF
        panels = detector.detect_panels(qimage, QSizeF(w, h))
        print(f"  📊 detect_panels résultat: {len(panels)} panels")
        
        for i, panel in enumerate(panels[:3]):
            print(f"    Panel {i+1}: x={panel.x():.0f}, y={panel.y():.0f}, w={panel.width():.0f}, h={panel.height():.0f}")
            
    except Exception as e:
        print(f"  ❌ Erreur QImage: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detection_pipeline()
