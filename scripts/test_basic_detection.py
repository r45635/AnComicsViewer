#!/usr/bin/env python3
"""
Test de détection basique pour AnComicsViewer
Usage: python3 scripts/test_basic_detection.py
"""

import sys
import os
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_test_image():
    """Crée une image de test synthétique."""
    print("🎨 Création d'une image de test...")
    
    # Image de base (fond gris)
    img = np.ones((1200, 800, 3), dtype=np.uint8) * 200
    
    # Ajouter des rectangles blancs (simulent des panels)
    # Panel 1 (gauche)
    img[100:500, 50:350] = [255, 255, 255]
    
    # Panel 2 (droite)
    img[100:500, 400:750] = [255, 255, 255]
    
    # Panel 3 (bas)
    img[550:900, 100:700] = [255, 255, 255]
    
    print(f"✅ Image créée: {img.shape}")
    return img

def test_model_loading():
    """Test de chargement du modèle."""
    print("\n🔧 Test de chargement du modèle...")
    try:
        from src.ancomicsviewer.detectors.multibd_detector import MultiBDPanelDetector
        
        detector = MultiBDPanelDetector(device='cpu')
        detector._ensure_model_loaded()
        
        if detector.model is not None:
            print("✅ Modèle chargé avec succès")
            
            # Afficher les classes du modèle
            if hasattr(detector.model, 'names'):
                classes = detector.model.names
                print(f"✅ Classes disponibles: {classes}")
            
            return detector
        else:
            print("❌ Modèle non chargé")
            return None
            
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_raw_prediction(detector, img):
    """Test de prédiction brute."""
    print("\n🎯 Test de prédiction brute...")
    try:
        result = detector._predict_raw(img)
        print(f"✅ Prédictions: {len(result)} détections")
        
        if len(result) > 0:
            print("📊 Détails des détections:")
            for i, det in enumerate(result):
                x1, y1, x2, y2, score, cls = det
                print(f"  {i+1}: Box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] Score={score:.3f} Class={cls}")
        
        return len(result) > 0
        
    except Exception as e:
        print(f"❌ Erreur prédiction: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_model(detector, img):
    """Test direct avec le modèle YOLO."""
    print("\n🔬 Test direct avec modèle YOLO...")
    try:
        # Test avec plusieurs seuils de confiance
        conf_levels = [0.01, 0.05, 0.1, 0.25]
        
        for conf in conf_levels:
            results = detector.model.predict(
                img,
                conf=conf,
                verbose=False
            )
            
            if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                count = int(results[0].boxes.cls.shape[0]) if results[0].boxes.cls is not None else 0
                print(f"  Conf={conf}: {count} détections")
            else:
                print(f"  Conf={conf}: 0 détections")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test direct: {e}")
        return False

def test_config_values(detector):
    """Test des valeurs de configuration."""
    print("\n⚙️ Test des valeurs de configuration...")
    try:
        config = detector.config
        print(f"✅ CONF_BASE: {config.CONF_BASE}")
        print(f"✅ CONF_MIN: {config.CONF_MIN}")
        print(f"✅ IOU_NMS: {config.IOU_NMS}")
        print(f"✅ TARGET_MIN: {config.TARGET_MIN}")
        print(f"✅ TARGET_MAX: {config.TARGET_MAX}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur config: {e}")
        return False

def main():
    """Point d'entrée principal."""
    print("🚀 Test de détection basique AnComicsViewer")
    print("=" * 60)
    
    success = True
    
    # 1. Créer image de test
    test_img = create_test_image()
    
    # 2. Charger le modèle
    detector = test_model_loading()
    if not detector:
        print("❌ Impossible de charger le détecteur")
        return 1
    
    # 3. Test config
    if not test_config_values(detector):
        success = False
    
    # 4. Test prédiction brute
    if not test_raw_prediction(detector, test_img):
        success = False
    
    # 5. Test direct modèle
    if not test_direct_model(detector, test_img):
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Tests de détection basique réussis!")
    else:
        print("⚠️ Certains tests ont des problèmes")
    
    print("\n💡 Conseils:")
    print("  - Si aucune détection: le modèle peut nécessiter des images plus réalistes")
    print("  - Les images synthétiques simples peuvent ne pas être détectées")
    print("  - Testez avec de vraies pages de BD pour de meilleurs résultats")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
