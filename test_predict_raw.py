#!/usr/bin/env python3
"""
Test simple et direct de _predict_raw
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import logging

# Configurer les logs pour voir les messages
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Créer une vraie image de BD
def create_real_comic_page():
    """Crée une image plus réaliste d'une page de BD."""
    img = np.ones((1800, 1200, 3), dtype=np.uint8) * 245  # Fond blanc cassé
    
    # Panel 1 - case rectangulaire avec bordure noire épaisse
    img[100:800, 100:550] = [255, 255, 255]  # Fond blanc
    img[100:110, 100:550] = [0, 0, 0]        # Bordure haut (10px)
    img[790:800, 100:550] = [0, 0, 0]        # Bordure bas
    img[100:800, 100:110] = [0, 0, 0]        # Bordure gauche
    img[100:800, 540:550] = [0, 0, 0]        # Bordure droite
    
    # Panel 2 - case à droite
    img[100:800, 600:1050] = [255, 255, 255]
    img[100:110, 600:1050] = [0, 0, 0]
    img[790:800, 600:1050] = [0, 0, 0]
    img[100:800, 600:610] = [0, 0, 0]
    img[100:800, 1040:1050] = [0, 0, 0]
    
    # Panel 3 - grande case en bas
    img[850:1600, 100:1050] = [255, 255, 255]
    img[850:860, 100:1050] = [0, 0, 0]
    img[1590:1600, 100:1050] = [0, 0, 0]
    img[850:1600, 100:110] = [0, 0, 0]
    img[850:1600, 1040:1050] = [0, 0, 0]
    
    # Ajouter du contenu dans les cases pour être plus réaliste
    # Rectangles de "texte" simulé
    img[200:250, 150:500] = [220, 220, 220]  # Zone de texte grise
    img[400:600, 200:400] = [200, 200, 200]  # Personnage/contenu
    
    return img

def test_predict_raw_detailed():
    print("🔬 TEST DÉTAILLÉ DE _predict_raw")
    print("=" * 50)
    
    from src.ancomicsviewer.detectors.multibd_detector import MultiBDPanelDetector
    
    # 1. Créer le détecteur
    detector = MultiBDPanelDetector(device='cpu')
    print(f"✅ Détecteur créé - Device: {detector.device}")
    
    # 2. Créer une image réaliste
    test_img = create_real_comic_page()
    print(f"✅ Image réaliste créée: {test_img.shape}")
    
    # 3. Test _predict_raw avec logs détaillés
    print("\n🔍 Appel _predict_raw...")
    try:
        result = detector._predict_raw(test_img)
        print(f"📊 RÉSULTAT FINAL: {len(result)} détections")
        
        if len(result) > 0:
            print("📋 Détails des détections:")
            for i, det in enumerate(result[:5]):
                x1, y1, x2, y2, score, cls = det
                w, h = x2-x1, y2-y1
                print(f"  {i+1}: classe={cls} score={score:.3f} bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] size={w:.0f}x{h:.0f}")
        else:
            print("❌ Aucune détection")
            
    except Exception as e:
        print(f"❌ Erreur dans _predict_raw: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_predict_raw_detailed()
