#!/usr/bin/env python3
"""
Test des améliorations de sécurité pour AnComicsViewer
"""

import os
import sys
from pathlib import Path

# Configuration plus stricte par défaut (désactive split interne)
os.environ["ACV_CONF"] = "0.4"
os.environ["ACV_IOU"] = "0.55" 
os.environ["ACV_MIN_AREA_FRAC"] = "0.02"
os.environ["ACV_SPLIT_INTERNAL"] = "0"  # Désactivé par défaut

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from src.ancomicsviewer.detectors.multibd_detector import MultiBDPanelDetector

def test_improved_detection():
    """Test rapide des améliorations"""
    print("🧪 Test des améliorations de détection")
    print("=" * 50)
    print(f"📊 Configuration:")
    print(f"   • ACV_CONF: {os.getenv('ACV_CONF')}")
    print(f"   • ACV_IOU: {os.getenv('ACV_IOU')}")
    print(f"   • ACV_MIN_AREA_FRAC: {os.getenv('ACV_MIN_AREA_FRAC')}")
    print(f"   • ACV_SPLIT_INTERNAL: {os.getenv('ACV_SPLIT_INTERNAL')}")
    print("-" * 50)
    
    try:
        # Charger le détecteur avec les nouveaux paramètres
        detector = MultiBDPanelDetector()
        print(f"✅ Détecteur chargé avec conf={detector.conf}, iou={detector.iou}")
        print(f"   • min_area_frac={detector.min_area_frac}")
        print(f"   • enable_internal_split={detector.enable_internal_split}")
        
        # Test sur une image de validation
        test_image = "dataset/images/val/p0002.png"
        if Path(test_image).exists():
            print(f"\n📸 Test sur: {test_image}")
            image = cv2.imread(test_image)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Simuler QImage et QSizeF pour le test
            from PySide6.QtGui import QImage
            from PySide6.QtCore import QSizeF
            
            h, w = image.shape[:2]
            qimage = QImage(image.data, w, h, w * 3, QImage.Format_RGB888)
            page_size = QSizeF(w, h)
            
            # Détection
            panels = detector.detect_panels(qimage, page_size)
            
            print(f"🎯 {len(panels)} panels détectés (seuils plus stricts)")
            for i, panel in enumerate(panels):
                print(f"   Panel {i+1}: {panel.width():.0f}x{panel.height():.0f}")
                
        else:
            print(f"⚠️  Image de test non trouvée: {test_image}")
            print("💡 Utilisez une image de votre choix")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

def test_split_sensitivity():
    """Test de la sensibilité du split interne"""
    print("\n🔬 Test de sensibilité du split interne")
    print("=" * 50)
    
    # Test avec split activé sur panels très allongés
    os.environ["ACV_SPLIT_INTERNAL"] = "1"
    print("🔄 Activation temporaire du split interne...")
    
    try:
        detector = MultiBDPanelDetector()
        print(f"✅ Split interne: {detector.enable_internal_split}")
        
        print("💡 Le split ne s'activera que sur panels avec ratio > 1.8")
        print("📋 Tests conseillés:")
        print("   • Pages manga avec plusieurs colonnes")
        print("   • Pages avec panels très allongés horizontalement")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    # Restaurer les paramètres par défaut
    os.environ["ACV_SPLIT_INTERNAL"] = "0"

if __name__ == "__main__":
    print("🎯 Test des Améliorations AnComicsViewer Enhanced v2")
    print("🛡️  Sécurisation du découpage interne")
    print("🎛️  Paramètres configurables par environnement")
    print()
    
    test_improved_detection()
    test_split_sensitivity()
    
    print("\n📝 UTILISATION:")
    print("Pour ajuster les paramètres, définissez les variables d'environnement:")
    print("   export ACV_CONF=0.45        # Plus strict = moins de faux positifs")
    print("   export ACV_MIN_AREA_FRAC=0.025  # Plus strict = ignore micro-panels") 
    print("   export ACV_SPLIT_INTERNAL=1     # Active le split sur panels allongés")
    print("   python main.py --detector multibd")
    
    print("\n✅ Tests terminés !")
