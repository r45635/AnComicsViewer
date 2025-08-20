#!/usr/bin/env python3
"""
Test d'intégration du modèle Multi-BD Enhanced v2.0
==================================================
Teste le détecteur optimisé dans AnComicsViewer.
"""

import sys
import os
from pathlib import Path

def test_detector():
    """Test du détecteur optimisé"""
    print("🧪 Test Multi-BD Enhanced v2.0 dans AnComicsViewer")
    print("=" * 50)
    
    try:
        # Configuration PyTorch 2.8
        import torch
        try:
            import torch.serialization
            import ultralytics.nn.tasks
            torch.serialization.add_safe_globals([
                ultralytics.nn.tasks.DetectionModel,
                ultralytics.nn.tasks.SegmentationModel,
                ultralytics.nn.tasks.ClassificationModel,
                ultralytics.nn.tasks.PoseModel,
                ultralytics.nn.tasks.OBBModel
            ])
        except Exception:
            pass
        
        # Import du détecteur
        sys.path.append('detectors')
        from multibd_detector import MultiBDPanelDetector
        
        # Test de chargement
        print("📦 Chargement détecteur...")
        detector = MultiBDPanelDetector()
        
        # Info modèle
        info = detector.get_model_info()
        print(f"✅ {info['name']} {info['version']}")
        print(f"📊 Performance: mAP50 {info['performance']['mAP50']}")
        print(f"🎯 Confidence: {info['confidence']}")
        
        # Test sur image de validation si disponible
        val_images = list(Path("dataset/yolo/images/val").glob("*.jpg"))
        if val_images:
            from PySide6.QtGui import QImage
            from PySide6.QtCore import QSizeF
            
            test_image = val_images[0]
            print(f"\n🖼️  Test image: {test_image.name}")
            
            # Chargement image
            qimg = QImage(str(test_image))
            if qimg.isNull():
                print("❌ Erreur chargement image")
                return False
            
            page_size = QSizeF(qimg.width(), qimg.height())
            
            # Détection
            panels = detector.detect_panels(qimg, page_size)
            print(f"📋 Panneaux détectés: {len(panels)}")
            
            if panels:
                for i, panel in enumerate(panels[:3]):  # Afficher les 3 premiers
                    print(f"   Panel {i+1}: {panel.width():.0f}x{panel.height():.0f} at ({panel.x():.0f},{panel.y():.0f})")
        
        print("\n✅ Test détecteur réussi!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test principal"""
    success = test_detector()
    
    if success:
        print("\n🎉 Le modèle Multi-BD Enhanced v2.0 est prêt!")
        print("\n📋 Pour utiliser dans AnComicsViewer:")
        print("   1. Lancer: python AnComicsViewer.py")
        print("   2. Ouvrir: La Pin-up du B24 - T01.pdf")
        print("   3. Menu: Panels > Detector > Multi-BD Enhanced")
        print("   4. Tester la détection automatique")
        
        return 0
    else:
        print("\n❌ Problème avec le modèle")
        return 1

if __name__ == "__main__":
    exit(main())
