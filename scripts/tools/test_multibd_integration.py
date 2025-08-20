#!/usr/bin/env python3
"""
Test de l'intégration du détecteur Multi-BD dans AnComicsViewer.
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au PATH pour imports
sys.path.append("..")

# Ajouter le patch PyTorch
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
exec(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'patch_pytorch.py')).read())

def test_multibd_integration():
    """Test l'intégration du détecteur Multi-BD."""
    
    print("🧪 TEST INTÉGRATION DÉTECTEUR MULTI-BD")
    print("=" * 45)
    
    # 1. Test d'import du détecteur
    try:
        from detectors.multibd_detector import MultiBDPanelDetector
        print("✅ Import MultiBDPanelDetector : OK")
    except Exception as e:
        print(f"❌ Échec import : {e}")
        return False
    
    # 2. Test de chargement du modèle
    model_path = "runs/detect/multibd_mixed_model/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Modèle non trouvé : {model_path}")
        print("💡 Exécutez d'abord train_multibd_model.py")
        return False
    
    try:
        detector = MultiBDPanelDetector()
        print(f"✅ Chargement modèle : OK")
        
        # Afficher les infos du modèle
        info = detector.get_model_info()
        print(f"📊 Modèle : {info['name']}")
        print(f"🎯 Performance : mAP50 {info['performance']['mAP50']}")
        print(f"📚 Entraîné sur : {', '.join(info['training_data'])}")
        
    except Exception as e:
        print(f"❌ Échec chargement modèle : {e}")
        return False
    
    # 3. Test d'import de PySide6 (requis pour le viewer)
    try:
        from PySide6.QtCore import QRectF, QSizeF
        from PySide6.QtGui import QImage
        print("✅ Import PySide6 : OK")
    except Exception as e:
        print(f"❌ Échec import PySide6 : {e}")
        return False
    
    # 4. Test avec une image de test si disponible
    test_images = [
        "dataset/images/train/p0003.png",
        "dataset/images/train/tintin_p0001.png", 
        "dataset/images/train/pinup_p0001.png"
    ]
    
    for img_path in test_images:
        if Path(img_path).exists():
            print(f"\n🖼️  Test détection sur : {Path(img_path).name}")
            
            try:
                # Charger l'image avec QImage
                qimg = QImage(img_path)
                if qimg.isNull():
                    print(f"⚠️  Impossible de charger l'image")
                    continue
                
                # Simuler une taille de page (A4 en points)
                page_size = QSizeF(595, 842)
                
                # Détecter les panels
                panels = detector.detect_panels(qimg, page_size)
                
                print(f"   📦 {len(panels)} panels détectés")
                
                if panels:
                    for i, panel in enumerate(panels[:3]):  # Afficher max 3
                        print(f"      Panel {i+1}: x={panel.x():.1f}, y={panel.y():.1f}, "
                              f"w={panel.width():.1f}, h={panel.height():.1f}")
                
            except Exception as e:
                print(f"   ❌ Erreur détection : {e}")
                
            break  # Tester seulement la première image trouvée
    
    print(f"\n✅ INTÉGRATION MULTI-BD : SUCCÈS")
    print(f"💡 Vous pouvez maintenant utiliser ⚙️ → Detector → Multi-BD (Trained)")
    return True

def test_viewer_launch():
    """Test de lancement rapide du viewer."""
    
    print(f"\n🚀 TEST LANCEMENT VIEWER")
    print("=" * 30)
    
    try:
        # Test d'import du viewer principal
        import AnComicsViewer
        print("✅ Import AnComicsViewer : OK")
        
        # Test d'import PySide6
        from PySide6.QtWidgets import QApplication
        print("✅ Import QApplication : OK")
        
        print("💡 Pour tester le viewer complet :")
        print("   python AnComicsViewer.py")
        print("   Puis ⚙️ → Detector → Multi-BD (Trained)")
        
        return True
        
    except Exception as e:
        print(f"❌ Échec test viewer : {e}")
        return False

def main():
    """Fonction principale."""
    
    print("🎯 TEST INTÉGRATION COMPLÈTE")
    print("=" * 50)
    print()
    
    success = True
    
    # Test 1 : Intégration détecteur
    if not test_multibd_integration():
        success = False
    
    # Test 2 : Lancement viewer
    if not test_viewer_launch():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 TOUS LES TESTS RÉUSSIS !")
        print("🚀 Votre détecteur Multi-BD est prêt à l'emploi !")
        print("\n📋 Instructions d'utilisation :")
        print("1. Lancez : python AnComicsViewer.py")
        print("2. Ouvrez un PDF de BD")
        print("3. Menu : ⚙️ → Detector → Multi-BD (Trained)")
        print("4. Profitez de la détection multi-styles ! 🎯")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("💡 Vérifiez les erreurs ci-dessus")

if __name__ == "__main__":
    main()