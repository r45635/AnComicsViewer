#!/usr/bin/env python3
"""
Démonstration complète du détecteur Multi-BD intégré.
Lance le viewer avec le nouveau détecteur activé automatiquement.
"""

import sys
import os
from pathlib import Path

# Ajouter le patch PyTorch dès le début
exec(open('patch_pytorch.py').read())

def setup_multibd_demo():
    """Prépare l'environnement pour la démo Multi-BD."""
    
    print("🎬 PRÉPARATION DÉMO MULTI-BD")
    print("=" * 35)
    
    # Vérifier que le modèle existe
    model_path = "runs/detect/multibd_mixed_model/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Modèle non trouvé : {model_path}")
        print("💡 Exécutez d'abord : python train_multibd_model.py")
        return False
    
    print(f"✅ Modèle Multi-BD trouvé")
    
    # Vérifier les dépendances
    try:
        from detectors.multibd_detector import MultiBDPanelDetector
        from PySide6.QtWidgets import QApplication
        print("✅ Dépendances OK")
    except Exception as e:
        print(f"❌ Dépendances manquantes : {e}")
        return False
    
    # Tester le modèle
    try:
        detector = MultiBDPanelDetector()
        info = detector.get_model_info()
        print(f"📊 Performance : {info['performance']['mAP50']} mAP50")
        print(f"🎯 Styles supportés : {len(info['training_data'])} ({', '.join(info['training_data'])})")
    except Exception as e:
        print(f"❌ Erreur modèle : {e}")
        return False
    
    return True

def launch_viewer_with_multibd():
    """Lance le viewer avec le détecteur Multi-BD."""
    
    print(f"\n🚀 LANCEMENT VIEWER MULTI-BD")
    print("=" * 35)
    
    try:
        # Import du viewer
        from AnComicsViewer import main as viewer_main
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        import AnComicsViewer
        
        # Créer l'application
        app = QApplication(sys.argv)
        app.setApplicationName("AnComicsViewer Multi-BD")
        app.setOrganizationName("AnComics")
        
        print("✅ Application créée")
        
        # Créer la fenêtre principale
        window = AnComicsViewer.ComicsView()
        
        # Activer automatiquement le détecteur Multi-BD
        try:
            from detectors.multibd_detector import MultiBDPanelDetector
            window._panel_detector = MultiBDPanelDetector()
            print("✅ Détecteur Multi-BD activé automatiquement")
            
            # Mettre à jour l'interface si possible
            window.setWindowTitle("AnComicsViewer - Multi-BD Detector")
            
        except Exception as e:
            print(f"⚠️  Impossible d'activer Auto Multi-BD : {e}")
            print("💡 Utilisez le menu ⚙️ → Detector → Multi-BD (Trained)")
        
        # Afficher la fenêtre
        window.show()
        window.raise_()
        window.activateWindow()
        
        print(f"\n🎉 VIEWER LANCÉ AVEC SUCCÈS !")
        print("=" * 40)
        print("📖 Instructions :")
        print("1. 📁 Ouvrez un PDF de BD (File → Open)")
        print("2. ⚙️ Menu → Detector → Multi-BD (Trained)")
        print("3. 🎯 Profitez de la détection multi-styles !")
        print()
        print("🔧 Fonctionnalités disponibles :")
        print("   • Détection Golden City (complexe)")
        print("   • Détection Tintin (simple)")  
        print("   • Détection Pin-up du B24 (aviation)")
        print("   • Navigation panel par panel")
        print("   • Réglage de la confiance")
        print()
        print("💡 Testez avec différents styles de BD !")
        
        # Lancer la boucle d'événements
        return app.exec()
        
    except Exception as e:
        print(f"❌ Erreur lancement viewer : {e}")
        return 1

def show_demo_info():
    """Affiche les informations de la démo."""
    
    print("🎯 DÉMONSTRATION ANCOMICSVIEWER MULTI-BD")
    print("=" * 50)
    print()
    print("🎪 Cette démo présente :")
    print("   • Détecteur YOLO entraîné sur 3 styles de BD")
    print("   • Performance : 91.1% mAP50, 88.3% mAP50-95")
    print("   • Support multi-styles : Golden City, Tintin, Pin-up du B24")
    print("   • Interface intégrée avec basculement facile")
    print()
    print("📚 Dataset d'entraînement :")
    print("   • 50 images annotées (160 images totales)")
    print("   • 377 annotations de panels")
    print("   • Classes : panel, panel_inset")
    print()
    print("🚀 Prêt pour le lancement...")
    print()

def main():
    """Fonction principale de la démo."""
    
    show_demo_info()
    
    # Préparer l'environnement
    if not setup_multibd_demo():
        print("\n❌ Échec préparation démo")
        sys.exit(1)
    
    # Lancer le viewer
    exit_code = launch_viewer_with_multibd()
    
    print(f"\n👋 Fin de la démo Multi-BD")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
