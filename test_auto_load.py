#!/usr/bin/env python3
"""
Test rapide du chargement automatique du modèle amélioré
"""

import sys
import os
sys.path.insert(0, '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch')

from PySide6.QtWidgets import QApplication
from main import PdfYoloViewer

def test_auto_load():
    """Test du chargement automatique du modèle amélioré"""

    print("🧪 TEST DU CHARGEMENT AUTOMATIQUE")
    print("=" * 40)

    # Créer une application Qt (nécessaire pour PySide6)
    app = QApplication(sys.argv)

    try:
        # Créer l'instance de l'application
        viewer = PdfYoloViewer()

        # Vérifier si le modèle a été chargé
        if hasattr(viewer, 'model') and viewer.model is not None:
            model_path = getattr(viewer.model, 'ckpt_path', 'Unknown')
            print("✅ Modèle chargé automatiquement !")
            print(f"   📁 Chemin: {model_path}")
            print(f"   🎯 Classes: {viewer.model.names if hasattr(viewer.model, 'names') else 'N/A'}")

            # Vérifier que c'est bien le modèle amélioré
            if 'multibd_enhanced_v2.pt' in str(model_path):
                print("🎉 SUCCÈS: Le modèle amélioré est chargé automatiquement !")
                return True
            else:
                print("⚠️  ATTENTION: Un autre modèle a été chargé")
                return False
        else:
            print("❌ ÉCHEC: Aucun modèle n'a été chargé")
            return False

    except Exception as e:
        print(f"❌ ERREUR lors du test: {e}")
        return False

    finally:
        # Fermer proprement l'application
        app.quit()

if __name__ == "__main__":
    success = test_auto_load()
    print(f"\n📊 RÉSULTAT: {'RÉUSSI' if success else 'ÉCHEC'}")
    sys.exit(0 if success else 1)
