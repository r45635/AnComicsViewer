#!/usr/bin/env python3
"""
Test rapide du chargement automatique du modèle dans main.py
"""

import sys
import os
sys.path.append('.')

def test_model_loading():
    """Test rapide du chargement du modèle"""
    print("🧪 Test du chargement automatique du modèle...")

    # Simuler l'import de main.py
    try:
        # Importer les dépendances nécessaires
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        import numpy as np

        # Créer une application Qt minimale
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        # Tester l'import du module main
        from main import PdfYoloViewer

        # Créer une instance minimale pour tester le chargement du modèle
        viewer = PdfYoloViewer.__new__(PdfYoloViewer)
        viewer.status = type('MockStatus', (), {'showMessage': lambda x: print(f"Status: {x}")})()
        viewer.model_status = type('MockAction', (), {'setText': lambda x: print(f"Model status: {x}")})()

        # Tester le chargement automatique
        viewer._auto_load_model()

        if hasattr(viewer, 'model') and viewer.model is not None:
            print("✅ Modèle chargé avec succès !")
            print(f"   📊 Classes: {getattr(viewer.model, 'names', 'Unknown')}")
            return True
        else:
            print("❌ Échec du chargement du modèle")
            return False

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Le modèle entraîné sera chargé automatiquement !")
    else:
        print("\n⚠️  Problème avec le chargement automatique")
