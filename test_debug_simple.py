#!/usr/bin/env python3
"""
Script simple pour tester le debug de détection
"""

import sys
import os

# Ajouter le répertoire au path
sys.path.insert(0, '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch')

def test_debug():
    """Test basique du debug"""
    print("🧪 Test du système de debug de détection")
    print("=" * 40)

    try:
        # Importer les fonctions de debug
        from main import debug_detection_stats, enable_detection_debug

        # Activer le debug
        enable_detection_debug(True)
        print("✅ Debug activé")

        # Tester la fonction de debug avec des données fictives
        from PySide6.QtCore import QRectF

        panels = [(0, 0.8, QRectF(10, 10, 100, 200))]
        balloons = [(1, 0.9, QRectF(50, 50, 50, 30))]

        debug_detection_stats("TEST", panels, balloons, 100000)

        print("\n✅ Test du debug terminé avec succès!")

    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("💡 Assurez-vous que le fichier main.py est dans le bon répertoire")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug()
