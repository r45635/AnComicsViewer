#!/usr/bin/env python3
"""
Script de test pour analyser la pipeline de détection avec debug détaillé
"""

import sys
import os
sys.path.insert(0, '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch')

from PySide6.QtWidgets import QApplication
from main import PdfYoloViewer, enable_detection_debug
import fitz

def test_detection_debug(pdf_path: str, page_num: int = 0):
    """Test la détection avec debug activé"""

    print("🧪 TEST DE DÉTECTION AVEC DEBUG")
    print("=" * 50)

    # S'assurer que nous sommes dans le bon répertoire
    os.chdir('/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch')
    print(f"📁 Répertoire de travail: {os.getcwd()}")

    # Activer le debug
    enable_detection_debug(True)

    # Créer l'application
    app = QApplication(sys.argv)

    try:
        # Créer l'instance
        viewer = PdfYoloViewer()

        # Charger le PDF
        if os.path.exists(pdf_path):
            print(f"📖 Chargement du PDF: {pdf_path}")
            viewer.pdf = fitz.open(pdf_path)
            viewer.page_index = page_num

            # Charger la page
            viewer.load_page(page_num)

            print("\n✅ Test terminé - vérifiez les logs de debug ci-dessus")
            print(f"📊 Page {page_num + 1} analysée avec {len(viewer.dets)} détections finales")

        else:
            print(f"❌ PDF non trouvé: {pdf_path}")

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        app.quit()

def compare_with_previous_config(pdf_path: str, page_num: int = 0):
    """Compare les résultats avec et sans les nouveaux paramètres"""

    print("\n🔄 COMPARAISON AVEC CONFIGURATION PRÉCÉDENTE")
    print("=" * 50)

    # Test avec configuration actuelle
    print("\n📈 AVEC NOUVELLE CONFIGURATION:")
    test_detection_debug(pdf_path, page_num)

    # Désactiver le debug pour le prochain test
    enable_detection_debug(False)

    print("\n📉 AVEC CONFIGURATION DE BASE (simulation):")
    print("   ⚠️  Note: Pour une vraie comparaison, il faudrait charger l'ancienne config")
    print("   💡 Les seuils par défaut seraient plus élevés (panel_conf=0.18, balloon_conf=0.22)")

if __name__ == "__main__":
    # Chemin vers un PDF de test (à adapter selon vos besoins)
    test_pdf = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/test_comics.pdf"

    if len(sys.argv) > 1:
        test_pdf = sys.argv[1]

    page_to_test = 0
    if len(sys.argv) > 2:
        page_to_test = int(sys.argv[2])

    # Test principal
    test_detection_debug(test_pdf, page_to_test)

    # Comparaison (optionnel)
    # compare_with_previous_config(test_pdf, page_to_test)
