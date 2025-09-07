#!/usr/bin/env python3
"""
Test de détection parfaite avec configuration ultra-sensible
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

def test_perfect_detection(pdf_path, page_num=22):
    """Test de détection avec paramètres ultra-sensibles"""

    print("🧪 TEST DE DÉTECTION PARFAITE")
    print("=" * 50)
    print(f"📖 PDF: {pdf_path}")
    print(f"📄 Page: {page_num}")
    print()

    # Exécuter le test directement
    import subprocess

    cmd = [sys.executable, "test_detection_debug.py", pdf_path, str(page_num)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Return code: {result.returncode}")
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")

    print("\n" + "=" * 50)
    print("🎯 ANALYSE DES RÉSULTATS:")
    print("Si vous voyez beaucoup de panels détectés (>15-20), c'est BON!")
    print("Si vous voyez peu de panels (<5), il y a un problème de configuration.")
    print("=" * 50)

if __name__ == "__main__":
    # Test avec Tintin page 22
    pdf_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf"
    test_perfect_detection(pdf_path, 22)
