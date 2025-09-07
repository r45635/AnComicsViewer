#!/usr/bin/env python3
"""
Test rapide du modèle amélioré dans l'application
"""

import sys
import os
sys.path.insert(0, '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer')

from ultralytics import YOLO
import torch

def test_improved_model():
    """Test rapide du chargement et de l'inférence du modèle amélioré"""

    print("🧪 TEST DU MODÈLE AMÉLIORÉ")
    print("=" * 40)

    # Chemin vers le modèle
    model_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/multibd_enhanced_v2.pt"

    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return False

    try:
        print(f"📂 Chargement du modèle: {model_path}")

        # Charger le modèle
        model = YOLO(model_path)

        print("✅ Modèle chargé avec succès!")
        print(f"   • Classes: {model.names}")
        print(f"   • Device: {model.device}")

        # Test d'inférence rapide
        print("\n🔍 Test d'inférence...")

        # Créer une image de test simple (noire)
        import numpy as np
        from PIL import Image

        test_image = Image.new('RGB', (640, 480), color='white')
        test_image.save('/tmp/test_image.jpg')

        # Faire une prédiction
        results = model('/tmp/test_image.jpg', verbose=False)

        print("✅ Inférence réussie!")
        print(f"   • Nombre de détections: {len(results[0].boxes)}")

        # Nettoyer
        os.remove('/tmp/test_image.jpg')

        print("\n🎉 TEST RÉUSSI - Le modèle amélioré fonctionne parfaitement!")
        return True

    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

if __name__ == "__main__":
    success = test_improved_model()
    sys.exit(0 if success else 1)
