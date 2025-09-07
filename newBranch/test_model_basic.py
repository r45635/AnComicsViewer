#!/usr/bin/env python3
"""
Test simple pour vérifier le fonctionnement de base du modèle YOLO
"""

import sys
import os
import numpy as np

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_basic_model():
    """Test basique du modèle YOLO"""
    print("🔍 TEST BASIQUE DU MODÈLE YOLO")

    try:
        from ultralytics import YOLO
        print("✅ Ultralytics importé avec succès")
    except Exception as e:
        print(f"❌ Erreur import ultralytics: {e}")
        return

    # Tester le chargement du modèle
    model_paths = [
        "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/multibd_enhanced_v2.pt",
        "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/anComicsViewer_v01.pt",
        "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/anComicsViewer_v01.pt"
    ]

    model = None
    for path in model_paths:
        if os.path.exists(path):
            try:
                print(f"📂 Tentative de chargement: {path}")
                model = YOLO(path)
                print(f"✅ Modèle chargé: {os.path.basename(path)}")
                break
            except Exception as e:
                print(f"❌ Erreur chargement {path}: {e}")
                continue

    if model is None:
        print("❌ Aucun modèle n'a pu être chargé")
        return

    # Tester une prédiction simple
    print("\n🧪 TEST DE PRÉDICTION")

    # Créer une image de test simple (rectangle noir avec un rectangle blanc)
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Dessiner un rectangle blanc (simulant un panel)
    test_img[100:300, 100:500] = 255

    try:
        results = model.predict(
            source=test_img,
            conf=0.01,  # Très bas pour voir toutes les détections
            iou=0.1,
            max_det=100,
            verbose=False
        )

        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"📊 Nombre de détections: {len(boxes)}")

                if len(boxes) > 0:
                    print("🔍 DÉTECTIONS TROUVÉES:")
                    for i, (xyxy, cls, conf) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
                        x1, y1, x2, y2 = xyxy
                        print(".2f"
                              f"     Classe: {int(cls)} ({'panel' if int(cls)==0 else 'balloon'})")
                        print(".2f"
                              f"     Taille: {x2-x1:.0f} x {y2-y1:.0f} px")
                else:
                    print("⚠️  Aucune détection trouvée sur l'image de test")
            else:
                print("❌ Pas de boxes dans le résultat")
        else:
            print("❌ Aucune prédiction obtenue")

    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ Test terminé")

if __name__ == "__main__":
    test_basic_model()
