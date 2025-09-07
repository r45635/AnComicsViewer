#!/usr/bin/env python3
"""
Test avec une vraie image de BD pour vérifier le modèle
"""

import sys
import os
import numpy as np

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_with_real_comic():
    """Test avec une vraie image de BD"""
    print("📰 TEST AVEC UNE VRAIE IMAGE DE BD")

    try:
        from ultralytics import YOLO
        from PySide6.QtGui import QImage
        import fitz
        print("✅ Imports réussis")
    except Exception as e:
        print(f"❌ Erreur imports: {e}")
        return

    # Charger le modèle
    model_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/multibd_enhanced_v2.pt"
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return

    try:
        model = YOLO(model_path)
        print("✅ Modèle chargé")
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        return

    # Charger une vraie image de BD
    pdf_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF non trouvé: {pdf_path}")
        return

    try:
        doc = fitz.open(pdf_path)
        page = doc[0]  # Première page

        # Convertir en image haute résolution
        zoom = 2.0  # 2x zoom pour meilleure qualité
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

        # Convertir en numpy array
        img_data = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img_data.reshape(pix.height, pix.width, 3)

        print(f"✅ Image chargée: {img.shape}")

        # Test de prédiction avec différents seuils
        thresholds = [0.001, 0.01, 0.05, 0.1, 0.2]

        for conf_thresh in thresholds:
            print(f"\n🎯 Test avec conf={conf_thresh}")

            try:
                results = model.predict(
                    source=img,
                    conf=conf_thresh,
                    iou=0.1,
                    max_det=100,
                    verbose=False
                )

                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        print(f"   📊 Détections: {len(boxes)}")

                        if len(boxes) > 0:
                            # Compter par classe
                            panel_count = 0
                            balloon_count = 0

                            for cls, conf in zip(boxes.cls, boxes.conf):
                                if int(cls) == 0:
                                    panel_count += 1
                                elif int(cls) == 1:
                                    balloon_count += 1

                            print(f"   📦 Panels: {panel_count}")
                            print(f"   💬 Balloons: {balloon_count}")

                            # Afficher les 3 meilleures détections
                            if len(boxes) > 0:
                                print("   🔝 Top 3 détections:")
                                # Trier par confiance décroissante
                                indices = np.argsort(boxes.conf)[::-1]
                                for i in range(min(3, len(indices))):
                                    idx = indices[i]
                                    xyxy = boxes.xyxy[idx]
                                    cls = int(boxes.cls[idx])
                                    conf = boxes.conf[idx]
                                    x1, y1, x2, y2 = xyxy
                                    print(".3f"
                                          f"      Classe: {'PANEL' if cls==0 else 'BALLOON'}")
                        else:
                            print("   ⚠️  Aucune détection")
                    else:
                        print("   ❌ Pas de boxes")
                else:
                    print("   ❌ Aucune prédiction")

            except Exception as e:
                print(f"   ❌ Erreur prédiction: {e}")

    except Exception as e:
        print(f"❌ Erreur traitement PDF: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ Test terminé")

if __name__ == "__main__":
    test_with_real_comic()
