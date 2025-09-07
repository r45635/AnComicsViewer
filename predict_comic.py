#!/usr/bin/env python3
"""
Script de prédiction pour la détection de panels et balloons dans les comics
"""

import sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def predict_comic(image_path, conf_threshold=0.25, save_results=True):
    """
    Prédit les panels et balloons dans une image de comic

    Args:
        image_path (str): Chemin vers l'image
        conf_threshold (float): Seuil de confiance pour les prédictions
        save_results (bool): Sauvegarder les résultats
    """
    print("🎯 Détection de panels et balloons...")
    print("=" * 50)

    # Vérifier que l'image existe
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"❌ Image introuvable: {image_path}")
        return None

    # Trouver le modèle entraîné
    model_dirs = sorted(Path("./runs/detect").glob("ancomics_final_optimized*"))
    if not model_dirs:
        print("❌ Aucun modèle trouvé")
        return None

    model_dir = model_dirs[-1]
    model_path = model_dir / "weights" / "best.pt"

    if not model_path.exists():
        model_path = model_dir / "weights" / "last.pt"

    if not model_path.exists():
        print("❌ Modèle introuvable")
        return None

    print(f"📁 Modèle: {model_path}")
    print(f"🖼️  Image: {img_path}")

    # Charger le modèle
    model = YOLO(str(model_path))

    # Faire la prédiction
    results = model(img_path, conf=conf_threshold, iou=0.6, verbose=True)

    if not results or len(results) == 0:
        print("❌ Aucune prédiction")
        return None

    result = results[0]

    # Afficher les résultats
    print("\n📊 RÉSULTATS:")
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        class_names = {0: "Panel", 1: "Balloon"}

        for i, (cls, conf) in enumerate(zip(boxes.cls, boxes.conf)):
            class_name = class_names.get(int(cls), f"Classe {int(cls)}")
            confidence = conf.item() * 100
            print(f"   {i+1}. {class_name} - Confiance: {confidence:.1f}%")

        print(f"\n🔢 Total détecté: {len(boxes)} objets")
    else:
        print("   Aucun objet détecté")

    # Sauvegarder les résultats si demandé
    if save_results and result.boxes is not None:
        output_dir = Path("./predictions")
        output_dir.mkdir(exist_ok=True)

        # Sauvegarder l'image avec les détections
        result.save(str(output_dir / f"{img_path.stem}_detected{img_path.suffix}"))
        print(f"\n💾 Résultats sauvegardés dans: {output_dir}")

    return result

def main():
    """Fonction principale"""
    if len(sys.argv) < 2:
        print("Usage: python3 predict_comic.py <image_path> [confidence_threshold]")
        print("Exemple: python3 predict_comic.py ./dataset/images/val/p0001.png 0.3")
        sys.exit(1)

    image_path = sys.argv[1]
    conf_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

    predict_comic(image_path, conf_threshold)

if __name__ == "__main__":
    main()
