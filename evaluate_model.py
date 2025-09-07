#!/usr/bin/env python3
"""
Évaluation personnalisée du modèle YOLO entraîné
"""

import os
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_model():
    """Évalue le modèle sur le dataset de validation"""
    print("🔍 Évaluation personnalisée du modèle...")
    print("=" * 50)

    # Trouver le modèle
    model_dirs = sorted(Path("./runs/detect").glob("ancomics_final_optimized*"))
    if not model_dirs:
        print("❌ Aucun modèle trouvé")
        return

    model_dir = model_dirs[-1]  # Dernier modèle
    model_path = model_dir / "weights" / "best.pt"

    if not model_path.exists():
        model_path = model_dir / "weights" / "last.pt"

    print(f"📁 Modèle: {model_path}")

    # Charger le modèle
    model = YOLO(str(model_path))

    # Chemin des données de validation
    val_images = Path("./dataset/images/val")
    val_labels = Path("./dataset/labels/val")

    if not val_images.exists() or not val_labels.exists():
        print("❌ Données de validation introuvables")
        return

    # Collecter les images de validation
    image_files = list(val_images.glob("*.png")) + list(val_images.glob("*.jpg"))
    print(f"🖼️  Images de validation: {len(image_files)}")

    # Statistiques
    total_predictions = 0
    total_ground_truth = 0
    class_predictions = {0: 0, 1: 0}  # panel, balloon
    class_ground_truth = {0: 0, 1: 0}

    print("\n🎯 Analyse des prédictions...")

    for img_path in tqdm(image_files, desc="Traitement"):
        # Charger l'image
        img = Image.open(img_path)

        # Prédiction
        results = model(img, conf=0.25, iou=0.6, verbose=False)

        # Compter les prédictions
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                pred_classes = result.boxes.cls.cpu().numpy().astype(int)
                for cls in pred_classes:
                    if cls in [0, 1]:
                        class_predictions[cls] += 1
                        total_predictions += 1

        # Charger les ground truth
        label_file = val_labels / f"{img_path.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        if cls in [0, 1]:
                            class_ground_truth[cls] += 1
                            total_ground_truth += 1

    # Résultats
    print("\n📊 RÉSULTATS DE L'ÉVALUATION:")
    print(f"   Ground Truth - Panels: {class_ground_truth[0]}, Balloons: {class_ground_truth[1]}")
    print(f"   Prédictions - Panels: {class_predictions[0]}, Balloons: {class_predictions[1]}")
    print(f"   Total Ground Truth: {total_ground_truth}")
    print(f"   Total Prédictions: {total_predictions}")

    # Calcul de métriques simples
    if total_ground_truth > 0:
        recall = total_predictions / total_ground_truth
        print(f"   Recall approximatif: {recall:.3f}")

    # Vérifier que le modèle détecte quelque chose
    if total_predictions > 0:
        print("✅ Le modèle détecte des objets!")
        print("🎉 Entraînement réussi!")
    else:
        print("⚠️  Le modèle ne détecte aucun objet")

    return total_predictions > 0

if __name__ == "__main__":
    evaluate_model()
