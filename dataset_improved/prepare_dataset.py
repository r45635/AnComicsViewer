#!/usr/bin/env python3
"""
Script de conversion JSON vers YOLO et augmentation des données
"""

import sys
import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

def json_to_yolo():
    """Convertit les annotations JSON vers le format YOLO"""

    print("🔄 CONVERSION JSON → YOLO")
    print("=" * 50)

    source_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/backup_annotations_20250822_182146"
    target_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset_improved"

    # Classes YOLO
    classes = ["panel", "balloon"]
    class_mapping = {cls: i for i, cls in enumerate(classes)}

    converted_count = 0

    # Lister tous les fichiers JSON
    json_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]

    print(f"📁 {len(json_files)} fichiers JSON à convertir")

    for json_file in json_files:
        try:
            json_path = os.path.join(source_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extraire les dimensions de l'image (si disponibles)
            # Pour l'instant, on utilise une taille par défaut
            img_width, img_height = 1700, 2200  # Taille typique des pages

            # Fichier de labels YOLO
            label_file = json_file.replace('.json', '.txt')
            label_path = os.path.join(target_dir, 'labels', label_file)

            with open(label_path, 'w') as f:
                for shape in data['shapes']:
                    label = shape['label']
                    if label not in class_mapping:
                        continue

                    class_id = class_mapping[label]
                    points = shape['points']

                    # Convertir les points en format YOLO (x_center, y_center, width, height normalisés)
                    x1, y1 = points[0]
                    x2, y2 = points[1]

                    # Calculer le centre et les dimensions
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    # Écrire la ligne YOLO
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            converted_count += 1

            if converted_count % 20 == 0:
                print(f"   ✅ {converted_count}/{len(json_files)} fichiers convertis")

        except Exception as e:
            print(f"   ❌ Erreur avec {json_file}: {e}")

    print(f"\n🎉 Conversion terminée: {converted_count} fichiers convertis")

    # Créer le fichier data.yaml pour YOLO
    create_data_yaml(target_dir, classes)

def create_data_yaml(target_dir, classes):
    """Crée le fichier de configuration data.yaml pour YOLO"""

    yaml_content = f"""# Dataset configuration for YOLO training
path: {target_dir}
train: images
val: images  # Pour l'instant, même dossier (à splitter plus tard)

# Classes
names:
"""

    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\n"

    yaml_path = os.path.join(target_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"📄 Fichier data.yaml créé: {yaml_path}")

def augment_difficult_pages():
    """Augmente les pages difficiles (pages 1 et 6 de Pin-up)"""

    print("\n📈 AUGMENTATION DES PAGES DIFFICILES")
    print("=" * 50)

    source_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/backup_annotations_20250822_182146"
    target_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset_improved"

    # Pages problématiques à augmenter
    difficult_pages = [
        "pinup_p0001.json",
        "pinup_p0006.json"
    ]

    augmentations_per_page = 20  # 20 augmentations par page

    for page_json in difficult_pages:
        print(f"\n🔄 Augmentation de {page_json}")

        json_path = os.path.join(source_dir, page_json)
        if not os.path.exists(json_path):
            print(f"   ⚠️ Fichier non trouvé: {page_json}")
            continue

        # Pour l'instant, on ne peut pas créer d'images augmentées sans les images originales
        # On duplique juste les annotations avec des modifications mineures

        base_name = page_json.replace('.json', '')

        for i in range(augmentations_per_page):
            # Créer une version augmentée des annotations
            aug_json = f"{base_name}_aug{i:02d}.json"
            aug_txt = f"{base_name}_aug{i:02d}.txt"

            try:
                # Copier et modifier légèrement les annotations
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # Appliquer une petite variation aux boîtes (simuler rotation/échelle)
                for shape in data['shapes']:
                    if 'points' in shape:
                        points = shape['points']
                        # Petite variation aléatoire (±2%)
                        for j, (x, y) in enumerate(points):
                            variation_x = random.uniform(-0.02, 0.02)
                            variation_y = random.uniform(-0.02, 0.02)
                            points[j] = [x * (1 + variation_x), y * (1 + variation_y)]

                # Sauvegarder la version augmentée
                aug_json_path = os.path.join(target_dir, 'labels', aug_json)
                with open(aug_json_path, 'w') as f:
                    json.dump(data, f, indent=2)

                # Convertir en format YOLO
                convert_single_json_to_yolo(aug_json_path, os.path.join(target_dir, 'labels', aug_txt))

                print(f"   ✅ {aug_json} créé")

            except Exception as e:
                print(f"   ❌ Erreur avec {aug_json}: {e}")

def convert_single_json_to_yolo(json_path, txt_path):
    """Convertit un seul fichier JSON vers YOLO"""

    classes = ["panel", "balloon"]
    class_mapping = {cls: i for i, cls in enumerate(classes)}

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        img_width, img_height = 1700, 2200  # Taille par défaut

        with open(txt_path, 'w') as f:
            for shape in data['shapes']:
                label = shape['label']
                if label not in class_mapping:
                    continue

                class_id = class_mapping[label]
                points = shape['points']

                x1, y1 = points[0]
                x2, y2 = points[1]

                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    except Exception as e:
        print(f"Erreur conversion {json_path}: {e}")

def create_training_script():
    """Crée un script de lancement de l'entraînement"""

    script_content = '''#!/bin/bash
# Script d'entraînement YOLO avec dataset amélioré

echo "🚀 DÉMARRAGE DE L'ENTRAÎNEMENT YOLO"
echo "==================================="

# Vérifier que YOLO est installé
if ! command -v yolo &> /dev/null; then
    echo "❌ YOLO CLI non trouvé. Installer avec: pip install ultralytics"
    exit 1
fi

# Configuration
MODEL="yolov8m.pt"
DATA_FILE="dataset_improved/data.yaml"
EPOCHS=100
BATCH_SIZE=16
IMAGE_SIZE=640

echo "📊 Configuration:"
echo "   • Modèle: $MODEL"
echo "   • Dataset: $DATA_FILE"
echo "   • Epochs: $EPOCHS"
echo "   • Batch size: $BATCH_SIZE"
echo "   • Image size: $IMAGE_SIZE"

# Lancer l'entraînement
echo ""
echo "🎯 Lancement de l'entraînement..."
yolo train \\
    model=$MODEL \\
    data=$DATA_FILE \\
    epochs=$EPOCHS \\
    imgsz=$IMAGE_SIZE \\
    batch=$BATCH_SIZE \\
    name=ancomics_improved \\
    save=True \\
    save_period=10 \\
    cache=True \\
    workers=4 \\
    device=mps  # Pour Mac avec GPU

echo ""
echo "✅ Entraînement terminé!"
echo "📁 Résultats dans: runs/train/ancomics_improved/"
'''

    script_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset_improved/train.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    # Rendre le script exécutable
    os.chmod(script_path, 0o755)

    print("📜 Script d'entraînement créé: dataset_improved/train.sh")

def main():
    """Fonction principale"""
    print("🛠️ PRÉPARATION DU DATASET AMÉLIORÉ")
    print("=" * 60)

    # Étape 1: Conversion JSON → YOLO
    json_to_yolo()

    # Étape 2: Augmentation des pages difficiles
    augment_difficult_pages()

    # Étape 3: Créer le script d'entraînement
    create_training_script()

    print("\n🎉 PRÉPARATION TERMINÉE!")
    print("=" * 60)
    print("📁 Structure créée:")
    print("   • dataset_improved/images/     # Images (à ajouter manuellement)")
    print("   • dataset_improved/labels/     # Labels YOLO")
    print("   • dataset_improved/data.yaml   # Configuration YOLO")
    print("   • dataset_improved/train.sh    # Script d'entraînement")

    print("\n📊 Statistiques:")
    labels_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset_improved/labels"
    if os.path.exists(labels_dir):
        txt_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        print(f"   • Fichiers de labels: {len(txt_files)}")

    print("\n🚀 PROCHAINES ÉTAPES:")
    print("   1. Ajouter les images correspondantes dans dataset_improved/images/")
    print("   2. Lancer l'entraînement: ./dataset_improved/train.sh")
    print("   3. Valider les résultats sur toutes les pages")

if __name__ == "__main__":
    main()
