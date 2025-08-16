#!/usr/bin/env python3
"""
Test avec fusion des classes panel et panel_inset
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import shutil

# Appliquer le patch PyTorch dès le début
exec(open('patch_pytorch.py').read())

def create_single_class_dataset():
    """Crée un dataset avec une seule classe 'panel'."""
    
    print("🔄 CRÉATION DATASET CLASSE UNIQUE")
    print("=" * 40)
    
    # Créer le dossier de sortie
    output_dir = Path("dataset/yolo_single_class")
    output_dir.mkdir(exist_ok=True)
    
    # Copier les images
    print("📁 Copie des images...")
    shutil.copytree("dataset/yolo/images", output_dir / "images", dirs_exist_ok=True)
    
    # Créer les labels avec classe unique
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    
    for split in ["train", "val"]:
        split_dir = labels_dir / split
        split_dir.mkdir(exist_ok=True)
        
        source_dir = Path(f"dataset/yolo/labels/{split}")
        if source_dir.exists():
            for label_file in source_dir.glob("*.txt"):
                # Lire et convertir les labels (0 et 1 → 0)
                with open(label_file) as f:
                    lines = f.readlines()
                
                converted_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        # Convertir classe 1 → 0 (tout devient 'panel')
                        parts[0] = "0"
                        converted_lines.append(" ".join(parts) + "\n")
                
                # Écrire le fichier converti
                output_file = split_dir / label_file.name
                with open(output_file, "w") as f:
                    f.writelines(converted_lines)
    
    # Créer le data.yaml
    data_yaml = output_dir / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"""
train: {output_dir.absolute()}/images/train
val: {output_dir.absolute()}/images/val
nc: 1
names: ['panel']
""")
    
    print(f"✅ Dataset classe unique créé: {output_dir}")
    
    # Statistiques
    train_labels = list((labels_dir / "train").glob("*.txt"))
    val_labels = list((labels_dir / "val").glob("*.txt"))
    
    total_annotations = 0
    for label_file in train_labels + val_labels:
        with open(label_file) as f:
            total_annotations += len(f.readlines())
    
    print(f"📊 Statistiques:")
    print(f"   Images train: {len(list((output_dir / 'images/train').glob('*.*')))}")
    print(f"   Images val: {len(list((output_dir / 'images/val').glob('*.*')))}")
    print(f"   Labels train: {len(train_labels)}")
    print(f"   Labels val: {len(val_labels)}")
    print(f"   Annotations totales: {total_annotations}")
    
    return str(data_yaml)

def train_single_class_model(data_yaml):
    """Entraîne un modèle avec une seule classe."""
    
    print("\n🚀 ENTRAÎNEMENT MODÈLE CLASSE UNIQUE")
    print("=" * 40)
    
    try:
        # Charger le modèle
        model = YOLO('yolov8n.pt')
        
        # Entraîner
        results = model.train(
            data=data_yaml,
            epochs=40,  # Moins d'époques pour tester
            batch=4,
            lr0=0.003,
            name="single_class_panels",
            patience=10,
            save_period=10,
            # Augmentations légères
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,
            degrees=0.0,
            translate=0.05,
            scale=0.3,
            fliplr=0.5
        )
        
        print("✅ Entraînement terminé!")
        print(f"📁 Résultats: runs/detect/single_class_panels/")
        return True
        
    except Exception as e:
        print(f"❌ Erreur entraînement: {e}")
        return False

def test_single_class_model():
    """Test le modèle classe unique."""
    
    print("\n🧪 TEST MODÈLE CLASSE UNIQUE")
    print("=" * 30)
    
    model_path = "runs/detect/single_class_panels/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    model = YOLO(model_path)
    
    # Images de test
    test_images = [
        ("dataset/images/train/p0003.png", "Golden City"),
        ("dataset/images/train/tintin_p0001.png", "Tintin")
    ]
    
    for img_path, style in test_images:
        if not Path(img_path).exists():
            continue
            
        print(f"\n📸 {style} - {Path(img_path).name}")
        
        # Test avec différents seuils
        for conf in [0.1, 0.2, 0.3]:
            results = model.predict(img_path, conf=conf, verbose=False)
            panels = len(results[0].boxes) if results[0].boxes is not None else 0
            print(f"   Conf {conf}: {panels} panels")

if __name__ == "__main__":
    print("🎯 TEST MODÈLE CLASSE UNIQUE")
    print("=" * 50)
    
    # 1. Créer le dataset
    data_yaml = create_single_class_dataset()
    
    # 2. Entraîner
    if train_single_class_model(data_yaml):
        # 3. Tester
        test_single_class_model()
    
    print("\n✅ Test terminé!")
