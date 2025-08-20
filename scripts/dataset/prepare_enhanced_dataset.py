#!/usr/bin/env python3
"""
Préparation et analyse du dataset Multi-BD Enhanced
Analyse les nouvelles annotations et prépare l'entraînement
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
import sys

def analyze_dataset():
    """Analyse le dataset actuel."""
    print("🔍 Analyse du Dataset Multi-BD Enhanced")
    print("=" * 50)
    
    base_dir = Path("dataset")
    labels_dir = base_dir / "labels"
    images_dir = base_dir / "images"
    
    # Compter les annotations par série
    series_stats = {}
    total_annotations = 0
    total_panels = 0
    
    for split in ["train", "val"]:
        split_dir = labels_dir / split
        if not split_dir.exists():
            continue
            
        print(f"\n📁 {split.upper()}:")
        
        for json_file in split_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Détecter la série
                filename = json_file.stem
                if "tintin" in filename.lower():
                    series = "Tintin"
                elif any(x in filename.lower() for x in ["golden", "city"]):
                    series = "Golden City" 
                elif "pinup" in filename.lower():
                    series = "Pin-up du B24"
                else:
                    series = "Autres"
                
                if series not in series_stats:
                    series_stats[series] = {"files": 0, "panels": 0}
                
                # Compter les panels
                panel_count = 0
                for shape in data.get("shapes", []):
                    if shape.get("label") == "panel":
                        panel_count += 1
                
                series_stats[series]["files"] += 1
                series_stats[series]["panels"] += panel_count
                total_annotations += 1
                total_panels += panel_count
                
                print(f"   📄 {filename}: {panel_count} panels")
                
            except Exception as e:
                print(f"   ❌ Erreur {json_file}: {e}")
    
    print(f"\n📊 Statistiques Globales:")
    print(f"   • Total annotations: {total_annotations}")
    print(f"   • Total panels: {total_panels}")
    print(f"   • Moyenne panels/page: {total_panels/max(1,total_annotations):.1f}")
    
    print(f"\n📚 Répartition par Série:")
    for series, stats in series_stats.items():
        print(f"   • {series}: {stats['files']} fichiers, {stats['panels']} panels")
    
    return series_stats, total_annotations, total_panels

def split_dataset(train_ratio: float = 0.8):
    """Répartit le dataset entre train et validation."""
    print(f"\n🔄 Répartition Dataset (train: {train_ratio*100:.0f}%, val: {(1-train_ratio)*100:.0f}%)")
    print("=" * 60)
    
    base_dir = Path("dataset")
    labels_train = base_dir / "labels" / "train" 
    labels_val = base_dir / "labels" / "val"
    images_train = base_dir / "images" / "train"
    images_val = base_dir / "images" / "val"
    
    # Créer les dossiers val si nécessaire
    labels_val.mkdir(exist_ok=True)
    images_val.mkdir(exist_ok=True)
    
    # Récupérer tous les fichiers d'annotations
    all_annotations = list(labels_train.glob("*.json"))
    
    if not all_annotations:
        print("❌ Aucune annotation trouvée dans dataset/labels/train/")
        return False
    
    # Mélanger aléatoirement
    random.seed(42)  # Pour reproductibilité
    random.shuffle(all_annotations)
    
    # Calculer la répartition
    n_train = int(len(all_annotations) * train_ratio)
    train_files = all_annotations[:n_train]
    val_files = all_annotations[n_train:]
    
    print(f"📂 Répartition:")
    print(f"   • Train: {len(train_files)} fichiers")
    print(f"   • Val: {len(val_files)} fichiers")
    
    # Déplacer les fichiers de validation
    moved_count = 0
    for json_file in val_files:
        try:
            # Déplacer l'annotation
            val_json = labels_val / json_file.name
            shutil.move(str(json_file), str(val_json))
            
            # Trouver et déplacer l'image correspondante
            stem = json_file.stem
            for ext in [".png", ".jpg", ".jpeg"]:
                img_file = images_train / f"{stem}{ext}"
                if img_file.exists():
                    val_img = images_val / f"{stem}{ext}"
                    shutil.move(str(img_file), str(val_img))
                    break
            
            moved_count += 1
            
        except Exception as e:
            print(f"⚠️  Erreur déplacement {json_file.name}: {e}")
    
    print(f"✅ {moved_count} fichiers déplacés vers validation")
    return True

def convert_to_yolo():
    """Convertit les annotations LabelMe en format YOLO."""
    print(f"\n🔄 Conversion vers format YOLO")
    print("=" * 40)
    
    # Utiliser le script existant
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "tools/labelme_to_yolo.py"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ Conversion YOLO réussie")
            print(result.stdout)
        else:
            print("❌ Erreur conversion YOLO")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Erreur script conversion: {e}")
        return False
    
    return True

def prepare_training_config():
    """Prépare la configuration d'entraînement."""
    print(f"\n⚙️  Préparation Configuration Entraînement")
    print("=" * 50)
    
    # Vérifier que les dossiers YOLO existent
    yolo_dir = Path("dataset/yolo")
    if not yolo_dir.exists():
        print("❌ Dossier dataset/yolo manquant - conversion YOLO requise")
        return False
    
    # Compter les images train/val
    train_imgs = len(list((yolo_dir / "images" / "train").glob("*")))
    val_imgs = len(list((yolo_dir / "images" / "val").glob("*")))
    train_labels = len(list((yolo_dir / "labels" / "train").glob("*.txt")))
    val_labels = len(list((yolo_dir / "labels" / "val").glob("*.txt")))
    
    print(f"📊 Dataset YOLO préparé:")
    print(f"   • Train: {train_imgs} images, {train_labels} labels")
    print(f"   • Val: {val_imgs} images, {val_labels} labels")
    
    if train_imgs != train_labels or val_imgs != val_labels:
        print("⚠️  Incohérence images/labels détectée")
        return False
    
    # Créer la config YOLO mise à jour
    config_content = f"""# Multi-BD Enhanced Dataset Configuration
# Generated on {Path().absolute()}

path: {Path().absolute() / "dataset" / "yolo"}
train: images/train
val: images/val

# Classes
names:
  0: panel

# Dataset info
nc: 1  # number of classes
total_images: {train_imgs + val_imgs}
train_images: {train_imgs}
val_images: {val_imgs}

# Training hyperparameters (optimized for panel detection)
# Use with: yolo train data=this_config.yaml model=yolov8n.pt epochs=100
"""
    
    config_file = Path("dataset/multibd_enhanced.yaml")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Configuration sauvée: {config_file}")
    return True

def main():
    """Processus complet de préparation du dataset."""
    print("🚀 Préparation Dataset Multi-BD Enhanced")
    print("=" * 60)
    
    # 1. Analyser le dataset actuel
    series_stats, total_annotations, total_panels = analyze_dataset()
    
    if total_annotations == 0:
        print("❌ Aucune annotation trouvée - Vérifiez dataset/labels/train/")
        return False
    
    # 2. Répartir train/val si nécessaire
    val_count = len(list(Path("dataset/labels/val").glob("*.json")))
    if val_count == 0 and total_annotations >= 10:
        print("\n⚠️  Aucun fichier de validation - Répartition automatique")
        if not split_dataset(train_ratio=0.8):
            return False
    elif val_count == 0:
        print(f"\n⚠️  Dataset trop petit ({total_annotations} annotations) - Pas de split val")
    
    # 3. Conversion YOLO
    if not convert_to_yolo():
        return False
    
    # 4. Préparer la configuration
    if not prepare_training_config():
        return False
    
    print(f"\n🎉 Dataset Multi-BD Enhanced prêt pour l'entraînement!")
    print(f"📋 Commande d'entraînement:")
    print(f"   cd /Users/vincentcruvellier/Documents/GitHub/AnComicsViewer")
    print(f"   .venv/bin/python -m ultralytics.YOLO train \\")
    print(f"       data=dataset/multibd_enhanced.yaml \\")
    print(f"       model=yolov8n.pt \\")
    print(f"       epochs=150 \\")
    print(f"       imgsz=640 \\")
    print(f"       batch=16 \\")
    print(f"       name=multibd_enhanced_v2")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
