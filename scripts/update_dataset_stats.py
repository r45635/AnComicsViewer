#!/usr/bin/env python3
"""
Script pour mettre à jour les statistiques du dataset Multi-BD
==============================================================

Met à jour automatiquement le fichier multibd_enhanced.yaml avec les
nouvelles statistiques après ajout de fichiers.

Usage:
    python scripts/update_dataset_stats.py

Auteur: Vincent Cruvellier
"""

import os
import sys
from pathlib import Path
import yaml

# Configuration
DATASET_DIR = Path(__file__).parent.parent / "dataset"
TRAIN_IMAGES_DIR = DATASET_DIR / "images" / "train"
VAL_IMAGES_DIR = DATASET_DIR / "images" / "val"
TRAIN_LABELS_DIR = DATASET_DIR / "labels" / "train"
VAL_LABELS_DIR = DATASET_DIR / "labels" / "val"
CONFIG_FILE = DATASET_DIR / "multibd_enhanced.yaml"

def count_files_by_extension(directory: Path, extension: str) -> int:
    """Compte les fichiers avec une extension donnée."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(f"*.{extension}")))

def count_annotations(directory: Path) -> int:
    """Compte le nombre total d'annotations dans tous les fichiers .txt."""
    total = 0
    if not directory.exists():
        return 0
    
    for txt_file in directory.glob("*.txt"):
        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                # Compter les lignes qui ne sont pas des commentaires ou vides
                annotations = [line.strip() for line in lines 
                             if line.strip() and not line.strip().startswith('#')]
                total += len(annotations)
        except Exception as e:
            print(f"⚠️ Erreur lecture {txt_file}: {e}")
    
    return total

def analyze_dataset_composition(directory: Path) -> dict:
    """Analyse la composition du dataset par préfixe."""
    composition = {}
    if not directory.exists():
        return composition
    
    for img_file in directory.glob("*.png"):
        # Extraire le préfixe (avant le premier '_' ou les 5 premiers caractères)
        name = img_file.stem
        if '_' in name:
            prefix = name.split('_')[0]
        else:
            prefix = name[:5] if len(name) >= 5 else name
        
        if prefix not in composition:
            composition[prefix] = 0
        composition[prefix] += 1
    
    return composition

def update_yaml_config():
    """Met à jour le fichier de configuration YAML."""
    # Compter les fichiers
    train_images = count_files_by_extension(TRAIN_IMAGES_DIR, "png")
    val_images = count_files_by_extension(VAL_IMAGES_DIR, "png")
    total_images = train_images + val_images
    
    # Compter les annotations
    train_annotations = count_annotations(TRAIN_LABELS_DIR)
    val_annotations = count_annotations(VAL_LABELS_DIR)
    total_annotations = train_annotations + val_annotations
    
    # Analyser la composition
    train_composition = analyze_dataset_composition(TRAIN_IMAGES_DIR)
    val_composition = analyze_dataset_composition(VAL_IMAGES_DIR)
    
    # Lire le fichier YAML existant ou créer un nouveau
    config = {
        'path': str(DATASET_DIR / "yolo"),
        'train': "images/train",
        'val': "images/val",
        'names': {0: 'panel', 1: 'panel_inset'},
        'nc': 2
    }
    
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                existing_config = yaml.safe_load(f)
                if existing_config:
                    config.update(existing_config)
        except Exception as e:
            print(f"⚠️ Erreur lecture YAML existant: {e}")
    
    # Mettre à jour les statistiques
    config.update({
        'total_images': total_images,
        'train_images': train_images,
        'val_images': val_images,
        'total_annotations': total_annotations,
        'train_annotations': train_annotations,
        'val_annotations': val_annotations
    })
    
    # Sauvegarder
    with open(CONFIG_FILE, 'w') as f:
        # En-tête
        f.write("# Multi-BD Enhanced Dataset Configuration\n")
        f.write(f"# Generated on {DATASET_DIR}\n\n")
        
        # Configuration de base
        f.write(f"path: {config['path']}\n")
        f.write(f"train: {config['train']}\n")
        f.write(f"val: {config['val']}\n\n")
        
        # Classes
        f.write("# Classes\n")
        f.write("names:\n")
        for class_id, class_name in config['names'].items():
            f.write(f"  {class_id}: {class_name}\n")
        f.write("\n")
        
        # Statistiques
        f.write("# Dataset info\n")
        f.write(f"nc: {config['nc']}  # number of classes\n")
        f.write(f"total_images: {config['total_images']}\n")
        f.write(f"train_images: {config['train_images']}\n")
        f.write(f"val_images: {config['val_images']}\n")
        f.write(f"total_annotations: {config['total_annotations']}\n")
        f.write(f"train_annotations: {config['train_annotations']}\n")
        f.write(f"val_annotations: {config['val_annotations']}\n\n")
        
        # Composition détaillée
        if train_composition:
            f.write("# Training set composition\n")
            f.write("train_composition:\n")
            for prefix, count in sorted(train_composition.items()):
                f.write(f"  {prefix}: {count}  # {prefix}_*.png files\n")
            f.write("\n")
        
        if val_composition:
            f.write("# Validation set composition\n")
            f.write("val_composition:\n")
            for prefix, count in sorted(val_composition.items()):
                f.write(f"  {prefix}: {count}  # {prefix}_*.png files\n")
            f.write("\n")
        
        # Instructions d'entraînement
        f.write("# Training hyperparameters (optimized for panel detection)\n")
        f.write("# Use with: yolo train data=this_config.yaml model=yolov8n.pt epochs=100\n")
    
    return config, train_composition, val_composition

def main():
    print("📊 Mise à jour des statistiques du dataset Multi-BD")
    print(f"📁 Dataset: {DATASET_DIR}")
    print("-" * 60)
    
    # Vérifier que les répertoires existent
    if not TRAIN_IMAGES_DIR.exists():
        print(f"❌ Répertoire d'entraînement non trouvé: {TRAIN_IMAGES_DIR}")
        return 1
    
    # Mettre à jour la configuration
    try:
        config, train_comp, val_comp = update_yaml_config()
        
        print("✅ Configuration mise à jour:")
        print(f"   📸 Images d'entraînement: {config['train_images']}")
        print(f"   📸 Images de validation: {config['val_images']}")
        print(f"   📸 Total images: {config['total_images']}")
        print(f"   🏷️ Annotations d'entraînement: {config['train_annotations']}")
        print(f"   🏷️ Annotations de validation: {config['val_annotations']}")
        print(f"   🏷️ Total annotations: {config['total_annotations']}")
        
        if train_comp:
            print(f"\n📊 Composition du set d'entraînement:")
            for prefix, count in sorted(train_comp.items()):
                print(f"   • {prefix}: {count} images")
        
        if val_comp:
            print(f"\n📊 Composition du set de validation:")
            for prefix, count in sorted(val_comp.items()):
                print(f"   • {prefix}: {count} images")
        
        print(f"\n💾 Configuration sauvegardée: {CONFIG_FILE}")
        
        # Recommandations
        total_files = config['total_images']
        annotated_files = sum(1 for f in TRAIN_LABELS_DIR.glob("*.txt") 
                             if count_annotations(Path(f).parent) > 0)
        
        print(f"\n🎯 RECOMMANDATIONS:")
        if total_files < 100:
            print(f"   📈 Ajouter plus d'images (actuellement: {total_files}, recommandé: 100+)")
        
        if config['val_images'] == 0:
            print(f"   🔄 Déplacer quelques images vers dataset/images/val/ pour validation")
        
        if config['total_annotations'] == 0:
            print(f"   🏷️ Commencer l'annotation des {total_files} images")
            print(f"   💡 Utiliser LabelImg ou Roboflow pour annoter")
        
        print(f"\n🚀 Prochaines étapes:")
        print(f"   1. Annoter les fichiers dans dataset/labels/train/")
        print(f"   2. Lancer l'entraînement: python scripts/start_optimized_training.py")
        
        return 0
        
    except Exception as e:
        print(f"❌ Erreur lors de la mise à jour: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
