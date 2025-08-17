#!/usr/bin/env python3
"""
Intégration Tintin au Dataset
Ajoute les pages Tintin au dataset existant avec préfixes pour éviter les conflits.
"""

import os
import shutil
from pathlib import Path
import json

def integrate_tintin_to_dataset():
    """Intègre les pages Tintin au dataset principal."""
    
    print("🚀 INTÉGRATION TINTIN AU DATASET")
    print("=" * 40)
    
    # Chemins
    tintin_dir = Path("temp_tintin")
    train_dir = Path("dataset/images/train")
    labels_dir = Path("dataset/labels/train")
    
    # Vérifications
    if not tintin_dir.exists():
        print("❌ Dossier temp_tintin introuvable")
        return False
    
    tintin_pages = list(tintin_dir.glob("*.png"))
    if not tintin_pages:
        print("❌ Aucune page Tintin trouvée")
        return False
    
    print(f"📚 Pages Tintin trouvées: {len(tintin_pages)}")
    print(f"📚 Pages Golden City existantes: {len(list(train_dir.glob('*.png')))}")
    
    # Créer les dossiers si nécessaire
    train_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Copier les pages Tintin avec préfixe
    copied_count = 0
    skipped_count = 0
    
    print("\n📋 Intégration des pages...")
    
    for tintin_page in sorted(tintin_pages):
        # Nouveau nom avec préfixe tintin_
        new_name = f"tintin_{tintin_page.name}"
        dest_path = train_dir / new_name
        
        # Vérifier si le fichier existe déjà
        if dest_path.exists():
            print(f"⚠️  Existe déjà: {new_name}")
            skipped_count += 1
            continue
        
        # Copier le fichier
        try:
            shutil.copy2(tintin_page, dest_path)
            print(f"✅ Copié: {tintin_page.name} → {new_name}")
            copied_count += 1
        except Exception as e:
            print(f"❌ Erreur copie {tintin_page.name}: {e}")
    
    print(f"\n📊 RÉSULTAT:")
    print(f"   Fichiers copiés: {copied_count}")
    print(f"   Fichiers ignorés: {skipped_count}")
    print(f"   Total images train: {len(list(train_dir.glob('*.png')))}")
    
    return copied_count > 0

def update_dataset_split():
    """Met à jour la répartition train/val avec les nouvelles images."""
    
    print("\n🔄 MISE À JOUR DU SPLIT TRAIN/VAL")
    print("=" * 35)
    
    train_dir = Path("dataset/images/train")
    val_dir = Path("dataset/images/val")
    
    all_images = list(train_dir.glob("*.png"))
    val_images = list(val_dir.glob("*.png"))
    
    total_images = len(all_images) + len(val_images)
    current_val_ratio = len(val_images) / total_images if total_images > 0 else 0
    
    print(f"📊 Images actuelles:")
    print(f"   Train: {len(all_images)}")
    print(f"   Val: {len(val_images)}")
    print(f"   Ratio val: {current_val_ratio:.1%}")
    
    # Recommandation pour un bon split
    recommended_val_count = max(5, int(total_images * 0.15))  # 15% minimum 5
    
    if len(val_images) < recommended_val_count:
        needed_val = recommended_val_count - len(val_images)
        print(f"\n💡 Recommandation: Déplacer {needed_val} images vers validation")
        print(f"   Pour atteindre ~15% validation ({recommended_val_count} images)")
    else:
        print(f"\n✅ Split validation OK ({len(val_images)} images)")

def analyze_new_dataset():
    """Analyse le dataset étendu."""
    
    print("\n📈 ANALYSE DU DATASET ÉTENDU")
    print("=" * 35)
    
    train_dir = Path("dataset/images/train")
    labels_dir = Path("dataset/labels/train")
    
    # Compter par source
    golden_city_images = [f for f in train_dir.glob("*.png") if not f.name.startswith("tintin_")]
    tintin_images = [f for f in train_dir.glob("*.png") if f.name.startswith("tintin_")]
    
    # Compter les annotations
    golden_city_labels = 0
    tintin_labels = 0
    
    for json_file in labels_dir.glob("*.json"):
        if json_file.name.startswith("tintin_"):
            tintin_labels += 1
        else:
            golden_city_labels += 1
    
    print(f"📚 Composition du dataset:")
    print(f"   Golden City: {len(golden_city_images)} images ({golden_city_labels} annotées)")
    print(f"   Tintin: {len(tintin_images)} images ({tintin_labels} annotées)")
    print(f"   Total: {len(golden_city_images) + len(tintin_images)} images")
    
    # Progression annotations
    total_images = len(golden_city_images) + len(tintin_images)
    total_labels = golden_city_labels + tintin_labels
    
    if total_images > 0:
        progress = total_labels / total_images * 100
        print(f"\n📊 Progression annotations: {total_labels}/{total_images} ({progress:.1f}%)")
        
        if progress < 30:
            print("💡 Recommandation: Annoter au moins 30% avant re-entraînement")
        elif progress > 50:
            print("🚀 Prêt pour un nouvel entraînement !")

def main():
    """Fonction principale d'intégration."""
    
    # 1. Intégrer les pages Tintin
    if not integrate_tintin_to_dataset():
        print("❌ Échec de l'intégration")
        return
    
    # 2. Analyser le nouveau dataset
    analyze_new_dataset()
    
    # 3. Suggestions pour le split
    update_dataset_split()
    
    print("\n🎯 PROCHAINES ÉTAPES:")
    print("=" * 20)
    print("1. 🏷️  Annoter les nouvelles pages Tintin:")
    print("   python start_annotation.py")
    print()
    print("2. 🔄 Régénérer le dataset YOLO:")
    print("   python tools/labelme_to_yolo.py")
    print()
    print("3. 🏋️  Relancer l'entraînement:")
    print("   python continue_training.py")
    print()
    print("4. 📊 Suivre les progrès:")
    print("   python annotation_progress.py")

if __name__ == "__main__":
    main()
