#!/usr/bin/env python3
"""
Script de nettoyage et mise à jour du dataset Multi-BD v2
========================================================

Exécute les 6 étapes de nettoyage pour préparer l'entraînement:
1. Backup des annotations
2. Suppression des titres dans JSON
3. Mise à jour YAML
4. Conversion Labelme→YOLO
5. Purge cache YOLO
6. Validation finale

Usage:
    python tools/cleanup_dataset_v2.py --simulation
    python tools/cleanup_dataset_v2.py --execute

Auteur: Vincent Cruvellier
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import sys

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
LABELS_DIR = DATASET_DIR / "labels" / "train"
YAML_FILE = DATASET_DIR / "multibd_enhanced.yaml"

# Classes finales
FINAL_CLASSES = {
    0: "panel",
    1: "balloon"
}

def create_backup(dry_run: bool = False):
    """Étape 1: Créer un backup des annotations."""
    print("📦 ÉTAPE 1: Backup des annotations")
    print("-" * 40)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = PROJECT_DIR / f"backup_annotations_{timestamp}"
    
    if dry_run:
        print(f"🧪 Simulation backup vers: {backup_dir}")
        json_count = len(list(LABELS_DIR.glob("*.json")))
        txt_count = len(list(LABELS_DIR.glob("*.txt")))
        print(f"   📄 {json_count} fichiers JSON à sauvegarder")
        print(f"   📄 {txt_count} fichiers TXT à sauvegarder")
    else:
        print(f"💾 Création backup: {backup_dir}")
        backup_dir.mkdir(exist_ok=True)
        
        # Copier tous les fichiers d'annotations
        files_copied = 0
        for ext in ["*.json", "*.txt"]:
            for file in LABELS_DIR.glob(ext):
                shutil.copy2(file, backup_dir / file.name)
                files_copied += 1
        
        print(f"✅ {files_copied} fichiers sauvegardés")
    
    print()
    return backup_dir

def remove_titles_from_json(dry_run: bool = False):
    """Étape 2: Supprimer les titres des fichiers JSON."""
    print("🧹 ÉTAPE 2: Suppression des titres JSON")
    print("-" * 40)
    
    json_files = list(LABELS_DIR.glob("*.json"))
    removed_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filtrer les shapes pour garder seulement panel et balloon
            original_count = len(data.get('shapes', []))
            filtered_shapes = []
            
            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower()
                if label in ['panel', 'balloon']:
                    filtered_shapes.append(shape)
                else:
                    removed_count += 1
            
            data['shapes'] = filtered_shapes
            new_count = len(filtered_shapes)
            
            if original_count != new_count:
                if dry_run:
                    print(f"🧪 {json_file.name}: {original_count} → {new_count} shapes")
                else:
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"✏️ {json_file.name}: {original_count} → {new_count} shapes")
                    
        except Exception as e:
            print(f"❌ Erreur {json_file.name}: {e}")
    
    if dry_run:
        print(f"🧪 {removed_count} annotations à supprimer")
    else:
        print(f"✅ {removed_count} annotations supprimées")
    
    print()
    return removed_count

def update_yaml_config(dry_run: bool = False):
    """Étape 3: Mettre à jour le fichier YAML."""
    print("📝 ÉTAPE 3: Mise à jour YAML")
    print("-" * 40)
    
    yaml_content = f"""# Configuration Multi-BD Enhanced v2
# Classes: panel (cases), balloon (bulles)
# Mise à jour: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

path: ./dataset
train: images/train
val: images/val

# Classes
nc: 2
names:
  0: panel
  1: balloon

# Métadonnées
version: 2.0
description: "Dataset Multi-BD Enhanced avec classes panel/balloon"
sources:
  - "Golden City (82 images)"
  - "Tintin (61 images)"
  - "Pin-up (48 images)"
  - "Sisters (67 images)"
total_images: 258
"""
    
    if dry_run:
        print("🧪 Simulation mise à jour YAML:")
        print("   nc: 2")
        print("   names: [panel, balloon]")
    else:
        with open(YAML_FILE, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        print(f"✅ YAML mis à jour: {YAML_FILE}")
    
    print()

def convert_annotations(dry_run: bool = False):
    """Étape 4: Convertir Labelme vers YOLO."""
    print("🔄 ÉTAPE 4: Conversion Labelme → YOLO")
    print("-" * 40)
    
    converter_script = SCRIPT_DIR / "convert_labelme_to_yolo_v2.py"
    
    cmd = [sys.executable, str(converter_script), "--clean"]
    if dry_run:
        cmd.append("--dry-run")
    
    if dry_run:
        print("🧪 Simulation conversion:")
        print(f"   Commande: {' '.join(cmd)}")
    else:
        print("🚀 Lancement conversion...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Conversion terminée")
            # Afficher les dernières lignes du résultat
            lines = result.stdout.strip().split('\n')[-5:]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
        else:
            print("❌ Erreur conversion:")
            print(result.stderr)
    
    print()

def purge_yolo_cache(dry_run: bool = False):
    """Étape 5: Purger le cache YOLO."""
    print("🗑️ ÉTAPE 5: Purge cache YOLO")
    print("-" * 40)
    
    cache_dirs = [
        PROJECT_DIR / "runs",
        PROJECT_DIR / "dataset" / "train",
        PROJECT_DIR / "dataset" / "val"
    ]
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            if dry_run:
                size_info = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                size_mb = size_info / (1024 * 1024)
                print(f"🧪 À supprimer: {cache_dir.name} ({size_mb:.1f} MB)")
            else:
                shutil.rmtree(cache_dir)
                print(f"🗑️ Supprimé: {cache_dir}")
        else:
            print(f"ℹ️ Déjà absent: {cache_dir.name}")
    
    print()

def validate_dataset(dry_run: bool = False):
    """Étape 6: Validation finale."""
    print("✅ ÉTAPE 6: Validation finale")
    print("-" * 40)
    
    # Compter les fichiers
    json_files = list(LABELS_DIR.glob("*.json"))
    txt_files = list(LABELS_DIR.glob("*.txt"))
    
    print(f"📄 Fichiers JSON: {len(json_files)}")
    print(f"📄 Fichiers TXT: {len(txt_files)}")
    
    # Vérifier l'appariement
    json_stems = {f.stem for f in json_files}
    txt_stems = {f.stem for f in txt_files}
    
    missing_txt = json_stems - txt_stems
    orphan_txt = txt_stems - json_stems
    
    if missing_txt:
        print(f"⚠️ TXT manquants: {len(missing_txt)}")
        for stem in sorted(list(missing_txt)[:5]):
            print(f"   {stem}.txt")
        if len(missing_txt) > 5:
            print(f"   ... et {len(missing_txt) - 5} autres")
    
    if orphan_txt:
        print(f"⚠️ TXT orphelins: {len(orphan_txt)}")
    
    # Analyser les classes
    if txt_files:
        class_counts = {0: 0, 1: 0}
        total_annotations = 0
        
        for txt_file in txt_files[:10] if dry_run else txt_files:  # Limiter en simulation
            try:
                with open(txt_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if class_id in class_counts:
                                    class_counts[class_id] += 1
                                total_annotations += 1
            except:
                pass
        
        print(f"🏷️ Annotations totales: {total_annotations}")
        for class_id, count in class_counts.items():
            class_name = FINAL_CLASSES.get(class_id, f"class_{class_id}")
            print(f"   {class_id} ({class_name}): {count}")
    
    # État final
    if len(json_files) == len(txt_files) and not missing_txt and not orphan_txt:
        print("\n🎯 Dataset prêt pour l'entraînement!")
        if not dry_run:
            print("   Commande: python train_enhanced_v2.py")
    else:
        print("\n⚠️ Dataset nécessite des corrections")
    
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Nettoyage complet du dataset Multi-BD v2",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--simulation", action="store_true", 
                      help="Mode simulation (dry-run)")
    group.add_argument("--execute", action="store_true", 
                      help="Exécution réelle")
    
    args = parser.parse_args()
    
    dry_run = args.simulation
    
    print("🚀 NETTOYAGE DATASET MULTI-BD V2")
    print("=" * 50)
    print(f"📁 Projet: {PROJECT_DIR}")
    print(f"🏷️ Classes finales: {FINAL_CLASSES}")
    if dry_run:
        print("🧪 MODE SIMULATION")
    else:
        print("⚡ MODE EXÉCUTION")
    print("=" * 50)
    print()
    
    try:
        # Exécuter les 6 étapes
        backup_dir = create_backup(dry_run)
        removed_titles = remove_titles_from_json(dry_run)
        update_yaml_config(dry_run)
        convert_annotations(dry_run)
        purge_yolo_cache(dry_run)
        validate_dataset(dry_run)
        
        # Résumé final
        print("🎉 NETTOYAGE TERMINÉ")
        print("=" * 50)
        if dry_run:
            print("🧪 Simulation réussie")
            print("   Pour exécuter: python tools/cleanup_dataset_v2.py --execute")
        else:
            print("✅ Nettoyage réussi")
            print(f"💾 Backup disponible: {backup_dir}")
            print("🎯 Prêt pour l'entraînement!")
        
        return 0
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
