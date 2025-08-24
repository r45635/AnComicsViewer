#!/usr/bin/env python3
"""
Convertisseur Labelme vers YOLO pour dataset Multi-BD v2
========================================================

Convertit les annotations Labelme JSON vers le format YOLO TXT
avec support pour les classes: panel, balloon

Usage:
    python tools/convert_labelme_to_yolo_v2.py
    python tools/convert_labelme_to_yolo_v2.py --dry-run
    python tools/convert_labelme_to_yolo_v2.py --clean

Classes supportées:
    0 = panel (cases de BD)
    1 = balloon (bulles de dialogue)

Auteur: Vincent Cruvellier
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration des classes
CLASS_MAPPING = {
    "panel": 0,
    "balloon": 1,
    # Ignorer d'autres classes si elles existent
}

# Chemins
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR.parent / "dataset"
LABELS_DIR = DATASET_DIR / "labels" / "train"

def convert_bbox_to_yolo(bbox_points: List[List[float]], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """Convertit les coordonnées de bounding box vers le format YOLO."""
    # Extraire les coordonnées min/max
    x1, y1 = bbox_points[0]
    x2, y2 = bbox_points[1]
    
    # S'assurer que x1,y1 est le coin supérieur gauche
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1, x2)
    y_max = max(y1, y2)
    
    # Calculer centre et dimensions
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normaliser par rapport à la taille de l'image
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return center_x_norm, center_y_norm, width_norm, height_norm

def convert_labelme_to_yolo(json_path: Path, dry_run: bool = False) -> Dict:
    """Convertit un fichier Labelme JSON vers YOLO TXT."""
    results = {
        "status": "success",
        "classes_found": set(),
        "annotations_count": 0,
        "ignored_classes": set(),
        "output_path": None
    }
    
    try:
        # Lire le fichier JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_width = data.get('imageWidth', 0)
        img_height = data.get('imageHeight', 0)
        
        if img_width == 0 or img_height == 0:
            results["status"] = "error"
            results["error"] = f"Dimensions d'image invalides: {img_width}x{img_height}"
            return results
        
        yolo_annotations = []
        
        # Traiter chaque shape
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            shape_type = shape.get('shape_type', '')
            points = shape.get('points', [])
            
            # Vérifier que c'est un rectangle
            if shape_type != 'rectangle':
                continue
                
            # Vérifier que la classe est supportée
            if label not in CLASS_MAPPING:
                results["ignored_classes"].add(label)
                continue
                
            if len(points) != 2:
                continue
                
            # Convertir vers YOLO
            class_id = CLASS_MAPPING[label]
            center_x, center_y, width, height = convert_bbox_to_yolo(
                points, img_width, img_height
            )
            
            # Ajouter l'annotation
            yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            yolo_annotations.append(yolo_line)
            
            results["classes_found"].add(label)
            results["annotations_count"] += 1
        
        # Écrire le fichier YOLO TXT
        if not dry_run:
            output_path = json_path.with_suffix('.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in yolo_annotations:
                    f.write(line + '\n')
            results["output_path"] = output_path
        else:
            results["output_path"] = json_path.with_suffix('.txt')
            
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
    
    return results

def clean_old_txt_files(dry_run: bool = False):
    """Supprime les anciens fichiers TXT qui ne correspondent à aucun JSON."""
    txt_files = list(LABELS_DIR.glob("*.txt"))
    json_files = {f.stem for f in LABELS_DIR.glob("*.json")}
    
    removed_count = 0
    for txt_file in txt_files:
        if txt_file.stem not in json_files:
            if dry_run:
                print(f"🧪 Simulation suppression: {txt_file.name}")
            else:
                print(f"🗑️ Suppression ancien fichier: {txt_file.name}")
                txt_file.unlink()
            removed_count += 1
    
    return removed_count

def main():
    parser = argparse.ArgumentParser(
        description="Convertir annotations Labelme vers YOLO v2",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="Simulation sans écriture")
    parser.add_argument("--clean", action="store_true", help="Nettoyer les anciens fichiers TXT")
    
    args = parser.parse_args()
    
    print("🔄 Conversion Labelme → YOLO v2")
    print("=" * 50)
    print(f"📁 Répertoire: {LABELS_DIR}")
    print(f"🏷️ Classes: {CLASS_MAPPING}")
    if args.dry_run:
        print("🧪 MODE SIMULATION")
    print("-" * 50)
    
    # Nettoyer les anciens fichiers si demandé
    if args.clean:
        removed = clean_old_txt_files(args.dry_run)
        print(f"🗑️ {removed} anciens fichiers {'à supprimer' if args.dry_run else 'supprimés'}\n")
    
    # Trouver tous les fichiers JSON
    json_files = list(LABELS_DIR.glob("*.json"))
    if not json_files:
        print("❌ Aucun fichier JSON trouvé")
        return 1
    
    print(f"📄 {len(json_files)} fichiers JSON trouvés")
    print()
    
    # Statistiques globales
    total_success = 0
    total_errors = 0
    all_classes = set()
    all_ignored = set()
    total_annotations = 0
    
    # Convertir chaque fichier
    for i, json_file in enumerate(json_files, 1):
        print(f"[{i:3d}/{len(json_files)}] {json_file.name:<25}", end=" → ")
        
        results = convert_labelme_to_yolo(json_file, args.dry_run)
        
        if results["status"] == "success":
            print(f"✅ {results['annotations_count']} annotations")
            total_success += 1
            all_classes.update(results["classes_found"])
            all_ignored.update(results["ignored_classes"])
            total_annotations += results["annotations_count"]
            
            if results["classes_found"]:
                classes_str = ", ".join(sorted(results["classes_found"]))
                print(f"    Classes: {classes_str}")
                
        else:
            print(f"❌ {results.get('error', 'Erreur inconnue')}")
            total_errors += 1
    
    # Résumé final
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DE LA CONVERSION")
    print("=" * 50)
    print(f"✅ Fichiers convertis: {total_success}")
    print(f"❌ Erreurs: {total_errors}")
    print(f"🏷️ Total annotations: {total_annotations}")
    
    if all_classes:
        print(f"\n🎯 Classes trouvées:")
        for class_name in sorted(all_classes):
            class_id = CLASS_MAPPING[class_name]
            print(f"  {class_id}: {class_name}")
    
    if all_ignored:
        print(f"\n⚠️ Classes ignorées: {', '.join(sorted(all_ignored))}")
    
    if not args.dry_run and total_success > 0:
        print(f"\n🎯 PROCHAINES ÉTAPES:")
        print(f"  1. Vérifier dataset/multibd_enhanced.yaml")
        print(f"  2. Nettoyer cache YOLO: rm -rf runs/")
        print(f"  3. Relancer entraînement: python train_enhanced_v2.py")
    elif args.dry_run:
        print(f"\n🧪 SIMULATION TERMINÉE")
        print(f"  Pour exécuter: python tools/convert_labelme_to_yolo_v2.py --clean")
    
    return 0 if total_errors == 0 else 1

if __name__ == "__main__":
    exit(main())
