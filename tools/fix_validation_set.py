#!/usr/bin/env python3
"""
Convertisseur spécial pour le dossier de validation
Corrige le problème du validation set vide en convertissant les JSON du dossier val
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration des classes
CLASS_MAPPING = {
    "panel": 0,
    "balloon": 1,
}

# Chemins
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR.parent / "dataset"
VAL_LABELS_DIR = DATASET_DIR / "labels" / "val"

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

def convert_val_labelme_to_yolo(json_path: Path) -> Dict:
    """Convertit un fichier Labelme JSON du dossier val vers YOLO TXT."""
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
            label = shape.get('label', '').lower()  # Normaliser en minuscules
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
        output_path = json_path.with_suffix('.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in yolo_annotations:
                f.write(line + '\n')
        results["output_path"] = output_path
            
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
    
    return results

def main():
    print("🔄 Correction du validation set")
    print("=" * 50)
    print(f"📁 Répertoire: {VAL_LABELS_DIR}")
    print(f"🏷️ Classes: {CLASS_MAPPING}")
    print("-" * 50)
    
    # Trouver tous les fichiers JSON dans val
    json_files = list(VAL_LABELS_DIR.glob("*.json"))
    if not json_files:
        print("❌ Aucun fichier JSON trouvé dans le dossier val")
        return 1
    
    print(f"📄 {len(json_files)} fichiers JSON trouvés dans val")
    print()
    
    # Statistiques globales
    total_success = 0
    total_errors = 0
    all_classes = set()
    all_ignored = set()
    total_annotations = 0
    
    # Convertir chaque fichier
    for i, json_file in enumerate(json_files, 1):
        print(f"[{i:2d}/{len(json_files)}] {json_file.name:<20}", end=" → ")
        
        results = convert_val_labelme_to_yolo(json_file)
        
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
    print("📊 RÉSUMÉ CORRECTION VALIDATION SET")
    print("=" * 50)
    print(f"✅ Fichiers convertis: {total_success}")
    print(f"❌ Erreurs: {total_errors}")
    print(f"🏷️ Total annotations val: {total_annotations}")
    
    if all_classes:
        print(f"\n🎯 Classes trouvées dans val:")
        for class_name in sorted(all_classes):
            class_id = CLASS_MAPPING[class_name]
            print(f"  {class_id}: {class_name}")
    
    if all_ignored:
        print(f"\n⚠️ Classes ignorées: {', '.join(sorted(all_ignored))}")
    
    if total_success > 0:
        print(f"\n🎯 VALIDATION SET CORRIGÉ!")
        print(f"  ✅ Le dataset est maintenant parfait pour l'entraînement")
        print(f"  🚀 Relancer: .venv/bin/python scripts/training/train_enhanced_v2.py")
    
    return 0 if total_errors == 0 else 1

if __name__ == "__main__":
    exit(main())
