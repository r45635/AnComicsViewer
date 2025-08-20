#!/usr/bin/env python3
"""
Script pour convertir les annotations Labelme en format YOLO
===========================================================

Ce script convertit les fichiers JSON de Labelme vers le format YOLO (.txt)
utilisé par le dataset Multi-BD Enhanced.

Usage:
    python scripts/convert_labelme_to_yolo.py --input dataset/annotations_labelme/ --output dataset/labels/train/

Auteur: Vincent Cruvellier
"""

import os
import json
import argparse
from pathlib import Path


def convert_labelme_to_yolo(json_path: str, output_path: str, img_width: int, img_height: int):
    """
    Convertit un fichier JSON Labelme en format YOLO.
    
    Args:
        json_path: Chemin vers le fichier JSON Labelme
        output_path: Chemin de sortie pour le fichier YOLO .txt
        img_width: Largeur de l'image
        img_height: Hauteur de l'image
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Classes Multi-BD
    class_mapping = {
        'panel': 0,
        'panel_inset': 1
    }
    
    yolo_annotations = []
    
    for shape in data.get('shapes', []):
        label = shape['label'].lower()
        if label not in class_mapping:
            print(f"⚠️  Classe inconnue ignorée: {label}")
            continue
            
        class_id = class_mapping[label]
        points = shape['points']
        
        # Convertir en bounding box
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Normaliser en format YOLO (0-1)
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Sauvegarder
    with open(output_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))
    
    return len(yolo_annotations)


def main():
    parser = argparse.ArgumentParser(description='Convertir annotations Labelme vers YOLO')
    parser.add_argument('--input', required=True, help='Dossier contenant les JSON Labelme')
    parser.add_argument('--output', required=True, help='Dossier de sortie pour les fichiers YOLO')
    parser.add_argument('--images', required=True, help='Dossier contenant les images correspondantes')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    images_dir = Path(args.images)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Conversion Labelme → YOLO")
    print("=" * 40)
    
    converted = 0
    total_annotations = 0
    
    for json_file in input_dir.glob("*.json"):
        # Trouver l'image correspondante
        image_name = json_file.stem
        image_files = list(images_dir.glob(f"{image_name}.*"))
        
        if not image_files:
            print(f"❌ Image non trouvée pour: {json_file.name}")
            continue
            
        image_path = image_files[0]
        
        # Obtenir les dimensions de l'image
        from PIL import Image
        with Image.open(image_path) as img:
            img_width, img_height = img.size
        
        # Convertir
        output_file = output_dir / f"{image_name}.txt"
        annotations_count = convert_labelme_to_yolo(
            str(json_file), 
            str(output_file), 
            img_width, 
            img_height
        )
        
        total_annotations += annotations_count
        converted += 1
        print(f"✅ {json_file.name} → {output_file.name} ({annotations_count} annotations)")
    
    print(f"\n🎉 Conversion terminée!")
    print(f"📁 Fichiers convertis: {converted}")
    print(f"🏷️ Total annotations: {total_annotations}")


if __name__ == "__main__":
    main()
