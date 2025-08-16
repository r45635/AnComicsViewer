#!/usr/bin/env python3
"""
Script corrigé pour analyser le dataset multi-BD
"""

import os
from pathlib import Path
import argparse

def count_images_by_series(train_dir):
    """Compte les images par série de manière précise."""
    
    all_files = list(Path(train_dir).glob("*.png"))
    
    golden_files = []
    tintin_files = []
    pinup_files = []
    other_files = []
    
    for file in all_files:
        name = file.name
        if name.startswith("pinup_"):
            pinup_files.append(name)
        elif name.startswith("tintin_"):
            tintin_files.append(name)
        elif name.startswith("p") and not ("tintin" in name or "pinup" in name):
            golden_files.append(name)
        else:
            other_files.append(name)
    
    return {
        "Golden City": golden_files,
        "Tintin": tintin_files,
        "Pin-up du B24": pinup_files,
        "Autres": other_files
    }

def count_annotations_by_series(labels_dir):
    """Compte les annotations par série."""
    
    if not Path(labels_dir).exists():
        return {}
    
    all_json = list(Path(labels_dir).glob("*.json"))
    
    golden_ann = []
    tintin_ann = []
    pinup_ann = []
    other_ann = []
    
    for file in all_json:
        name = file.stem  # Nom sans extension
        if name.startswith("pinup_"):
            pinup_ann.append(name)
        elif name.startswith("tintin_"):
            tintin_ann.append(name)
        elif name.startswith("p") and not ("tintin" in name or "pinup" in name):
            golden_ann.append(name)
        else:
            other_ann.append(name)
    
    return {
        "Golden City": golden_ann,
        "Tintin": tintin_ann,
        "Pin-up du B24": pinup_ann,
        "Autres": other_ann
    }

def analyze_dataset():
    """Analyse complète du dataset."""
    
    print("📊 ANALYSE DATASET MULTI-BD")
    print("=" * 40)
    
    # Compter les images
    images_by_series = count_images_by_series("dataset/images/train")
    annotations_by_series = count_annotations_by_series("dataset/labels/train")
    
    # Afficher le tableau
    print("Série                | Images | Annotées | Couverture")
    print("-" * 50)
    
    total_images = 0
    total_annotations = 0
    
    for series in ["Golden City", "Tintin", "Pin-up du B24", "Autres"]:
        img_count = len(images_by_series.get(series, []))
        ann_count = len(annotations_by_series.get(series, []))
        
        if img_count > 0:
            coverage = (ann_count / img_count) * 100
            print(f"{series:<20} | {img_count:6} | {ann_count:8} | {coverage:7.1f}%")
            total_images += img_count
            total_annotations += ann_count
    
    print("-" * 50)
    overall_coverage = (total_annotations / total_images) * 100 if total_images > 0 else 0
    print(f"{'TOTAL':<20} | {total_images:6} | {total_annotations:8} | {overall_coverage:7.1f}%")
    
    # Recommandations
    print("\n💡 PRIORITÉS D'ANNOTATION:")
    
    recommendations = []
    for series in ["Golden City", "Tintin", "Pin-up du B24"]:
        img_count = len(images_by_series.get(series, []))
        ann_count = len(annotations_by_series.get(series, []))
        
        if img_count == 0:
            continue
            
        coverage = (ann_count / img_count) * 100
        target = min(20, max(10, img_count // 3))  # Cible: 10-20 annotations par série
        needed = max(0, target - ann_count)
        
        recommendations.append((coverage, needed, series))
    
    # Trier par priorité (couverture la plus faible d'abord)
    recommendations.sort()
    
    for coverage, needed, series in recommendations:
        if needed > 0:
            if coverage < 15:
                priority = "🔴 HAUTE"
            elif coverage < 30:
                priority = "🟡 MOYENNE"
            else:
                priority = "🟢 BASSE"
            print(f"   {priority}: {series} - annoter {needed} images de plus")
        else:
            print(f"   ✅ {series}: couverture suffisante ({coverage:.1f}%)")
    
    # Style des BD
    print(f"\n🎨 DIVERSITÉ DES STYLES:")
    print(f"   • Golden City: {len(images_by_series['Golden City'])} images - Style moderne complexe")
    print(f"   • Tintin: {len(images_by_series['Tintin'])} images - Style classique simple")
    print(f"   • Pin-up du B24: {len(images_by_series['Pin-up du B24'])} images - Style aviation/guerre")
    print(f"   ➡️  Excellent mix pour un modèle robuste!")

def list_series():
    """Liste détaillée des séries."""
    
    print("📚 SÉRIES DANS LE DATASET")
    print("=" * 30)
    
    images_by_series = count_images_by_series("dataset/images/train")
    
    for series, files in images_by_series.items():
        if len(files) > 0:
            print(f"\n📖 {series}: {len(files)} images")
            
            # Quelques exemples
            examples = sorted(files)[:3]
            for example in examples:
                print(f"     - {example}")
            if len(files) > 3:
                print(f"     ... et {len(files) - 3} autres")

def main():
    parser = argparse.ArgumentParser(description="Analyse du dataset multi-BD")
    parser.add_argument('command', choices=['list', 'coverage'], help='Commande à exécuter')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_series()
    elif args.command == 'coverage':
        analyze_dataset()

if __name__ == "__main__":
    main()
