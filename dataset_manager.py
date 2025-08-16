#!/usr/bin/env python3
"""
Script pour analyser et gérer le dataset multi-BD
"""

import os
from pathlib import Path
import argparse

def list_series_in_dataset():
    """Affiche les séries présentes dans le dataset."""
    
    print("📚 SÉRIES DANS LE DATASET")
    print("=" * 30)
    
    train_path = Path("dataset/images/train")
    if not train_path.exists():
        print("❌ Dossier dataset/images/train non trouvé")
        return
    
    # Analyser les noms de fichiers
    all_images = list(train_path.glob("*.png"))
    series_stats = {}
    
    for img in all_images:
        name = img.name
        if name.startswith("p") and not any(x in name for x in ["tintin", "pinup"]):
            series = "Golden City"
        elif name.startswith("tintin_"):
            series = "Tintin"
        elif name.startswith("pinup_"):
            series = "Pin-up du B24"
        else:
            # Extraire le préfixe avant _p
            parts = name.split("_p")
            if len(parts) > 1:
                series = parts[0].replace("_", " ").title()
            else:
                series = "Autre"
        
        if series not in series_stats:
            series_stats[series] = []
        series_stats[series].append(img.name)
    
    # Afficher les statistiques
    total_images = 0
    for series, files in sorted(series_stats.items()):
        count = len(files)
        total_images += count
        print(f"  📖 {series}: {count} images")
        
        # Quelques exemples de fichiers
        examples = sorted(files)[:3]
        for example in examples:
            print(f"     - {example}")
        if len(files) > 3:
            print(f"     ... et {len(files) - 3} autres")
        print()
    
    print(f"Total: {total_images} images")
    
    # Vérifier les annotations
    labels_path = Path("dataset/labels/train")
    if labels_path.exists():
        annotated_count = len(list(labels_path.glob("*.json")))
        coverage = (annotated_count / total_images) * 100 if total_images else 0
        print(f"Annotées: {annotated_count} images ({coverage:.1f}%)")

def analyze_annotation_coverage():
    """Analyse la couverture d'annotation par série."""
    
    print("📊 ANALYSE DE COUVERTURE D'ANNOTATION")
    print("=" * 40)
    
    train_path = Path("dataset/images/train")
    labels_path = Path("dataset/labels/train")
    
    if not train_path.exists():
        print("❌ Dossier images non trouvé")
        return
    
    # Grouper par série
    series_images = {}
    series_annotations = {}
    
    # Images
    for img in train_path.glob("*.png"):
        name = img.name
        if name.startswith("p") and not "tintin" in name:
            series = "Golden City"
        elif name.startswith("tintin_"):
            series = "Tintin"
        else:
            parts = name.split("_p")
            series = parts[0].replace("_", " ").title() if len(parts) > 1 else "Autre"
        
        if series not in series_images:
            series_images[series] = []
        series_images[series].append(name)
    
    # Annotations
    if labels_path.exists():
        for json_file in labels_path.glob("*.json"):
            name = json_file.stem + ".png"  # Nom de l'image correspondante
            if name.startswith("p") and not any(x in name for x in ["tintin", "pinup"]):
                series = "Golden City"
            elif name.startswith("tintin_"):
                series = "Tintin"
            elif name.startswith("pinup_"):
                series = "Pin-up du B24"
            else:
                parts = name.split("_p")
                series = parts[0].replace("_", " ").title() if len(parts) > 1 else "Autre"
            
            if series not in series_annotations:
                series_annotations[series] = []
            series_annotations[series].append(name)
    
    # Afficher la couverture
    print("Série                | Images | Annotées | Couverture")
    print("-" * 50)
    for series in sorted(series_images.keys()):
        img_count = len(series_images[series])
        ann_count = len(series_annotations.get(series, []))
        coverage = (ann_count / img_count) * 100 if img_count > 0 else 0
        
        print(f"{series:<20} | {img_count:6} | {ann_count:8} | {coverage:7.1f}%")
    
    # Recommandations
    print("\n💡 Recommandations:")
    for series in sorted(series_images.keys()):
        img_count = len(series_images[series])
        ann_count = len(series_annotations.get(series, []))
        coverage = (ann_count / img_count) * 100 if img_count > 0 else 0
        
        if coverage < 30:
            needed = max(10, int(img_count * 0.3)) - ann_count
            print(f"   📝 {series}: annoter {needed} images de plus (priorité haute)")
        elif coverage < 50:
            needed = max(15, int(img_count * 0.5)) - ann_count  
            print(f"   🎯 {series}: annoter {needed} images de plus (priorité moyenne)")
        else:
            print(f"   ✅ {series}: couverture suffisante ({coverage:.1f}%)")

def show_add_instructions(series_name):
    """Instructions pour ajouter manuellement un nouveau PDF."""
    
    print(f"📝 INSTRUCTIONS POUR AJOUTER: {series_name}")
    print("=" * 50)
    print()
    print("1. 📁 Place ton PDF dans le dossier courant")
    print("2. 🔧 Utilise un des scripts existants comme modèle:")
    print("   - integrate_tintin.py pour voir comment extraire les pages")
    print("3. 📝 Modifie le script pour ton nouveau PDF:")
    print("   - Change le nom du fichier PDF")
    print("   - Change le préfixe des images (ex: asterix_, spirou_)")
    print("   - Ajuste le nombre de pages à extraire")
    print()
    print("4. 🏷️ Convention de nommage recommandée:")
    print("   - asterix_p0001.png, asterix_p0002.png...")
    print("   - spirou_p0001.png, spirou_p0002.png...")
    print("   - blake_p0001.png (Blake et Mortimer)")
    print()
    print("5. 🎯 Ensuite utilise start_annotation.py pour annoter")
    print()
    print("💡 Exemples de séries recommandées pour diversifier:")
    print("   • Astérix - Style simple comme Tintin")
    print("   • Spirou - Style intermédiaire")
    print("   • Blake et Mortimer - Style détaillé")
    print("   • Lucky Luke - Style cartoon")
    print("   • XIII - Style moderne photo-réaliste")

def main():
    parser = argparse.ArgumentParser(description="Gestion du dataset multi-BD")
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande pour lister les séries
    list_parser = subparsers.add_parser('list', help='Lister les series dans le dataset')
    
    # Commande pour analyser la couverture
    coverage_parser = subparsers.add_parser('coverage', help='Analyser la couverture d annotation')
    
    # Commande pour instructions d'ajout
    add_parser = subparsers.add_parser('add-help', help='Instructions pour ajouter une serie')
    add_parser.add_argument('name', help='Nom de la serie a ajouter')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_series_in_dataset()
    elif args.command == 'coverage':
        analyze_annotation_coverage()
    elif args.command == 'add-help':
        show_add_instructions(args.name)
    else:
        parser.print_help()
        print("\n💡 Exemples d'usage:")
        print("  python manage_dataset.py list")
        print("  python manage_dataset.py coverage") 
        print("  python manage_dataset.py add-help Asterix")

if __name__ == "__main__":
    main()
