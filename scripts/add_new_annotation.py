#!/usr/bin/env python3
"""
Script pour ajouter un nouveau fichier d'annotation au dataset Multi-BD
=======================================================================

Ce script facilite l'ajout de nouvelles images et annotations au dataset
d'entraînement Multi-BD Enhanced.

Usage:
    python scripts/add_new_annotation.py --help
    python scripts/add_new_annotation.py --image chemin/vers/image.png --name mon_fichier
    python scripts/add_new_annotation.py --pdf chemin/vers/bd.pdf --page 5 --name ma_bd_p05

Auteur: Vincent Cruvellier
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import subprocess

# Configuration
DATASET_DIR = Path(__file__).parent.parent / "dataset"
TRAIN_IMAGES_DIR = DATASET_DIR / "images" / "train"
TRAIN_LABELS_DIR = DATASET_DIR / "labels" / "train"

def extract_page_from_pdf(pdf_path: str, page_number: int, output_path: str, dpi: int = 200) -> bool:
    """Extrait une page d'un PDF en PNG."""
    try:
        # Utiliser pdftoppm ou équivalent
        cmd = [
            "pdftoppm", 
            "-png", 
            "-f", str(page_number), 
            "-l", str(page_number),
            "-r", str(dpi),
            pdf_path, 
            output_path.replace(".png", "")
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print(f"❌ Erreur pdftoppm: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ pdftoppm non trouvé. Installez poppler-utils:")
        print("  macOS: brew install poppler")
        print("  Ubuntu: sudo apt install poppler-utils")
        return False

def add_image_to_dataset(image_path: str, name: str) -> bool:
    """Ajoute une image au dataset d'entraînement."""
    # Vérifier que l'image existe
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        return False
    
    # Générer le nom de fichier de destination
    dest_name = f"{name}.png"
    dest_image = TRAIN_IMAGES_DIR / dest_name
    dest_label = TRAIN_LABELS_DIR / f"{name}.txt"
    
    # Vérifier qu'il n'existe pas déjà
    if dest_image.exists():
        print(f"⚠️ Le fichier existe déjà: {dest_image}")
        response = input("Remplacer ? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # Copier l'image
    try:
        shutil.copy2(image_path, dest_image)
        print(f"✅ Image copiée: {dest_image}")
        
        # Créer un fichier d'annotation vide
        with open(dest_label, 'w') as f:
            f.write("# Fichier d'annotation YOLO (format: class x_center y_center width height)\n")
            f.write("# Classes disponibles: 0=panel, 1=panel_inset\n")
            f.write("# Exemple: 0 0.5 0.3 0.4 0.6\n")
        print(f"✅ Template d'annotation créé: {dest_label}")
        
        print(f"\n🎯 PROCHAINE ÉTAPE:")
        print(f"  1. Ouvrir l'image dans un outil d'annotation")
        print(f"  2. Annoter les panels (class 0) et panels incrustés (class 1)")
        print(f"  3. Sauvegarder au format YOLO dans: {dest_label}")
        print(f"\n💡 Outils recommandés:")
        print(f"  • LabelImg: https://github.com/HumanSignal/labelImg")
        print(f"  • Roboflow: https://roboflow.com")
        print(f"  • CVAT: https://cvat.ai")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la copie: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Ajouter un nouveau fichier d'annotation au dataset Multi-BD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Ajouter une image existante
  python scripts/add_new_annotation.py --image ma_bd_page.png --name ma_bd_p01

  # Extraire une page d'un PDF et l'ajouter
  python scripts/add_new_annotation.py --pdf comics.pdf --page 5 --name comics_p05

  # Extraire avec une résolution spécifique
  python scripts/add_new_annotation.py --pdf comics.pdf --page 10 --name comics_p10 --dpi 300

Format des annotations YOLO:
  Chaque ligne: class x_center y_center width height
  Coordonnées normalisées (0.0 à 1.0)
  Classes: 0=panel, 1=panel_inset
        """
    )
    
    # Options d'entrée mutuellement exclusives
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Chemin vers une image existante")
    input_group.add_argument("--pdf", help="Chemin vers un fichier PDF")
    
    # Options communes
    parser.add_argument("--name", required=True, help="Nom du fichier (sans extension)")
    
    # Options spécifiques au PDF
    parser.add_argument("--page", type=int, help="Numéro de page à extraire (requis avec --pdf)")
    parser.add_argument("--dpi", type=int, default=200, help="Résolution d'extraction (défaut: 200)")
    
    args = parser.parse_args()
    
    # Validation des arguments
    if args.pdf and not args.page:
        parser.error("--page est requis avec --pdf")
    
    # Créer les répertoires si nécessaire
    TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🎯 Ajout d'un nouveau fichier d'annotation au dataset Multi-BD")
    print(f"📁 Dataset: {DATASET_DIR}")
    print(f"🏷️ Nom: {args.name}")
    print("-" * 60)
    
    if args.image:
        # Mode image existante
        print(f"📸 Source: {args.image}")
        success = add_image_to_dataset(args.image, args.name)
        
    elif args.pdf:
        # Mode extraction PDF
        print(f"📄 PDF: {args.pdf}")
        print(f"📄 Page: {args.page}")
        print(f"🔍 DPI: {args.dpi}")
        
        # Extraire la page
        temp_image = f"/tmp/{args.name}.png"
        print(f"📤 Extraction de la page {args.page}...")
        
        if extract_page_from_pdf(args.pdf, args.page, temp_image, args.dpi):
            print(f"✅ Page extraite: {temp_image}")
            success = add_image_to_dataset(temp_image, args.name)
            # Nettoyer le fichier temporaire
            try:
                os.remove(temp_image)
            except:
                pass
        else:
            success = False
    
    if success:
        print(f"\n🎉 Fichier ajouté avec succès au dataset!")
        print(f"\n📊 Prochaines étapes:")
        print(f"  1. Annoter le fichier avec un outil d'annotation")
        print(f"  2. Mettre à jour le dataset: python scripts/update_dataset_stats.py")
        print(f"  3. Re-entraîner le modèle: python scripts/start_optimized_training.py")
        return 0
    else:
        print(f"\n❌ Échec de l'ajout du fichier.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
