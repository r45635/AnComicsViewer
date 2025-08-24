#!/usr/bin/env python3
"""
Script pour extraire des pages d'un PDF et les ajouter au dataset Multi-BD
==========================================================================

Ce script extrait automatiquement plusieurs pages d'un PDF et les prépare
pour l'annotation dans le dataset Multi-BD Enhanced.

Usage:
    python scripts/extract_pdf_to_dataset.py --pdf "dataset/pdfs/comic.pdf" --pages 1,3,5-10 --prefix sisters
    python scripts/extract_pdf_to_dataset.py --pdf "dataset/pdfs/comic.pdf" --pages 1-5 --prefix comic --dpi 300

Auteur: Vincent Cruvellier
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import re

# Configuration
DATASET_DIR = Path(__file__).parent.parent / "dataset"
TRAIN_IMAGES_DIR = DATASET_DIR / "images" / "train"
TRAIN_LABELS_DIR = DATASET_DIR / "labels" / "train"

def parse_page_range(page_spec: str) -> list:
    """Parse une spécification de pages comme '1,3,5-10' en liste d'entiers."""
    pages = []
    for part in page_spec.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))  # Supprimer les doublons et trier

def extract_page_from_pdf(pdf_path: str, page_number: int, output_path: str, dpi: int = 200) -> bool:
    """Extrait une page d'un PDF en PNG."""
    try:
        # Utiliser pdftoppm
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
            print(f"❌ Erreur pdftoppm pour page {page_number}: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ pdftoppm non trouvé. Installez poppler-utils:")
        print("  macOS: brew install poppler")
        print("  Ubuntu: sudo apt install poppler-utils")
        return False

def get_pdf_page_count(pdf_path: str) -> int:
    """Récupère le nombre de pages d'un PDF."""
    try:
        cmd = ["pdfinfo", pdf_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Pages:'):
                    return int(line.split(':')[1].strip())
        return 0
    except:
        return 0

def create_annotation_template(label_path: str, image_name: str) -> None:
    """Crée un template d'annotation vide."""
    with open(label_path, 'w') as f:
        f.write(f"# Annotation YOLO pour {image_name}\n")
        f.write("# Format: class x_center y_center width height (coordonnées normalisées 0.0-1.0)\n")
        f.write("# Classes: 0=panel, 1=panel_inset\n")
        f.write("# Exemple: 0 0.5 0.3 0.4 0.6\n")
        f.write("\n")

def main():
    parser = argparse.ArgumentParser(
        description="Extraire des pages d'un PDF pour annotation Multi-BD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Extraire les pages 1, 3, 5 à 10
  python scripts/extract_pdf_to_dataset.py --pdf "dataset/pdfs/comic.pdf" --pages "1,3,5-10" --prefix sisters

  # Extraire les 5 premières pages
  python scripts/extract_pdf_to_dataset.py --pdf "dataset/pdfs/comic.pdf" --pages "1-5" --prefix comic

  # Avec une résolution spécifique
  python scripts/extract_pdf_to_dataset.py --pdf "dataset/pdfs/comic.pdf" --pages "1-3" --prefix comic --dpi 300

  # Pages spécifiques pour annotation
  python scripts/extract_pdf_to_dataset.py --pdf "dataset/pdfs/sisters.pdf" --pages "5,12,18,25" --prefix sisters
        """
    )
    
    parser.add_argument("--pdf", required=True, help="Chemin vers le fichier PDF")
    parser.add_argument("--pages", required=True, help="Pages à extraire (ex: '1,3,5-10')")
    parser.add_argument("--prefix", required=True, help="Préfixe pour les noms de fichiers")
    parser.add_argument("--dpi", type=int, default=200, help="Résolution d'extraction (défaut: 200)")
    parser.add_argument("--dry-run", action="store_true", help="Simulation sans extraction réelle")
    
    args = parser.parse_args()
    
    # Vérifications
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ PDF non trouvé: {pdf_path}")
        return 1
    
    # Parse des pages
    try:
        pages = parse_page_range(args.pages)
    except ValueError as e:
        print(f"❌ Format de pages invalide: {e}")
        return 1
    
    # Vérifier le nombre de pages du PDF
    total_pages = get_pdf_page_count(str(pdf_path))
    if total_pages > 0:
        invalid_pages = [p for p in pages if p < 1 or p > total_pages]
        if invalid_pages:
            print(f"❌ Pages invalides (PDF a {total_pages} pages): {invalid_pages}")
            return 1
    
    # Créer les répertoires si nécessaire
    TRAIN_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🎯 Extraction de pages PDF vers dataset Multi-BD")
    print(f"📄 PDF: {pdf_path}")
    print(f"📄 Pages: {pages}")
    print(f"🏷️ Préfixe: {args.prefix}")
    print(f"🔍 DPI: {args.dpi}")
    print(f"📁 Destination: {TRAIN_IMAGES_DIR}")
    if total_pages > 0:
        print(f"📊 PDF: {total_pages} pages total")
    print("-" * 60)
    
    if args.dry_run:
        print("🧪 MODE SIMULATION (--dry-run)")
        for page_num in pages:
            image_name = f"{args.prefix}_p{page_num:03d}.png"
            label_name = f"{args.prefix}_p{page_num:03d}.txt"
            print(f"   📸 {image_name} ← page {page_num}")
            print(f"   🏷️ {label_name}")
        print(f"\n✅ {len(pages)} fichiers seraient créés.")
        return 0
    
    # Extraction des pages
    success_count = 0
    failed_pages = []
    
    for i, page_num in enumerate(pages, 1):
        image_name = f"{args.prefix}_p{page_num:03d}.png"
        label_name = f"{args.prefix}_p{page_num:03d}.txt"
        
        image_path = TRAIN_IMAGES_DIR / image_name
        label_path = TRAIN_LABELS_DIR / label_name
        
        print(f"📤 [{i}/{len(pages)}] Extraction page {page_num} → {image_name}")
        
        # Vérifier si le fichier existe déjà
        if image_path.exists():
            print(f"   ⚠️ Fichier existe déjà: {image_path}")
            response = input("   Remplacer ? (y/N/a pour tous): ").lower()
            if response == 'a':
                pass  # Remplacer tous
            elif response != 'y':
                print(f"   ⏭️ Ignoré")
                continue
        
        # Extraire la page
        temp_path = f"/tmp/{image_name}"
        if extract_page_from_pdf(str(pdf_path), page_num, temp_path, args.dpi):
            # Le fichier extrait aura un suffixe de page de pdftoppm
            actual_temp = f"/tmp/{args.prefix}_p{page_num:03d}-{page_num:01d}.png"
            if os.path.exists(actual_temp):
                # Déplacer vers le dataset
                import shutil
                shutil.move(actual_temp, image_path)
                
                # Créer le template d'annotation
                create_annotation_template(str(label_path), image_name)
                
                print(f"   ✅ {image_name} créé")
                print(f"   ✅ {label_name} créé")
                success_count += 1
            else:
                print(f"   ❌ Fichier temporaire introuvable: {actual_temp}")
                failed_pages.append(page_num)
        else:
            failed_pages.append(page_num)
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DE L'EXTRACTION")
    print("=" * 60)
    print(f"✅ Pages extraites avec succès: {success_count}/{len(pages)}")
    if failed_pages:
        print(f"❌ Pages échouées: {failed_pages}")
    
    if success_count > 0:
        print(f"\n🎯 PROCHAINES ÉTAPES:")
        print(f"  1. Ouvrir les images dans un outil d'annotation:")
        print(f"     • LabelImg: https://github.com/HumanSignal/labelImg")
        print(f"     • Roboflow: https://roboflow.com")
        print(f"  2. Annoter les panels (classe 0) et panels incrustés (classe 1)")
        print(f"  3. Sauvegarder au format YOLO dans dataset/labels/train/")
        print(f"  4. Mettre à jour les stats: python scripts/update_dataset_stats.py")
        print(f"  5. Re-entraîner: python scripts/start_optimized_training.py")
        
        print(f"\n📁 Fichiers créés:")
        for page_num in pages:
            if page_num not in failed_pages:
                print(f"   📸 {args.prefix}_p{page_num:03d}.png")
                print(f"   🏷️ {args.prefix}_p{page_num:03d}.txt")
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
