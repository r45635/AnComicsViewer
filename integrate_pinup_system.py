#!/usr/bin/env python3
"""
Intégration de "La Pin-up du B24" dans le dataset multi-BD
Version utilisant pdftoppm (système)
"""

import os
import subprocess
from pathlib import Path
import shutil

def check_dependencies():
    """Vérifie que pdftoppm est disponible."""
    try:
        result = subprocess.run(['pdftoppm', '-h'], capture_output=True, text=True)
        return True
    except FileNotFoundError:
        print("❌ pdftoppm non trouvé. Installation requise:")
        print("   macOS: brew install poppler")
        print("   Ou utiliser une autre méthode d'extraction PDF")
        return False

def extract_pinup_pages_system():
    """Extrait les pages avec pdftoppm (système)."""
    
    print("📖 INTÉGRATION DE LA PIN-UP DU B24")
    print("=" * 40)
    
    # Fichiers source et destination
    pdf_path = "La Pin-up du B24 - T01.pdf"
    output_dir = Path("dataset/images/train")
    temp_dir = Path("temp_pinup_extraction")
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF non trouvé: {pdf_path}")
        return 0
    
    print(f"📁 Source: {pdf_path}")
    print(f"📁 Destination: {output_dir}")
    
    # Créer les dossiers
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Extraire avec pdftoppm
        print("🔄 Extraction des pages en cours...")
        
        cmd = [
            'pdftoppm',
            '-png',           # Format PNG
            '-r', '200',      # 200 DPI pour bonne qualité
            '-f', '1',        # Première page
            '-l', '50',       # Dernière page (max 50)
            pdf_path,
            str(temp_dir / 'pinup_page')
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Erreur pdftoppm: {result.stderr}")
            return 0
        
        # Renommer et déplacer les fichiers
        extracted_files = list(temp_dir.glob("pinup_page-*.png"))
        extracted_count = 0
        
        for i, temp_file in enumerate(sorted(extracted_files), 1):
            # Nouveau nom avec format standard
            new_name = f"pinup_p{i:04d}.png"
            final_path = output_dir / new_name
            
            # Déplacer le fichier
            shutil.move(str(temp_file), str(final_path))
            extracted_count += 1
            
            if extracted_count % 10 == 0:
                print(f"   ✅ {extracted_count} pages traitées...")
        
        # Nettoyer le dossier temporaire
        shutil.rmtree(temp_dir)
        
        print(f"✅ Extraction terminée: {extracted_count} pages de La Pin-up du B24")
        return extracted_count
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return 0

def extract_pinup_pages_manual():
    """Instructions pour extraction manuelle."""
    
    print("📖 EXTRACTION MANUELLE - LA PIN-UP DU B24")
    print("=" * 45)
    print()
    print("🔧 MÉTHODE ALTERNATIVE (si pdftoppm indisponible):")
    print()
    print("1. 📱 Ouvre le PDF dans Aperçu (Preview)")
    print("2. 🖼️  Sélectionne les 30-50 premières pages")
    print("3. 📁 Exporte en PNG (Fichier > Exporter)")
    print("4. 📝 Nomme les fichiers: pinup_p0001.png, pinup_p0002.png...")
    print("5. 📂 Place-les dans: dataset/images/train/")
    print()
    print("📋 CONVENTION DE NOMMAGE:")
    print("   pinup_p0001.png")
    print("   pinup_p0002.png") 
    print("   pinup_p0003.png")
    print("   ...")
    print()
    print("💡 Une fois fait, relance ce script pour voir les stats")

def analyze_updated_dataset():
    """Analyse le dataset après ajout potentiel de La Pin-up du B24."""
    
    print("\n📊 STATISTIQUES DU DATASET ACTUEL")
    print("=" * 40)
    
    train_dir = Path("dataset/images/train")
    if not train_dir.exists():
        print("❌ Dossier dataset non trouvé")
        return
    
    # Compter par série
    all_images = list(train_dir.glob("*.png"))
    
    series_counts = {
        "Golden City": len([f for f in all_images if f.name.startswith("p") and not any(x in f.name for x in ["tintin", "pinup"])]),
        "Tintin": len([f for f in all_images if f.name.startswith("tintin_")]),
        "Pin-up du B24": len([f for f in all_images if f.name.startswith("pinup_")])
    }
    
    # Compter les annotations
    labels_dir = Path("dataset/labels/train")
    series_annotations = {
        "Golden City": 0,
        "Tintin": 0, 
        "Pin-up du B24": 0
    }
    
    if labels_dir.exists():
        for json_file in labels_dir.glob("*.json"):
            name = json_file.stem
            if name.startswith("p") and not any(x in name for x in ["tintin", "pinup"]):
                series_annotations["Golden City"] += 1
            elif name.startswith("tintin_"):
                series_annotations["Tintin"] += 1
            elif name.startswith("pinup_"):
                series_annotations["Pin-up du B24"] += 1
    
    # Afficher les statistiques
    total_images = sum(series_counts.values())
    total_annotations = sum(series_annotations.values())
    
    print("Série                | Images | Annotées | Couverture")
    print("-" * 50)
    
    for series in series_counts:
        img_count = series_counts[series]
        ann_count = series_annotations[series]
        coverage = (ann_count / img_count) * 100 if img_count > 0 else 0
        
        print(f"{series:<20} | {img_count:6} | {ann_count:8} | {coverage:7.1f}%")
    
    print("-" * 50)
    print(f"{'TOTAL':<20} | {total_images:6} | {total_annotations:8} | {(total_annotations/total_images)*100:7.1f}%")
    
    # Recommandations d'annotation
    print("\n💡 PRIORITÉS D'ANNOTATION:")
    
    for series in series_counts:
        img_count = series_counts[series]
        ann_count = series_annotations[series]
        
        if img_count == 0:
            continue
            
        coverage = (ann_count / img_count) * 100
        target = min(20, max(10, img_count // 3))  # Cible: 10-20 annotations par série
        needed = max(0, target - ann_count)
        
        if needed > 0:
            priority = "🔴 HAUTE" if coverage < 15 else "🟡 MOYENNE" if coverage < 30 else "🟢 BASSE"
            print(f"   {priority}: {series} - annoter {needed} images de plus")
        else:
            print(f"   ✅ {series}: couverture suffisante ({coverage:.1f}%)")

def main():
    """Fonction principale."""
    
    print("🚀 AJOUT DE LA PIN-UP DU B24 AU DATASET")
    print("=" * 50)
    print()
    
    # Vérifier d'abord s'il y a déjà des fichiers Pin-up
    train_dir = Path("dataset/images/train")
    existing_pinup = len(list(train_dir.glob("pinup_*.png"))) if train_dir.exists() else 0
    
    if existing_pinup > 0:
        print(f"📁 {existing_pinup} fichiers Pin-up déjà présents")
        analyze_updated_dataset()
        return
    
    # Essayer l'extraction automatique
    if check_dependencies():
        extracted = extract_pinup_pages_system()
        if extracted > 0:
            analyze_updated_dataset()
            print(f"\n✅ INTÉGRATION TERMINÉE!")
            print(f"📈 {extracted} nouvelles pages ajoutées")
            print(f"🎯 Prochaine étape: annoter les images Pin-up du B24")
            print(f"💡 Utilise: python start_annotation.py")
        else:
            print("\n🔄 Extraction automatique échouée, passage en manuel...")
            extract_pinup_pages_manual()
    else:
        # Méthode manuelle si pdftoppm indisponible
        extract_pinup_pages_manual()
        analyze_updated_dataset()

if __name__ == "__main__":
    main()
