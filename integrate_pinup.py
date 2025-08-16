#!/usr/bin/env python3
"""
Intégration de "La Pin-up du B24" dans le dataset multi-BD
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
import shutil

def extract_pinup_pages():
    """Extrait les pages de La Pin-up du B24 et les ajoute au dataset."""
    
    print("📖 INTÉGRATION DE LA PIN-UP DU B24")
    print("=" * 40)
    
    # Fichiers source et destination
    pdf_path = "La Pin-up du B24 - T01.pdf"
    output_dir = Path("dataset/images/train")
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF non trouvé: {pdf_path}")
        return 0
    
    print(f"📁 Source: {pdf_path}")
    print(f"📁 Destination: {output_dir}")
    
    # Créer le dossier si nécessaire
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ouvrir le PDF
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"📄 Pages totales dans le PDF: {total_pages}")
        
        extracted = 0
        max_pages = min(50, total_pages)  # Limiter à 50 pages max
        
        print(f"🎯 Extraction de {max_pages} pages...")
        
        for page_num in range(max_pages):
            page = doc.load_page(page_num)
            
            # Convertir en image haute résolution
            mat = fitz.Matrix(2.0, 2.0)  # Zoom 2x pour meilleure qualité
            pix = page.get_pixmap(matrix=mat)
            
            # Nom du fichier de sortie
            output_file = output_dir / f"pinup_p{page_num + 1:04d}.png"
            
            # Sauvegarder
            pix.save(str(output_file))
            extracted += 1
            
            if extracted % 10 == 0:
                print(f"   ✅ {extracted}/{max_pages} pages extraites...")
        
        doc.close()
        print(f"✅ Extraction terminée: {extracted} pages de La Pin-up du B24")
        
        return extracted
        
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction: {e}")
        return 0

def analyze_updated_dataset():
    """Analyse le dataset après ajout de La Pin-up du B24."""
    
    print("\n📊 STATISTIQUES DU DATASET MIS À JOUR")
    print("=" * 45)
    
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
    
    # Recommandations
    print("\n💡 RECOMMANDATIONS POUR L'ANNOTATION:")
    print("🎯 Priorités d'annotation (pour équilibrer le dataset):")
    
    # Calculer les besoins d'annotation
    target_annotations_per_series = max(15, total_images // 10)  # Au moins 15 par série
    
    priorities = []
    for series in series_counts:
        img_count = series_counts[series]
        ann_count = series_annotations[series]
        if img_count > 0:
            coverage = (ann_count / img_count) * 100
            needed = max(0, min(target_annotations_per_series, img_count) - ann_count)
            priorities.append((coverage, needed, series))
    
    # Trier par priorité (couverture la plus faible d'abord)
    priorities.sort()
    
    for i, (coverage, needed, series) in enumerate(priorities):
        priority_level = ["🔴 HAUTE", "🟡 MOYENNE", "🟢 BASSE"][min(i, 2)]
        if needed > 0:
            print(f"   {priority_level}: {series} - annoter {needed} images de plus")
        else:
            print(f"   ✅ {series}: couverture suffisante ({coverage:.1f}%)")
    
    print(f"\n🎨 STYLES DE BD DANS LE DATASET:")
    print(f"   • Golden City: Style moderne, panels complexes")
    print(f"   • Tintin: Style classique, panels simples")
    print(f"   • Pin-up du B24: Style aviation/guerre, mix moderne-rétro")
    print(f"   ➡️  Bonne diversité pour un modèle robuste!")

def main():
    """Fonction principale."""
    
    print("🚀 AJOUT DE LA PIN-UP DU B24 AU DATASET")
    print("=" * 50)
    print()
    
    # 1. Extraire les pages
    extracted = extract_pinup_pages()
    
    if extracted > 0:
        # 2. Analyser le dataset mis à jour
        analyze_updated_dataset()
        
        print(f"\n✅ INTÉGRATION TERMINÉE!")
        print(f"📈 {extracted} nouvelles pages ajoutées")
        print(f"🎯 Prochaine étape: annoter les images Pin-up du B24")
        print(f"💡 Utilise: python start_annotation.py")
    else:
        print("❌ Échec de l'intégration")

if __name__ == "__main__":
    main()
