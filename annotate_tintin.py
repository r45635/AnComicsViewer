#!/usr/bin/env python3
"""
Annotation Tintin - Guide Spécialisé
Guide d'annotation optimisé pour les pages Tintin avec conseils spécifiques.
"""

import os
import subprocess
from pathlib import Path

def show_tintin_annotation_guide():
    """Affiche le guide d'annotation spécifique à Tintin."""
    
    print("🎯 ANNOTATION TINTIN - GUIDE SPÉCIALISÉ")
    print("=" * 50)
    print()
    
    print("📚 STYLE TINTIN vs GOLDEN CITY:")
    print("=" * 35)
    print("✅ TINTIN (Ligne Claire):")
    print("   • Panels rectangulaires bien définis")
    print("   • Contours noirs épais et nets")
    print("   • Compositions plus simples")
    print("   • Bulles de dialogue rondes/ovales")
    print()
    print("✅ GOLDEN CITY (Franco-Belge moderne):")
    print("   • Panels plus complexes, parfois irréguliers")
    print("   • Effets visuels et dégradés")
    print("   • Compositions dynamiques")
    print()
    
    print("🎯 STRATÉGIE D'ANNOTATION TINTIN:")
    print("=" * 40)
    print("1. 🚀 COMMENCER PAR TINTIN (plus facile)")
    print("   • Panels bien définis = annotations plus rapides")
    print("   • Permet d'atteindre rapidement 30% d'annotations")
    print()
    print("2. 🎨 CLASSES À ANNOTER:")
    print("   • 'panel' : Tous les panels de BD")
    print("   • 'text' : Bulles de dialogue et texte narratif")
    print()
    print("3. ⚡ CONSEILS RAPIDES:")
    print("   • Focus sur les panels principaux")
    print("   • Ignorer les bordures de page")
    print("   • Bulles = 'text', même si dans un panel")
    print("   • Précision importante pour l'entraînement")
    print()

def count_annotation_progress():
    """Compte le progrès des annotations par source."""
    
    labels_dir = Path("dataset/labels/train")
    
    golden_city_total = len(list(Path("dataset/images/train").glob("p*.png")))
    tintin_total = len(list(Path("dataset/images/train").glob("tintin_*.png")))
    
    golden_city_annotated = len([f for f in labels_dir.glob("*.json") 
                                if not f.name.startswith("tintin_")])
    tintin_annotated = len([f for f in labels_dir.glob("*.json") 
                           if f.name.startswith("tintin_")])
    
    print("📊 PROGRESSION ACTUELLE:")
    print("=" * 25)
    print(f"Golden City: {golden_city_annotated}/{golden_city_total} ({golden_city_annotated/golden_city_total*100:.1f}%)")
    print(f"Tintin:      {tintin_annotated}/{tintin_total} ({tintin_annotated/tintin_total*100:.1f}%)")
    print(f"Total:       {golden_city_annotated + tintin_annotated}/{golden_city_total + tintin_total} ({(golden_city_annotated + tintin_annotated)/(golden_city_total + tintin_total)*100:.1f}%)")
    print()
    
    return tintin_annotated, tintin_total

def suggest_annotation_strategy():
    """Suggère une stratégie d'annotation efficace."""
    
    tintin_annotated, tintin_total = count_annotation_progress()
    
    print("🎯 STRATÉGIE RECOMMANDÉE:")
    print("=" * 30)
    
    if tintin_annotated < 10:
        print("🚀 PHASE 1: Annoter 10 pages Tintin")
        print("   • Plus facile que Golden City")
        print("   • Permet de tester le modèle rapidement")
        print("   • Objectif: ~15 annotations en 30min")
        
    elif tintin_annotated < 20:
        print("⚡ PHASE 2: Continuer Tintin (10 pages de plus)")
        print("   • Atteindre 20 pages Tintin annotées")
        print("   • Mélange Tintin + Golden City possible")
        
    elif tintin_annotated < tintin_total // 2:
        print("🎨 PHASE 3: Mix Tintin + Golden City")
        print("   • Diversifier le dataset")
        print("   • Alterner entre les styles")
        
    else:
        print("🏆 PHASE 4: Finalisation")
        print("   • Compléter les annotations manquantes")
        print("   • Prêt pour entraînement final")
    
    print()

def launch_annotation_tool():
    """Lance l'outil d'annotation LabelMe."""
    
    print("🚀 LANCEMENT DE L'ANNOTATION")
    print("=" * 35)
    print()
    print("💡 CONSEILS PENDANT L'ANNOTATION:")
    print("   • Raccourci 'D' : Image suivante")
    print("   • Raccourci 'A' : Image précédente") 
    print("   • Ctrl+S : Sauvegarder")
    print("   • Polygon tool : Clic droit pour finir")
    print("   • Focus sur les pages tintin_p*.png d'abord")
    print()
    
    input("Appuyez sur Entrée pour lancer LabelMe...")
    
    # Lancer LabelMe
    try:
        cmd = [
            ".venv/bin/labelme",
            "dataset/images/train",
            "--output", "dataset/labels/train",
            "--nodata"
        ]
        
        print("Lancement de LabelMe...")
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du lancement: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Annotation interrompue par l'utilisateur")

def post_annotation_summary():
    """Affiche un résumé après annotation."""
    
    print("\n📊 RÉSUMÉ POST-ANNOTATION")
    print("=" * 30)
    
    count_annotation_progress()
    
    total_annotations = len(list(Path("dataset/labels/train").glob("*.json")))
    total_images = len(list(Path("dataset/images/train").glob("*.png")))
    
    progress = total_annotations / total_images * 100
    
    print("🎯 PROCHAINES ÉTAPES:")
    print("=" * 20)
    
    if progress >= 30:
        print("✅ Assez d'annotations pour réentraîner!")
        print("   python tools/labelme_to_yolo.py")
        print("   python continue_training.py")
    elif progress >= 20:
        print("⚡ Encore quelques annotations et c'est bon!")
        print(f"   Annotez {int((0.3 * total_images) - total_annotations)} images de plus")
    else:
        print("📈 Continuez l'annotation...")
        print(f"   Objectif: {int(0.3 * total_images)} annotations minimum")

def main():
    """Fonction principale."""
    
    # Vérifier l'environnement
    if not Path("dataset/images/train").exists():
        print("❌ Dataset non trouvé!")
        return
    
    # Guide d'annotation
    show_tintin_annotation_guide()
    
    # Progression actuelle
    count_annotation_progress()
    
    # Stratégie
    suggest_annotation_strategy()
    
    # Lancer l'annotation
    launch_annotation_tool()
    
    # Résumé final
    post_annotation_summary()

if __name__ == "__main__":
    main()
