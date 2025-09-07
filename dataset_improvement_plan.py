#!/usr/bin/env python3
"""
Stratégie d'amélioration du dataset pour atteindre 100% de précision
"""

import sys
import os
import json
import shutil
from pathlib import Path

sys.path.insert(0, '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch')

def analyze_problematic_pages():
    """Analyse détaillée des pages problématiques"""

    print("🔍 ANALYSE DES PAGES PROBLÉMATIQUES")
    print("=" * 60)

    annotations_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/backup_annotations_20250822_182146"
    pdf_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf"

    # Pages problématiques identifiées
    problematic_pages = [
        ("pinup_p0001.json", "Page 1 - 0/1 panel détecté"),
        ("pinup_p0006.json", "Page 6 - 2/4 panels détectés"),
    ]

    print("📋 PAGES À ANALYSER:")
    for page_file, description in problematic_pages:
        print(f"   • {page_file}: {description}")

    print("\n🎯 ANALYSE DÉTAILLÉE:")
    print("-" * 40)

    for page_file, description in problematic_pages:
        json_path = os.path.join(annotations_dir, page_file)

        if not os.path.exists(json_path):
            print(f"   ❌ {page_file}: fichier non trouvé")
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            panels = [s for s in data['shapes'] if s['label'] == 'panel']
            balloons = [s for s in data['shapes'] if s['label'] == 'balloon']

            print(f"\n📄 {page_file}")
            print(f"   Description: {description}")
            print(f"   Panels attendus: {len(panels)}")
            print(f"   Balloons attendus: {len(balloons)}")

            if panels:
                print("   📏 Caractéristiques des panels:")
                for i, panel in enumerate(panels):
                    x1, y1 = panel['points'][0]
                    x2, y2 = panel['points'][1]
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 0

                    print(f"      Panel {i+1}: {width:.0f}x{height:.0f} ({area:.0f}px², ratio={aspect_ratio:.2f})")

            if balloons:
                print("   💬 Caractéristiques des balloons:")
                for i, balloon in enumerate(balloons):
                    x1, y1 = balloon['points'][0]
                    x2, y2 = balloon['points'][1]
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    print(f"      Balloon {i+1}: {width:.0f}x{height:.0f} ({area:.0f}px²)")

        except Exception as e:
            print(f"   ❌ Erreur avec {page_file}: {e}")

def create_improvement_plan():
    """Crée un plan d'amélioration du dataset"""

    print("\n🚀 PLAN D'AMÉLIORATION DU DATASET")
    print("=" * 60)

    print("📊 PHASE 1: ANALYSE ACTUELLE")
    print("   • Dataset: 142 pages d'annotations")
    print("   • Styles: Pin-up, Sisters, Tintin, Autres")
    print("   • Problèmes: Pages 1 & 6 de Pin-up")

    print("\n🎯 PHASE 2: STRATÉGIE D'AMÉLIORATION")

    print("\n   2.1 AUGMENTATION DES DONNÉES DIFFICILES:")
    print("   • Dupliquer 50x les pages problématiques")
    print("   • Appliquer des augmentations variées:")
    print("     - Rotation: ±5°, ±10°, ±15°")
    print("     - Échelle: 0.9x, 1.1x, 1.2x")
    print("     - Contraste: 0.8x, 1.2x, 1.5x")
    print("     - Flou: léger flou gaussien")
    print("     - Bruit: bruit léger")

    print("\n   2.2 RÉÉQUILIBRAGE DU DATASET:")
    print("   • S'assurer que chaque style représente ~25%")
    print("   • Ajouter plus d'exemples de pages simples (1 panel)")
    print("   • Équilibrer panels/balloons par page")

    print("\n   2.3 AMÉLIORATION DES ANNOTATIONS:")
    print("   • Vérifier la précision des boîtes de délimitation")
    print("   • Ajouter des annotations pour les cas limites")
    print("   • Standardiser les labels et formats")

    print("\n🔧 PHASE 3: RÉENTRAÎNEMENT")

    print("\n   3.1 CONFIGURATION D'ENTRAÎNEMENT:")
    print("   • Modèle: YOLOv8-medium (ou large si nécessaire)")
    print("   • Epochs: 50-100")
    print("   • Batch size: 16-32")
    print("   • Learning rate: 0.001 initial, decay")
    print("   • Augmentations: activées")

    print("\n   3.2 VALIDATION:")
    print("   • Validation croisée sur 20% du dataset")
    print("   • Test sur toutes les 142 pages")
    print("   • Métriques: mAP@0.5, précision, rappel")

    print("\n📈 PHASE 4: OPTIMISATION FINALE")

    print("\n   4.1 AJUSTEMENTS POST-ENTRAÎNEMENT:")
    print("   • Fine-tuning des seuils de confiance")
    print("   • Optimisation des paramètres de post-processing")
    print("   • Test de différentes résolutions")

    print("\n   4.2 VALIDATION COMPLÈTE:")
    print("   • Test sur l'ensemble des 142 pages")
    print("   • Analyse des erreurs restantes")
    print("   • Comparaison avec les résultats actuels")

def create_action_script():
    """Crée un script d'actions concrètes"""

    print("\n⚡ SCRIPT D'ACTIONS CONCRÈTES")
    print("=" * 60)

    actions = [
        "1. 📁 CRÉER DOSSIER AMÉLIORÉ",
        "   mkdir -p dataset_improved/{images,labels}",
        "",
        "2. 🔄 COPIER DONNÉES EXISTANTES",
        "   cp -r backup_annotations_20250822_182146/* dataset_improved/",
        "",
        "3. 📈 AUGMENTER LES PAGES DIFFICILES",
        "   # Pour chaque page problématique:",
        "   for i in {1..50}; do",
        "     # Appliquer rotation, échelle, contraste",
        "     convert page1.jpg -rotate $((RANDOM%30-15)) -resize 110% page1_aug${i}.jpg",
        "   done",
        "",
        "4. 📊 VÉRIFIER L'ÉQUILIBRE",
        "   # Compter les exemples par classe",
        "   find dataset_improved -name '*.json' | wc -l",
        "",
        "5. 🚀 PRÉPARER POUR YOLO",
        "   # Convertir format JSON vers YOLO",
        "   python scripts/json_to_yolo.py",
        "",
        "6. 🎯 LANCER L'ENTRAÎNEMENT",
        "   yolo train \\",
        "     model=yolov8m.pt \\",
        "     data=dataset_improved/data.yaml \\",
        "     epochs=100 \\",
        "     imgsz=640 \\",
        "     batch=16",
        "",
        "7. ✅ VALIDER LES RÉSULTATS",
        "   # Tester sur toutes les pages",
        "   python validate_all_pages.py"
    ]

    for action in actions:
        print(f"   {action}")

def main():
    """Fonction principale"""
    analyze_problematic_pages()
    create_improvement_plan()
    create_action_script()

    print("\n🎯 RÉSULTAT ATTENDU:")
    print("   • Précision: 95%+ sur l'ensemble du dataset")
    print("   • Robustesse: Fonctionne sur tous les styles de BD")
    print("   • Fiabilité: Détection stable et cohérente")

    print("\n💡 PROCHAINES ÉTAPES:")
    print("   1. Créer le dossier dataset_improved/")
    print("   2. Lancer l'analyse détaillée des pages problématiques")
    print("   3. Commencer l'augmentation des données")
    print("   4. Préparer le réentraînement")

if __name__ == "__main__":
    main()
