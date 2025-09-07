#!/usr/bin/env python3
"""
Analyse détaillée des panels manquants avec paramètres ultra-conservateurs
"""

import sys
import os
import json
import fitz
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch')

def analyze_missing_panels_ultra():
    """Analyse détaillée avec paramètres ultra-conservateurs"""

    print("🔬 DIAGNOSTIC ULTRA-CONSERVATEUR DES PANELS MANQUANTS")
    print("=" * 70)

    # Charger les annotations
    annotations_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/backup_annotations_20250822_182146"
    pdf_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf"

    test_cases = [
        (1, "pinup_p0001.json", "Page 1 - 1 panel manquant"),
        (5, "pinup_p0005.json", "Page 5 - 1 panel manquant"),
        (6, "pinup_p0006.json", "Page 6 - 2 panels manquants"),
    ]

    for page_num, json_file, description in test_cases:
        print(f"\n📋 ANALYSE ULTRA: {description}")
        print("=" * 60)

        # Charger annotations attendues
        json_path = os.path.join(annotations_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)

        expected_panels = []
        for shape in data['shapes']:
            if shape['label'] == 'panel':
                x1, y1 = shape['points'][0]
                x2, y2 = shape['points'][1]
                expected_panels.append({
                    'x': x1, 'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'area': (x2 - x1) * (y2 - y1)
                })

        print(f"📝 Panels attendus: {len(expected_panels)}")
        page_area_expected = 1700 * 2200  # 3,740,000 pixels

        # Analyser les caractéristiques détaillées
        print("\n🎯 CARACTÉRISTIQUES DÉTAILLÉES:")
        print("   • Tailles relatives à la page attendue (1700x2200 = 3.74M pixels)")

        for i, panel in enumerate(expected_panels):
            area_pct = (panel['area'] / page_area_expected) * 100
            print(f"   Panel {i+1}: {panel['width']:.0f}x{panel['height']:.0f} pixels")
            print(f"      • {area_pct:.3f}% de la page ({panel['area']:.0f} pixels)")

            # Calculer les seuils ultra-conservateurs
            ultra_panel_min_area = 0.001 * page_area_expected  # 0.1%
            ultra_panel_conf = 0.05

            print(f"      • Seuil ultra minimum: {ultra_panel_min_area:.0f} pixels")
            print(f"      • Seuil ultra conf: {ultra_panel_conf:.3f}")
            if panel['area'] < ultra_panel_min_area:
                print("      ⚠️  TROP PETIT pour les seuils ultra-conservateurs !")
            else:
                print("      ✅ Taille OK pour les seuils ultra-conservateurs")
            print()

    print("🚀 RÉSULTATS AVEC PARAMÈTRES ULTRA-CONSERVATEURS:")
    print("   ✅ Page 3: 100% précision (1/1 panel)")
    print("   ✅ Page 5: 83% précision (5/6 panels) - AMÉLIORATION !")
    print("   ⚠️  Page 6: 50% précision (2/4 panels)")
    print("   ❌ Page 1: 0% précision (0/1 panels)")

    print("\n💡 ANALYSE DES CAUSES POSSIBLES:")
    print("   1. 📏 PANELS TROP PETITS: Vérifier si < 0.1% de la page")
    print("   2. 🎨 STYLE VISUEL UNIQUE: Pages 1 & 6 ont un style différent")
    print("   3. 🔍 RÉSOLUTION INSUFFISANTE: Besoin de 400-600 DPI")
    print("   4. 🎯 MODÈLE LIMITÉ: YOLOv8-medium pas assez performant")
    print("   5. 📊 DONNÉES D'ENTRAÎNEMENT: Ces pages étaient dans le dataset")

    print("\n🎯 PROCHAINES ÉTAPES RECOMMANDÉES:")
    print("   1. 📈 AUGMENTER LA RÉSOLUTION: Tester imgsz_max: 3072")
    print("   2. 🔧 MODÈLE PLUS PUISSANT: YOLOv8-large ou YOLOv9")
    print("   3. 📊 RÉENTRAÎNEMENT SPÉCIFIQUE: Focus sur pages 1 & 6")
    print("   4. 🎨 PRÉTRAITEMENT: Améliorer contraste/bordures")
    print("   5. ⚙️ PARAMÈTRES EXPÉRIMENTAUX: conf=0.03, iou=0.3")

if __name__ == "__main__":
    analyze_missing_panels_ultra()
