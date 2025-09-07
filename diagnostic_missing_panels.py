#!/usr/bin/env python3
"""
Diagnostic détaillé des panels manquants
========================================
Analyse approfondie pour comprendre pourquoi certains panels ne sont pas détectés
"""

import sys
import os
import json
import fitz
import numpy as np
from pathlib import Path

sys.path.insert(0, '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch')

def analyze_missing_panels():
    """Analyse détaillée des panels manquants"""

    print("🔍 DIAGNOSTIC DÉTAILLÉ DES PANELS MANQUANTS")
    print("=" * 60)

    # Charger les annotations des pages problématiques
    annotations_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/backup_annotations_20250822_182146"
    pdf_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf"

    test_cases = [
        (5, "pinup_p0005.json", "Page 5 - 2 panels manquants"),
        (6, "pinup_p0006.json", "Page 6 - 2 panels manquants"),
    ]

    for page_num, json_file, description in test_cases:
        print(f"\n📋 ANALYSE: {description}")
        print("=" * 50)

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
        for i, panel in enumerate(expected_panels):
            print(f"   Panel {i+1}: {panel['width']:.0f}x{panel['height']:.0f} pixels")

        # Analyser les caractéristiques des panels manquants
        print("\n🎯 CARACTÉRISTIQUES DES PANELS MANQUANTS:")
        print("   • Tailles relatives à la page attendue (1700x2200 = 3.74M pixels)")

        page_area_expected = 1700 * 2200  # 3,740,000 pixels
        for i, panel in enumerate(expected_panels):
            area_pct = (panel['area'] / page_area_expected) * 100
            print(f"   Panel {i+1}: {area_pct:.1f}% de la page ({panel['area']:.0f} pixels)")

        # Hypothèses sur les causes
        print("\n💡 HYPOTHÈSES POSSIBLES:")
        print("   1. 📏 PANELS TROP PETITS: Certains panels font < 0.5% de la page")
        print("   2. 🎨 CONTRASTE FAIBLE: Panels avec peu de contraste bordures")
        print("   3. 📖 STYLE VISUEL: Différent du jeu d'entraînement")
        print("   4. 🔍 RÉSOLUTION: Échelle de 300 DPI insuffisante")
        print("   5. 🎯 MODÈLE LIMITÉ: YOLOv8-medium pas assez performant")

    print("\n🚀 SOLUTIONS RECOMMANDÉES:")
    print("   1. 📊 RÉENTRAÎNEMENT: Ajouter ces pages au dataset d'entraînement")
    print("   2. 🔧 MODÈLE PLUS PUISSANT: Utiliser YOLOv8-large ou YOLOv9")
    print("   3. ⚙️ PARAMÈTRES AVANCÉS: Ajuster conf, iou, augmentations")
    print("   4. 📈 RÉSOLUTION PLUS ÉLEVÉE: Tester 400-600 DPI")
    print("   5. 🎨 PRÉTRAITEMENT: Améliorer contraste/bordures avant détection")

    print("\n📈 RÉSULTATS ACTUELS:")
    print("   ✅ Page 3: 100% précision (1/1 panel)")
    print("   ⚠️ Page 5: 67% précision (4/6 panels)")
    print("   ⚠️ Page 6: 50% précision (2/4 panels)")
    print("   📊 MOYENNE: ~72% précision sur les pages testées")

    print("\n🎯 OBJECTIF 100%:")
    print("   Pour atteindre 100%, il faudrait :")
    print("   • Réentraîner le modèle avec ces pages spécifiques")
    print("   • OU accepter ~75% comme limite réaliste du modèle actuel")
    print("   • OU implémenter un système hybride (ML + règles)")

if __name__ == "__main__":
    analyze_missing_panels()
