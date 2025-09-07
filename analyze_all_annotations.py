#!/usr/bin/env python3
"""
Analyse complète de toutes les annotations disponibles
"""

import sys
import os
import json
import fitz
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch')

def analyze_all_annotations():
    """Analyse complète de toutes les annotations disponibles"""

    print("🔍 ANALYSE COMPLÈTE DE TOUTES LES ANNOTATIONS")
    print("=" * 70)

    annotations_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/backup_annotations_20250822_182146"

    # Lister tous les fichiers JSON
    json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
    print(f"📁 Total fichiers JSON trouvés: {len(json_files)}")

    # Grouper par série
    series = defaultdict(list)
    for file in json_files:
        if file.startswith('pinup_'):
            series['pinup'].append(file)
        elif file.startswith('sisters_'):
            series['sisters'].append(file)
        elif file.startswith('tintin_'):
            series['tintin'].append(file)
        elif file.startswith('p') and not file.startswith('pinup'):
            series['other'].append(file)

    print("\n📊 RÉPARTITION PAR SÉRIE:")
    for serie, files in series.items():
        print(f"   {serie.upper()}: {len(files)} pages")

    # Analyser quelques exemples de chaque série
    print("\n🎯 ANALYSE DÉTAILLÉE PAR SÉRIE:")
    print("=" * 50)

    for serie_name, files in series.items():
        if not files:
            continue

        print(f"\n📋 SÉRIE: {serie_name.upper()}")
        print("-" * 30)

        # Analyser les 3 premiers fichiers de chaque série
        sample_files = files[:3]

        total_panels = 0
        total_balloons = 0
        page_count = 0

        total_panels = 0
        total_balloons = 0
        page_count = 0

        for file in sample_files:
            try:
                json_path = os.path.join(annotations_dir, file)
                with open(json_path, 'r') as f:
                    data = json.load(f)

                panels = [shape for shape in data['shapes'] if shape['label'] == 'panel']
                balloons = [shape for shape in data['shapes'] if shape['label'] == 'balloon']

                total_panels += len(panels)
                total_balloons += len(balloons)
                page_count += 1

                print(f"   {file}: {len(panels)} panels, {len(balloons)} balloons")

                # Analyser les tailles des panels
                if panels:
                    areas = []
                    for panel in panels:
                        x1, y1 = panel['points'][0]
                        x2, y2 = panel['points'][1]
                        area = (x2 - x1) * (y2 - y1)
                        areas.append(area)

                    if areas:
                        avg_area = sum(areas) / len(areas)
                        min_area = min(areas)
                        max_area = max(areas)
                        print(f"      📏 Tailles: min={min_area:.0f}, moy={avg_area:.0f}, max={max_area:.0f}")

            except Exception as e:
                print(f"   ❌ Erreur avec {file}: {e}")

        if page_count > 0:
            avg_panels = total_panels / page_count
            avg_balloons = total_balloons / page_count
            print(f"   📊 Moyenne série: {avg_panels:.1f} panels, {avg_balloons:.1f} balloons par page")
    print("\n🚀 POSSIBILITÉS DE TEST ÉTENDU:")
    print("=" * 50)
    print("📈 TESTS POSSIBLES:")
    print(f"   • Pin-up: {len(series['pinup'])} pages complètes")
    print(f"   • Sisters: {len(series['sisters'])} pages complètes")
    print(f"   • Tintin: {len(series['tintin'])} pages complètes")
    print(f"   • Autres: {len(series['other'])} pages diverses")

    total_pages = sum(len(files) for files in series.values())
    print(f"\n🎯 TOTAL: {total_pages} pages d'annotations disponibles!")

    print("\n💡 RECOMMANDATIONS:")
    print("   1. 🔬 TEST SUR 50+ PAGES: Validation robuste du modèle")
    print("   2. 📊 ANALYSE PAR STYLE: Comparaison entre séries")
    print("   3. 🎯 FOCUS SUR FAIBLESSES: Pages avec moins de 80% précision")
    print("   4. 📈 ÉVOLUTION TEMPORELLE: Amélioration au fil des pages")

    print("\n🎮 PROCHAINES ÉTAPES POSSIBLES:")
    print("   • Test sur 20 pages aléatoires de chaque série")
    print("   • Analyse comparative des styles de BD")
    print("   • Identification des pages les plus difficiles")
    print("   • Validation des paramètres optimaux à grande échelle")

if __name__ == "__main__":
    analyze_all_annotations()
