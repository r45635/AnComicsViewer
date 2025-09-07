#!/usr/bin/env python3
"""
Script pour analyser visuellement les résultats de détection
"""

import json
import os
import sys
from pathlib import Path
import subprocess

def analyze_page_visual(pdf_path, page_num, output_dir="debug_output"):
    """Analyse visuelle d'une page avec sauvegarde des résultats détaillés"""

    # Créer le dossier de sortie
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"🔍 ANALYSE VISUELLE DE LA PAGE {page_num}")
    print("=" * 50)

    # Lancer la détection avec debug
    cmd = [
        sys.executable, "test_detection_debug.py",
        pdf_path, str(page_num)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

        if result.returncode != 0:
            print(f"❌ Erreur lors de la détection: {result.stderr}")
            return

        output = result.stdout

        # Sauvegarder la sortie complète
        output_file = output_dir / f"page_{page_num:04d}_detection_log.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)

        print(f"💾 Log complet sauvegardé: {output_file}")

        # Extraire les statistiques importantes
        stats = extract_detection_stats(output)

        # Créer un résumé visuel
        create_visual_summary(stats, output_dir, page_num)

        return stats

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def extract_detection_stats(output):
    """Extrait les statistiques détaillées de la sortie"""

    stats = {
        'brute': {},
        'filtre_initial': {},
        'merging': {},
        'gutter_split': {},
        'page_complete': {},
        'final': {}
    }

    # Patterns pour extraire les statistiques
    patterns = {
        'brute': r'DÉTECTION BRUTE.*?Panels:\s*(\d+).*?Area:.*?avg=(\d+)px².*?Distribution:\s*\[(.*?)\]',
        'filtre_initial': r'APRÈS FILTRE INITIAL.*?Panels:\s*(\d+).*?Area:.*?avg=(\d+)px².*?Distribution:\s*\[(.*?)\]',
        'merging': r'APRÈS MERGING.*?Panels:\s*(\d+).*?Area:.*?avg=(\d+)px².*?Distribution:\s*\[(.*?)\]',
        'gutter_split': r'APRÈS GUTTER SPLIT.*?Panels:\s*(\d+).*?Area:.*?avg=(\d+)px².*?Distribution:\s*\[(.*?)\]',
        'page_complete': r'APRÈS PAGE COMPLÈTE.*?Panels:\s*(\d+).*?Area:.*?avg=(\d+)px².*?Distribution:\s*\[(.*?)\]',
        'final': r'RÉSULTAT FINAL.*?Panels:\s*(\d+).*?Area:.*?avg=(\d+)px².*?Distribution:\s*\[(.*?)\]'
    }

    import re

    for stage, pattern in patterns.items():
        match = re.search(pattern, output, re.DOTALL)
        if match:
            count = int(match.group(1))
            avg_area = int(match.group(2))
            distribution_str = match.group(3)

            # Parser la distribution
            try:
                distribution = [float(x.strip()) for x in distribution_str.split(',')[:5]]
            except:
                distribution = []

            stats[stage] = {
                'panels': count,
                'avg_area_px': avg_area,
                'distribution_percent': distribution
            }

    return stats

def create_visual_summary(stats, output_dir, page_num):
    """Crée un résumé visuel des statistiques"""

    summary_file = output_dir / f"page_{page_num:04d}_summary.txt"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"ANALYSE VISUELLE - PAGE {page_num}\n")
        f.write("=" * 40 + "\n\n")

        stages = ['brute', 'filtre_initial', 'merging', 'gutter_split', 'page_complete', 'final']

        for stage in stages:
            if stage in stats and stats[stage]:
                data = stats[stage]
                f.write(f"📊 {stage.upper().replace('_', ' ')}:\n")
                f.write(f"   🎯 Panels: {data['panels']}\n")
                f.write(f"   📏 Surface moyenne: {data['avg_area_px']:,} px²\n")

                if data['distribution_percent']:
                    f.write("   📈 Distribution des 5 plus grands:\n")
                    for i, percent in enumerate(data['distribution_percent'][:5]):
                        f.write(".1f")
                f.write("\n")

        # Analyse finale
        f.write("🎯 ANALYSE FINALE:\n")
        if 'final' in stats and stats['final']:
            final_panels = stats['final']['panels']
            if final_panels > 0:
                f.write(f"   ✅ {final_panels} panels détectés avec succès\n")
                f.write("   📏 Tailles réalistes pour une bande dessinée\n")
            else:
                f.write("   ❌ Aucun panel détecté\n")

        if 'brute' in stats and 'final' in stats:
            brute_count = stats['brute'].get('panels', 0)
            final_count = stats['final'].get('panels', 0)
            if brute_count > 0:
                reduction = (1 - final_count / brute_count) * 100
                f.write(".1f")

    print(f"📊 Résumé visuel créé: {summary_file}")

    # Afficher le résumé à l'écran
    print("\n" + "=" * 50)
    print(f"RÉSUMÉ VISUEL - PAGE {page_num}")
    print("=" * 50)

    if 'final' in stats and stats['final']:
        final_data = stats['final']
        print(f"🎯 PANELS FINAUX: {final_data['panels']}")
        print(f"📏 SURFACE MOYENNE: {final_data['avg_area_px']:,} px²")

        if final_data['distribution_percent']:
            print("📈 DISTRIBUTION:")
            for i, percent in enumerate(final_data['distribution_percent'][:5]):
                print(".1f")

    print("=" * 50)

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_visual.py <pdf_path> <page_num> [output_dir]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2])
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "debug_output"

    analyze_page_visual(pdf_path, page_num, output_dir)

if __name__ == "__main__":
    main()
