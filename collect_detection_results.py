#!/usr/bin/env python3
"""
Script pour collecter les résultats de détection sur plusieurs pages
Utilise le script test_detection_debug.py existant
"""

import json
import os
import sys
import subprocess
from pathlib import Path
import argparse
import re

def run_detection_test(pdf_path, page_num, config_path=None):
    """Lance le test de détection pour une page spécifique"""
    cmd = [sys.executable, "test_detection_debug.py", pdf_path, str(page_num)]

    if config_path:
        cmd.extend(["--config", config_path])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def parse_detection_output(output):
    """Parse la sortie du script de test pour extraire les résultats"""
    panels = []
    balloons = []
    image_size = None

    # Chercher les dimensions de l'image
    size_match = re.search(r'imageData.*width["\s:]+(\d+).*height["\s:]+(\d+)', output, re.IGNORECASE)
    if size_match:
        image_size = (int(size_match.group(1)), int(size_match.group(2)))

    # Chercher les panels détectés (format approximatif basé sur les logs)
    # Cette partie devra être adaptée selon le format exact des logs
    panel_matches = re.findall(r'Panel \d+:.*bbox=.*\[([0-9.,\s]+)\]', output)
    for match in panel_matches:
        coords = [float(x.strip()) for x in match.split(',')]
        if len(coords) == 4:
            panels.append(tuple(coords))

    balloon_matches = re.findall(r'Balloon \d+:.*bbox=.*\[([0-9.,\s]+)\]', output)
    for match in balloon_matches:
        coords = [float(x.strip()) for x in match.split(',')]
        if len(coords) == 4:
            balloons.append(tuple(coords))

    return panels, balloons, image_size

def collect_detection_results(pdf_path, pages, config_path=None):
    """Collecte les résultats de détection pour plusieurs pages"""
    results = {}

    for page_num in pages:
        print(f"🔍 Analyse de la page {page_num}...")

        # Lancer le test de détection
        stdout, stderr, returncode = run_detection_test(pdf_path, page_num, config_path)

        if returncode != 0:
            print(f"❌ Erreur lors de l'analyse de la page {page_num}: {stderr}")
            continue

        # Parser les résultats
        panels, balloons, image_size = parse_detection_output(stdout)

        page_name = f"{Path(pdf_path).stem}_p{page_num:04d}"
        results[page_name] = {
            'panels': panels,
            'balloons': balloons,
            'image_width': image_size[0] if image_size else 0,
            'image_height': image_size[1] if image_size else 0,
            'page_num': page_num
        }

        print(f"   ✅ {len(panels)} panels, {len(balloons)} balloons détectés")

    return results

def main():
    parser = argparse.ArgumentParser(description='Collect detection results for multiple pages')
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--pages', required=True, help='Page numbers (comma-separated, e.g., "0,1,2" or "0-5")')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--output', required=True, help='Output JSON file')

    args = parser.parse_args()

    # Parser les numéros de pages
    if '-' in args.pages:
        start, end = map(int, args.pages.split('-'))
        pages = list(range(start, end + 1))
    else:
        pages = [int(p.strip()) for p in args.pages.split(',')]

    print(f"🧪 COLLECTE DE RÉSULTATS DE DÉTECTION")
    print(f"📖 PDF: {args.pdf}")
    print(f"📄 Pages: {pages}")
    print(f"⚙️ Config: {args.config or 'défaut'}")
    print(f"💾 Output: {args.output}")

    # Collecter les résultats
    results = collect_detection_results(args.pdf, pages, args.config)

    if results:
        # Sauvegarder les résultats
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Collecte terminée - {len(results)} pages analysées")
        print(f"💾 Résultats sauvegardés dans: {args.output}")

        # Statistiques globales
        total_panels = sum(len(r['panels']) for r in results.values())
        total_balloons = sum(len(r['balloons']) for r in results.values())
        avg_panels = total_panels / len(results) if results else 0

        print(f"\n📊 STATISTIQUES GLOBALES:")
        print(f"   📄 Pages analysées: {len(results)}")
        print(f"   🎯 Panels totaux: {total_panels} (moyenne: {avg_panels:.1f} par page)")
        print(f"   💬 Balloons totaux: {total_balloons}")
    else:
        print("❌ Aucun résultat collecté")

if __name__ == "__main__":
    main()
