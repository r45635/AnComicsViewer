#!/usr/bin/env python3
"""
Script simple pour collecter les résultats de détection
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

def extract_detection_stats(output):
    """Extrait les statistiques de détection depuis les logs"""
    panels_count = 0
    balloons_count = 0

    # Chercher les statistiques finales
    final_match = re.search(r'RÉSULTAT FINAL.*?Panels:\s*(\d+).*?Balloons:\s*(\d+)', output, re.DOTALL)
    if final_match:
        panels_count = int(final_match.group(1))
        balloons_count = int(final_match.group(2))

    return panels_count, balloons_count

def collect_detection_results_simple(pdf_path, pages, config_path=None):
    """Collecte les résultats de détection pour plusieurs pages"""
    results = {}

    for page_num in pages:
        print(f"🔍 Analyse de la page {page_num}...")

        # Lancer le test de détection
        stdout, stderr, returncode = run_detection_test(pdf_path, page_num, config_path)

        if returncode != 0:
            print(f"❌ Erreur lors de l'analyse de la page {page_num}: {stderr}")
            continue

        # Extraire les statistiques
        panels_count, balloons_count = extract_detection_stats(stdout)

        page_name = f"{Path(pdf_path).stem}_p{page_num:04d}"
        results[page_name] = {
            'panels_count': panels_count,
            'balloons_count': balloons_count,
            'page_num': page_num,
            'raw_output': stdout  # Garder la sortie complète pour debug
        }

        print(f"   ✅ {panels_count} panels, {balloons_count} balloons détectés")

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

    print("🧪 COLLECTE DE RÉSULTATS DE DÉTECTION")
    print(f"📖 PDF: {args.pdf}")
    print(f"📄 Pages: {pages}")
    print(f"⚙️ Config: {args.config or 'défaut'}")
    print(f"💾 Output: {args.output}")

    # Collecter les résultats
    results = collect_detection_results_simple(args.pdf, pages, args.config)

    if results:
        # Sauvegarder les résultats
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Collecte terminée - {len(results)} pages analysées")
        print(f"💾 Résultats sauvegardés dans: {args.output}")

        # Statistiques globales
        total_panels = sum(r['panels_count'] for r in results.values())
        total_balloons = sum(r['balloons_count'] for r in results.values())
        avg_panels = total_panels / len(results) if results else 0

        print(f"\n📊 STATISTIQUES GLOBALES:")
        print(f"   📄 Pages analysées: {len(results)}")
        print(f"   🎯 Panels totaux: {total_panels} (moyenne: {avg_panels:.1f} par page)")
        print(f"   💬 Balloons totaux: {total_balloons}")
    else:
        print("❌ Aucun résultat collecté")

if __name__ == "__main__":
    main()
