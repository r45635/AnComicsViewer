#!/usr/bin/env python3
"""
Script de comparaison systématique entre détection automatique et annotations ground truth
"""

import json
import os
import sys
from pathlib import Path
import argparse

def load_ground_truth(annotations_dir, page_name):
    """Charge l'annotation ground truth pour une page donnée"""
    # Extraire le numéro de page du nom (format: "NomDuComic_pXXXX")
    import re
    match = re.search(r'_p(\d+)$', page_name)
    if not match:
        # Essayer l'ancien format
        match = re.search(r'p(\d+)$', page_name)
    if not match:
        return None

    page_num = int(match.group(1))
    page_str = "04d"

    # Essayer différents patterns de nom de fichier
    possible_names = [
        f"tintin_p{page_str}.json",
        f"pinup_p{page_str}.json",
        f"sisters_p{page_str}.json",
        f"p{page_str}.json"
    ]

    for name in possible_names:
        json_file = os.path.join(annotations_dir, name)
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            panels = []
            balloons = []

            for shape in data['shapes']:
                if shape['label'] == 'panel':
                    # Convertir rectangle en bbox (x1, y1, x2, y2)
                    x1, y1 = shape['points'][0]
                    x2, y2 = shape['points'][1]
                    panels.append((x1, y1, x2, y2))
                elif shape['label'] == 'balloon':
                    x1, y1 = shape['points'][0]
                    x2, y2 = shape['points'][1]
                    balloons.append((x1, y1, x2, y2))

            return {
                'panels': panels,
                'balloons': balloons,
                'image_width': data.get('imageWidth', 0),
                'image_height': data.get('imageHeight', 0)
            }

    return None

def calculate_iou(box1, box2):
    """Calcule l'IoU entre deux boîtes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def compare_detection_with_ground_truth(detection_results, ground_truth):
    """Compare les résultats de détection avec le ground truth"""
    if not ground_truth:
        return {
            'status': 'no_ground_truth',
            'detected_panels': len(detection_results.get('panels', [])),
            'detected_balloons': len(detection_results.get('balloons', [])),
            'ground_truth_panels': 0,
            'ground_truth_balloons': 0
        }

    gt_panels = ground_truth['panels']
    gt_balloons = ground_truth['balloons']

    detected_panels = detection_results.get('panels', [])
    detected_balloons = detection_results.get('balloons', [])

    # Calculer les correspondances pour les panels
    panel_matches = []
    for i, det_panel in enumerate(detected_panels):
        best_iou = 0.0
        best_match = -1
        for j, gt_panel in enumerate(gt_panels):
            iou = calculate_iou(det_panel, gt_panel)
            if iou > best_iou:
                best_iou = iou
                best_match = j
        panel_matches.append((i, best_match, best_iou))

    # Calculer les correspondances pour les balloons
    balloon_matches = []
    for i, det_balloon in enumerate(detected_balloons):
        best_iou = 0.0
        best_match = -1
        for j, gt_balloon in enumerate(gt_balloons):
            iou = calculate_iou(det_balloon, gt_balloon)
            if iou > best_iou:
                best_iou = iou
                best_match = j
        balloon_matches.append((i, best_match, best_iou))

    # Statistiques
    panel_precision = sum(1 for _, _, iou in panel_matches if iou >= 0.5) / len(panel_matches) if panel_matches else 0
    panel_recall = sum(1 for _, _, iou in panel_matches if iou >= 0.5) / len(gt_panels) if gt_panels else 0

    balloon_precision = sum(1 for _, _, iou in balloon_matches if iou >= 0.5) / len(balloon_matches) if balloon_matches else 0
    balloon_recall = sum(1 for _, _, iou in balloon_matches if iou >= 0.5) / len(gt_balloons) if gt_balloons else 0

    return {
        'status': 'compared',
        'detected_panels': len(detected_panels),
        'detected_balloons': len(detected_balloons),
        'ground_truth_panels': len(gt_panels),
        'ground_truth_balloons': len(gt_balloons),
        'panel_precision': panel_precision,
        'panel_recall': panel_recall,
        'balloon_precision': balloon_precision,
        'balloon_recall': balloon_recall,
        'panel_matches': panel_matches,
        'balloon_matches': balloon_matches
    }

def main():
    parser = argparse.ArgumentParser(description='Compare detection results with ground truth annotations')
    parser.add_argument('--annotations-dir', required=True, help='Directory containing ground truth JSON files')
    parser.add_argument('--results-file', required=True, help='JSON file with detection results')
    parser.add_argument('--output-file', help='Output file for comparison results')

    args = parser.parse_args()

    # Charger les résultats de détection
    if not os.path.exists(args.results_file):
        print(f"❌ Fichier de résultats non trouvé: {args.results_file}")
        return

    with open(args.results_file, 'r', encoding='utf-8') as f:
        detection_results = json.load(f)

    # Comparer chaque page
    comparisons = {}
    for page_name, results in detection_results.items():
        ground_truth = load_ground_truth(args.annotations_dir, page_name)
        comparison = compare_detection_with_ground_truth(results, ground_truth)
        comparisons[page_name] = comparison

        print(f"\n📊 {page_name}:")
        if comparison['status'] == 'no_ground_truth':
            print(f"   ❌ Pas d'annotation ground truth")
        else:
            print(f"   🎯 Panels: {comparison['detected_panels']} détectés vs {comparison['ground_truth_panels']} annotés")
            print(f"   💬 Balloons: {comparison['detected_balloons']} détectés vs {comparison['ground_truth_balloons']} annotés")
            if comparison['detected_panels'] > 0:
                print(".3f")
            if comparison['detected_balloons'] > 0:
                print(".3f")

    # Sauvegarder les résultats
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(comparisons, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Résultats sauvegardés dans: {args.output_file}")

if __name__ == "__main__":
    main()
