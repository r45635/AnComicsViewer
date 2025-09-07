#!/usr/bin/env python3
"""
Test Automatique Page 5 Tintin - Analyse Détaillée avec KPIs
Script complet pour analyser la précision de détection avec métriques avancées
"""

import sys
import os
import json
import time
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Ajouter le répertoire courant au path
sys.path.append('.')

# Importer les modules nécessaires
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
import fitz

# Importer notre application
from main import PdfYoloViewer

class DetailedTintinTest:
    def __init__(self):
        self.app = None
        self.viewer = None
        self.test_results = {}
        self.ground_truth_panels = []
        self.ground_truth_balloons = []
        self.detected_panels = []
        self.detected_balloons = []

    def load_ground_truth(self) -> Tuple[List[Dict], List[Dict], int, int]:
        """Charger les annotations de référence pour la page 5"""
        gt_file = 'dataset/labels/train/tintin_p0005.json'
        with open(gt_file, 'r') as f:
            data = json.load(f)

        panels_gt = []
        balloons_gt = []

        for shape in data['shapes']:
            if shape['label'] == 'panel':
                points = shape['points']
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                panel = {
                    'x': x_min, 'y': y_min,
                    'w': x_max - x_min, 'h': y_max - y_min,
                    'x_max': x_max, 'y_max': y_max,
                    'area': (x_max - x_min) * (y_max - y_min),
                    'center_x': (x_min + x_max) / 2,
                    'center_y': (y_min + y_max) / 2,
                    'aspect_ratio': (x_max - x_min) / (y_max - y_min) if y_max != y_min else 0
                }
                panels_gt.append(panel)

            elif shape['label'] == 'balloon':
                points = shape['points']
                x_min, y_min = points[0]
                x_max, y_max = points[1]

                balloon = {
                    'x': x_min, 'y': y_min,
                    'w': x_max - x_min, 'h': y_max - y_min,
                    'x_max': x_max, 'y_max': y_max,
                    'area': (x_max - x_min) * (y_max - y_min),
                    'center_x': (x_min + x_max) / 2,
                    'center_y': (y_min + y_max) / 2,
                    'aspect_ratio': (x_max - x_min) / (y_max - y_min) if y_max != y_min else 0
                }
                balloons_gt.append(balloon)

        return panels_gt, balloons_gt, data['imageWidth'], data['imageHeight']

    def find_tintin_pdf(self) -> str:
        """Trouver le PDF Tintin dans le projet"""
        possible_paths = [
            'data/examples/Tintin - 161 - Le Lotus Bleu -.pdf',
            'data/Tintin - 161 - Le Lotus Bleu -.pdf',
            'Tintin - 161 - Le Lotus Bleu -.pdf',
            'dataset/pdfs/Tintin - 161 - Le Lotus Bleu -.pdf'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Chercher récursivement
        for root, dirs, files in os.walk('.'):
            for file in files:
                if 'Tintin' in file and file.endswith('.pdf'):
                    return os.path.join(root, file)

        raise FileNotFoundError("PDF Tintin non trouvé")

    def calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculer l'IoU entre deux boîtes"""
        # Coordonnées des boîtes
        x1_min, y1_min = box1['x'], box1['y']
        x1_max, y1_max = box1['x_max'], box1['y_max']
        x2_min, y2_min = box2['x'], box2['y']
        x2_max, y2_max = box2['x_max'], box2['y_max']

        # Calcul de l'intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Calcul de l'union
        box1_area = box1['area']
        box2_area = box2['area']
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def calculate_center_distance(self, box1: Dict, box2: Dict) -> float:
        """Calculer la distance entre les centres de deux boîtes"""
        dx = box1['center_x'] - box2['center_x']
        dy = box1['center_y'] - box2['center_y']
        return math.sqrt(dx*dx + dy*dy)

    def find_best_matches(self, detected: List[Dict], ground_truth: List[Dict],
                         iou_threshold: float = 0.5) -> List[Tuple[int, int, float, float]]:
        """
        Trouver les meilleures correspondances entre détections et référence
        Retourne: [(idx_det, idx_gt, iou, center_dist), ...]
        """
        matches = []
        used_gt = set()

        for i, det in enumerate(detected):
            best_iou = 0.0
            best_gt_idx = -1
            best_dist = float('inf')

            for j, gt in enumerate(ground_truth):
                if j in used_gt:
                    continue

                iou = self.calculate_iou(det, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
                    best_dist = self.calculate_center_distance(det, gt)

            if best_iou >= iou_threshold:
                matches.append((i, best_gt_idx, best_iou, best_dist))
                used_gt.add(best_gt_idx)

        return matches

    def analyze_panel_centering(self, detected_panels: List[Dict],
                               ground_truth_panels: List[Dict]) -> Dict[str, Any]:
        """Analyser le centrage des panels détectés"""
        matches = self.find_best_matches(detected_panels, ground_truth_panels, 0.3)

        if not matches:
            return {
                'total_matches': 0,
                'avg_iou': 0.0,
                'avg_center_distance': 0.0,
                'max_center_distance': 0.0,
                'center_distances': [],
                'ious': []
            }

        ious = [match[2] for match in matches]
        center_distances = [match[3] for match in matches]

        # Calculer les statistiques de centrage
        avg_center_dist = sum(center_distances) / len(center_distances)
        max_center_dist = max(center_distances)

        # Calculer la précision de centrage (distance relative à la taille des panels)
        relative_distances = []
        for match in matches:
            det_idx, gt_idx, _, dist = match
            gt_panel = ground_truth_panels[gt_idx]
            # Distance relative à la diagonale du panel de référence
            diagonal = math.sqrt(gt_panel['w']**2 + gt_panel['h']**2)
            relative_dist = dist / diagonal if diagonal > 0 else 0
            relative_distances.append(relative_dist)

        return {
            'total_matches': len(matches),
            'avg_iou': sum(ious) / len(ious),
            'avg_center_distance': avg_center_dist,
            'max_center_distance': max_center_dist,
            'avg_relative_distance': sum(relative_distances) / len(relative_distances) if relative_distances else 0,
            'center_distances': center_distances,
            'relative_distances': relative_distances,
            'ious': ious,
            'matches': matches
        }

    def analyze_detection_quality(self) -> Dict[str, Any]:
        """Analyser la qualité globale de la détection"""
        # Analyse des panels
        panel_analysis = self.analyze_panel_centering(self.detected_panels, self.ground_truth_panels)

        # Analyse des ballons
        balloon_matches = self.find_best_matches(self.detected_balloons, self.ground_truth_balloons, 0.3)
        balloon_iou = [match[2] for match in balloon_matches] if balloon_matches else []
        balloon_center_dist = [match[3] for match in balloon_matches] if balloon_matches else []

        # Statistiques de taille
        gt_panel_areas = [p['area'] for p in self.ground_truth_panels]
        det_panel_areas = [p['area'] for p in self.detected_panels]
        gt_balloon_areas = [b['area'] for b in self.ground_truth_balloons]
        det_balloon_areas = [b['area'] for b in self.detected_balloons]

        return {
            'panels': {
                'detected': len(self.detected_panels),
                'ground_truth': len(self.ground_truth_panels),
                'precision': len(self.detected_panels) / len(self.ground_truth_panels),
                'recall': panel_analysis['total_matches'] / len(self.ground_truth_panels),
                'f1_score': 2 * (len(self.detected_panels) / len(self.ground_truth_panels)) * (panel_analysis['total_matches'] / len(self.ground_truth_panels)) /
                           (len(self.detected_panels) / len(self.ground_truth_panels) + panel_analysis['total_matches'] / len(self.ground_truth_panels)) if
                           (len(self.detected_panels) / len(self.ground_truth_panels) + panel_analysis['total_matches'] / len(self.ground_truth_panels)) > 0 else 0,
                'centering': panel_analysis,
                'size_stats': {
                    'gt_avg_area': sum(gt_panel_areas) / len(gt_panel_areas) if gt_panel_areas else 0,
                    'det_avg_area': sum(det_panel_areas) / len(det_panel_areas) if det_panel_areas else 0,
                    'area_ratio': (sum(det_panel_areas) / len(det_panel_areas) if det_panel_areas else 0) /
                                 (sum(gt_panel_areas) / len(gt_panel_areas) if gt_panel_areas else 0) if gt_panel_areas else 0
                }
            },
            'balloons': {
                'detected': len(self.detected_balloons),
                'ground_truth': len(self.ground_truth_balloons),
                'precision': len(self.detected_balloons) / len(self.ground_truth_balloons) if len(self.ground_truth_balloons) > 0 else 0,
                'recall': len(balloon_matches) / len(self.ground_truth_balloons) if len(self.ground_truth_balloons) > 0 else 0,
                'avg_iou': sum(balloon_iou) / len(balloon_iou) if balloon_iou else 0,
                'avg_center_distance': sum(balloon_center_dist) / len(balloon_center_dist) if balloon_center_dist else 0,
                'size_stats': {
                    'gt_avg_area': sum(gt_balloon_areas) / len(gt_balloon_areas) if gt_balloon_areas else 0,
                    'det_avg_area': sum(det_balloon_areas) / len(det_balloon_areas) if det_balloon_areas else 0,
                    'area_ratio': (sum(det_balloon_areas) / len(det_balloon_areas) if det_balloon_areas else 0) /
                                 (sum(gt_balloon_areas) / len(gt_balloon_areas) if gt_balloon_areas else 0) if gt_balloon_areas else 0
                }
            }
        }

    def run_automated_test(self) -> bool:
        """Exécuter le test automatisé complet avec analyse détaillée"""
        print("=" * 100)
        print("🔬 TEST DÉTAILLÉ PAGE 5 TINTIN - ANALYSE AVANCÉE")
        print("=" * 100)

        # Étape 1: Charger les données de référence
        print("\n📚 CHARGEMENT DES DONNÉES DE RÉFÉRENCE...")
        try:
            self.ground_truth_panels, self.ground_truth_balloons, img_width, img_height = self.load_ground_truth()
            print(f"✅ {len(self.ground_truth_panels)} panels et {len(self.ground_truth_balloons)} ballons chargés")
            print(f"   📐 Dimensions: {img_width}x{img_height} pixels")
        except Exception as e:
            print(f"❌ Erreur chargement référence: {e}")
            return False

        # Étape 2: Trouver le PDF
        print("\n🔍 RECHERCHE DU PDF TINTIN...")
        try:
            pdf_path = self.find_tintin_pdf()
        except FileNotFoundError:
            print("❌ PDF Tintin non trouvé")
            return False
        print(f"✅ PDF trouvé: {pdf_path}")

        # Étape 3: Initialiser l'application Qt
        print("\n🚀 INITIALISATION DE L'APPLICATION...")
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        # Étape 4: Créer la fenêtre viewer
        self.viewer = PdfYoloViewer()

        # Étape 5: Ouvrir le PDF
        print("
📖 OUVERTURE DU PDF...")
        try:
            pdf = fitz.open(pdf_path)
            self.viewer.pdf = pdf
            self.viewer.page_index = 4  # Page 5 (index 4)
            print(f"✅ PDF ouvert: {len(pdf)} pages, page actuelle: {self.viewer.page_index + 1}")
        except Exception as e:
            print(f"❌ Erreur ouverture PDF: {e}")
            return False

        # Étape 6: Charger la page 5
        print("
📄 CHARGEMENT DE LA PAGE 5...")
        try:
            self.viewer.load_page(4)  # Index 4 = page 5
            print("✅ Page 5 chargée")
        except Exception as e:
            print(f"❌ Erreur chargement page: {e}")
            return False

        # Étape 7: Attendre que la détection se termine
        print("
🔍 LANCEMENT DE LA DÉTECTION...")
        print("   (Attente de la fin du traitement...)")

        # Attendre un peu pour que la détection se termine
        time.sleep(3)

        # Étape 8: Extraire les résultats de détection
        print("
📊 EXTRACTION DES RÉSULTATS...")
        self.extract_detection_results()

        # Étape 9: Analyser les résultats en détail
        print("
🔬 ANALYSE DÉTAILLÉE...")
        analysis = self.analyze_detection_quality()

        # Étape 10: Afficher les résultats détaillés
        self.display_detailed_results(analysis)

        # Étape 11: Sauvegarder les résultats
        self.save_detailed_results(analysis)

        print("
✅ ANALYSE TERMINÉE")
        return True

    def extract_detection_results(self):
        """Extraire les résultats de détection du viewer"""
        if not hasattr(self.viewer, 'dets') or not self.viewer.dets:
            print("❌ Aucune détection trouvée")
            return

        # Séparer panels et ballons détectés
        detected_panels_raw = [d for d in self.viewer.dets if d.cls == 0]
        detected_balloons_raw = [d for d in self.viewer.dets if d.cls == 1]

        # Convertir en format dictionnaire pour l'analyse
        self.detected_panels = []
        for panel in detected_panels_raw:
            rect = panel.rect
            self.detected_panels.append({
                'x': rect.left(),
                'y': rect.top(),
                'w': rect.width(),
                'h': rect.height(),
                'x_max': rect.right(),
                'y_max': rect.bottom(),
                'area': rect.width() * rect.height(),
                'center_x': rect.center().x(),
                'center_y': rect.center().y(),
                'aspect_ratio': rect.width() / rect.height() if rect.height() > 0 else 0,
                'confidence': panel.conf
            })

        self.detected_balloons = []
        for balloon in detected_balloons_raw:
            rect = balloon.rect
            self.detected_balloons.append({
                'x': rect.left(),
                'y': rect.top(),
                'w': rect.width(),
                'h': rect.height(),
                'x_max': rect.right(),
                'y_max': rect.bottom(),
                'area': rect.width() * rect.height(),
                'center_x': rect.center().x(),
                'center_y': rect.center().y(),
                'aspect_ratio': rect.width() / rect.height() if rect.height() > 0 else 0,
                'confidence': balloon.conf
            })

    def display_detailed_results(self, analysis: Dict[str, Any]):
        """Afficher les résultats détaillés de l'analyse"""
        print("
" + "=" * 100)
        print("📊 RÉSULTATS DÉTAILLÉS DE L'ANALYSE")
        print("=" * 100)

        # Résultats généraux
        print("
🎯 RÉSULTATS GÉNÉRAUX:"        print(f"   📦 Panels détectés: {analysis['panels']['detected']}/{analysis['panels']['ground_truth']}")
        print(f"   💬 Ballons détectés: {analysis['balloons']['detected']}/{analysis['balloons']['ground_truth']}")

        # Analyse des panels
        print("
📦 ANALYSE DES PANELS:"        p = analysis['panels']
        print(f"   Précision: {p['precision']:.1%}")
        print(f"   Rappel: {p['recall']:.1%}")
        print(f"   F1-Score: {p['f1_score']:.3f}")

        centering = p['centering']
        if centering['total_matches'] > 0:
            print(f"   Correspondances: {centering['total_matches']}")
            print(f"   IoU moyen: {centering['avg_iou']:.3f}")
            print(f"   Distance centre moyenne: {centering['avg_center_distance']:.1f}px")
            print(f"   Distance centre relative: {centering['avg_relative_distance']:.3f}")
            print(f"   Distance centre max: {centering['max_center_distance']:.1f}px")

        size_stats = p['size_stats']
        print(f"   Taille moyenne GT: {size_stats['gt_avg_area']:.0f}px")
        print(f"   Taille moyenne détectée: {size_stats['det_avg_area']:.0f}px")
        print(f"   Ratio taille: {size_stats['area_ratio']:.2f}")

        # Analyse des ballons
        print("
💬 ANALYSE DES BALLONS:"        b = analysis['balloons']
        print(f"   Précision: {b['precision']:.1%}")
        print(f"   Rappel: {b['recall']:.1%}")
        print(f"   IoU moyen: {b['avg_iou']:.3f}")
        print(f"   Distance centre moyenne: {b['avg_center_distance']:.1f}px")

        size_stats_b = b['size_stats']
        print(f"   Taille moyenne GT: {size_stats_b['gt_avg_area']:.0f}px")
        print(f"   Taille moyenne détectée: {size_stats_b['det_avg_area']:.0f}px")
        print(f"   Ratio taille: {size_stats_b['area_ratio']:.2f}")

        # Statistiques détaillées de centrage
        if centering['total_matches'] > 0:
            print("
📍 STATISTIQUES DE CENTRAGE DÉTAILLÉES:"            distances = centering['center_distances']
            relative_distances = centering['relative_distances']
            ious = centering['ious']

            print(f"   Distances centres (px): min={min(distances):.1f}, max={max(distances):.1f}, std={math.sqrt(sum((d - centering['avg_center_distance'])**2 for d in distances) / len(distances)):.1f}")
            print(f"   Distances relatives: min={min(relative_distances):.3f}, max={max(relative_distances):.3f}, std={math.sqrt(sum((d - centering['avg_relative_distance'])**2 for d in relative_distances) / len(relative_distances)):.3f}")
            print(f"   IoU: min={min(ious):.3f}, max={max(ious):.3f}, std={math.sqrt(sum((iou - centering['avg_iou'])**2 for iou in ious) / len(ious)):.3f}")

        # Résumé final
        print("
🏆 RÉSUMÉ FINAL:"        overall_score = (p['f1_score'] + b['precision'] + b['recall']) / 3
        print(f"   Score global: {overall_score:.3f}/1.000")

        if overall_score >= 0.95:
            print("   ✅ EXCELLENT: Détection quasi-parfaite!")
        elif overall_score >= 0.85:
            print("   ✅ BON: Bonne détection avec quelques ajustements mineurs")
        elif overall_score >= 0.70:
            print("   ⚠️ MOYEN: Détection acceptable, ajustements recommandés")
        else:
            print("   ❌ À AMÉLIORER: Ajustements significatifs nécessaires")

    def save_detailed_results(self, analysis: Dict[str, Any]):
        """Sauvegarder les résultats détaillés"""
        results_file = f"detailed_analysis_page5_{int(time.time())}.json"

        # Préparer les données pour la sauvegarde
        save_data = {
            'timestamp': time.time(),
            'analysis': analysis,
            'ground_truth': {
                'panels': self.ground_truth_panels,
                'balloons': self.ground_truth_balloons
            },
            'detected': {
                'panels': self.detected_panels,
                'balloons': self.detected_balloons
            }
        }

        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"
💾 Résultats détaillés sauvegardés: {results_file}")

    def cleanup(self):
        """Nettoyer les ressources"""
        if self.viewer:
            self.viewer.close()
        if self.app:
            self.app.quit()

def main():
    """Fonction principale"""
    print("🚀 DÉMARRAGE DE L'ANALYSE DÉTAILLÉE...")

    # Activer le debug
    os.environ['DEBUG_DETECT'] = '1'

    # Créer et exécuter le test
    test = DetailedTintinTest()

    try:
        success = test.run_automated_test()
        if success:
            print("
🎉 ANALYSE RÉUSSIE !"        else:
            print("
❌ ANALYSE ÉCHOUÉE"
    except Exception as e:
        print(f"
💥 ERREUR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.cleanup()

if __name__ == "__main__":
    main()

import sys
import os
import json
import time
from pathlib import Path

# Ajouter le répertoire courant au path
sys.path.append('.')

# Importer les modules nécessaires
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
import fitz

# Importer notre application
from main import PdfYoloViewer

class AutomatedTintinTest:
    def __init__(self):
        self.app = None
        self.viewer = None
        self.test_results = {}

    def load_ground_truth(self):
        """Charger les annotations de référence pour la page 5"""
        gt_file = 'dataset/labels/train/tintin_p0005.json'
        with open(gt_file, 'r') as f:
            data = json.load(f)

        panels_gt = []
        balloons_gt = []

        for shape in data['shapes']:
            if shape['label'] == 'panel':
                points = shape['points']
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                panels_gt.append({
                    'x': x_min, 'y': y_min,
                    'w': x_max - x_min, 'h': y_max - y_min,
                    'x_max': x_max, 'y_max': y_max,
                    'area': (x_max - x_min) * (y_max - y_min)
                })
            elif shape['label'] == 'balloon':
                points = shape['points']
                x_min, y_min = points[0]
                x_max, y_max = points[1]
                balloons_gt.append({
                    'x': x_min, 'y': y_min,
                    'w': x_max - x_min, 'h': y_max - y_min,
                    'x_max': x_max, 'y_max': y_max,
                    'area': (x_max - x_min) * (y_max - y_min)
                })

        return panels_gt, balloons_gt, data['imageWidth'], data['imageHeight']

    def find_tintin_pdf(self):
        """Trouver le PDF Tintin dans le projet"""
        possible_paths = [
            'data/examples/Tintin - 161 - Le Lotus Bleu -.pdf',
            'data/Tintin - 161 - Le Lotus Bleu -.pdf',
            'Tintin - 161 - Le Lotus Bleu -.pdf',
            'dataset/pdfs/Tintin - 161 - Le Lotus Bleu -.pdf'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Chercher récursivement
        for root, dirs, files in os.walk('.'):
            for file in files:
                if 'Tintin' in file and file.endswith('.pdf'):
                    return os.path.join(root, file)

        return None

    def run_automated_test(self):
        """Exécuter le test automatisé complet"""
        print("=" * 80)
        print("🤖 TEST AUTOMATISÉ PAGE 5 TINTIN")
        print("=" * 80)

        # Étape 1: Charger les données de référence
        print("\n📚 CHARGEMENT DES DONNÉES DE RÉFÉRENCE...")
        try:
            panels_gt, balloons_gt, img_width, img_height = self.load_ground_truth()
            print(f"✅ {len(panels_gt)} panels et {len(balloons_gt)} ballons chargés")
            print(f"   📐 Dimensions: {img_width}x{img_height} pixels")
        except Exception as e:
            print(f"❌ Erreur chargement référence: {e}")
            return False

        # Étape 2: Trouver le PDF
        print("\n🔍 RECHERCHE DU PDF TINTIN...")
        pdf_path = self.find_tintin_pdf()
        if not pdf_path:
            print("❌ PDF Tintin non trouvé")
            return False
        print(f"✅ PDF trouvé: {pdf_path}")

        # Étape 3: Initialiser l'application Qt
        print("\n🚀 INITIALISATION DE L'APPLICATION...")
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        # Étape 4: Créer la fenêtre viewer
        self.viewer = PdfYoloViewer()

        # Étape 5: Ouvrir le PDF
        print("\n📖 OUVERTURE DU PDF...")
        try:
            pdf = fitz.open(pdf_path)
            self.viewer.pdf = pdf
            self.viewer.page_index = 4  # Page 5 (index 4)
            print(f"✅ PDF ouvert: {len(pdf)} pages, page actuelle: {self.viewer.page_index + 1}")
        except Exception as e:
            print(f"❌ Erreur ouverture PDF: {e}")
            return False

        # Étape 6: Charger la page 5
        print("\n📄 CHARGEMENT DE LA PAGE 5...")
        try:
            self.viewer.load_page(4)  # Index 4 = page 5
            print("✅ Page 5 chargée")
        except Exception as e:
            print(f"❌ Erreur chargement page: {e}")
            return False

        # Étape 7: Attendre que la détection se termine
        print("\n🔍 LANCEMENT DE LA DÉTECTION...")
        print("   (Attente de la fin du traitement...)")

        # Attendre un peu pour que la détection se termine
        time.sleep(3)

        # Étape 8: Analyser les résultats
        print("\n📊 ANALYSE DES RÉSULTATS...")
        self.analyze_results(panels_gt, balloons_gt)

        # Étape 9: Sauvegarder les résultats
        self.save_test_results()

        print("\n✅ TEST TERMINÉ")
        return True

    def analyze_results(self, panels_gt, balloons_gt):
        """Analyser les résultats de détection"""
        if not hasattr(self.viewer, 'dets') or not self.viewer.dets:
            print("❌ Aucune détection trouvée")
            return

        # Séparer panels et ballons détectés
        detected_panels = [d for d in self.viewer.dets if d.cls == 0]
        detected_balloons = [d for d in self.viewer.dets if d.cls == 1]

        print("\n🎯 RÉSULTATS DE DÉTECTION:")
        print(f"   📦 Panels détectés: {len(detected_panels)}")
        print(f"   💬 Ballons détectés: {len(detected_balloons)}")

        print("\n📈 COMPARAISON AVEC LA RÉFÉRENCE:")
        print(f"   📦 Panels référence: {len(panels_gt)}")
        print(f"   💬 Ballons référence: {len(balloons_gt)}")

        # Calculer les métriques de précision
        panel_precision = len(detected_panels) / len(panels_gt) if len(panels_gt) > 0 else 0
        balloon_precision = len(detected_balloons) / len(balloons_gt) if len(balloons_gt) > 0 else 0

        print("\n📊 PRÉCISION:")
        print(f"   📦 Panels: {panel_precision:.1%} ({len(detected_panels)}/{len(panels_gt)})")
        print(f"   💬 Ballons: {balloon_precision:.1%} ({len(detected_balloons)}/{len(balloons_gt)})")

        # Analyser les tailles
        if detected_panels:
            panel_areas = [d.rect.width() * d.rect.height() for d in detected_panels]
            print("\n📏 TAILLES DÉTECTÉES:")
            print(f"   📦 Panels: min={min(panel_areas):.0f}px, max={max(panel_areas):.0f}px, avg={sum(panel_areas)/len(panel_areas):.0f}px")

        if detected_balloons:
            balloon_areas = [d.rect.width() * d.rect.height() for d in detected_balloons]
            print(f"   💬 Ballons: min={min(balloon_areas):.0f}px, max={max(balloon_areas):.0f}px, avg={sum(balloon_areas)/len(balloon_areas):.0f}px")

        # Analyser les coordonnées (premiers éléments)
        if detected_panels:
            print("\n📍 PREMIERS PANELS DÉTECTÉS:")
            for i, panel in enumerate(detected_panels[:3]):
                rect = panel.rect
                print(f"   Panel {i+1}: x={rect.left():.0f}, y={rect.top():.0f}, w={rect.width():.0f}, h={rect.height():.0f}")

        if detected_balloons:
            print("\n💬 PREMIERS BALLONS DÉTECTÉS:")
            for i, balloon in enumerate(detected_balloons[:3]):
                rect = balloon.rect
                print(f"   Balloon {i+1}: x={rect.left():.0f}, y={rect.top():.0f}, w={rect.width():.0f}, h={rect.height():.0f}")

        # Stocker les résultats
        self.test_results = {
            'detected_panels': len(detected_panels),
            'detected_balloons': len(detected_balloons),
            'reference_panels': len(panels_gt),
            'reference_balloons': len(balloons_gt),
            'panel_precision': panel_precision,
            'balloon_precision': balloon_precision,
            'timestamp': time.time()
        }

    def save_test_results(self):
        """Sauvegarder les résultats du test"""
        results_file = f"test_results_page5_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\n💾 Résultats sauvegardés: {results_file}")

    def cleanup(self):
        """Nettoyer les ressources"""
        if self.viewer:
            self.viewer.close()
        if self.app:
            self.app.quit()

def main():
    """Fonction principale"""
    print("🚀 DÉMARRAGE DU TEST AUTOMATISÉ...")

    # Activer le debug
    os.environ['DEBUG_DETECT'] = '1'

    # Créer et exécuter le test
    test = AutomatedTintinTest()

    try:
        success = test.run_automated_test()
        if success:
            print("\n🎉 TEST RÉUSSI !")
        else:
            print("\n❌ TEST ÉCHOUÉ")
    except Exception as e:
        print(f"\n💥 ERREUR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.cleanup()

if __name__ == "__main__":
    main()
