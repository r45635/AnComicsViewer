#!/usr/bin/env python3
"""
Test Automatique Mu        # Configuration des formats de nom de fichier selon la BD
        self.gt_formats = {
            "tintin": "tintin_p{:04d}.json",
            "sisters": "sisters_p{:03d}.json",
            "pinup": "pinup_p{:04d}.json"  # Format correct pour pinup
        }

        format_str = self.gt_formats.get(self.comic_name, f'{self.comic_name}_p{{:04d}}.json')
        print(f"🔍 DEBUG FORMAT: format_str='{format_str}', page_number={page_number}")
        filename = format_str.format(page_number)        # Filtrer pour les ballons (ajusté pour Sisters - ballons plus grands)
        page_area = w * h
        min_area = int(page_area * 0.005)  # AUGMENTÉ de 0.0005 → 0.005 pour Sisters (211K px²)
        max_area = int(page_area * 0.15)   # AUGMENTÉ de 0.08 → 0.15 pour Sisters

        potential_balloons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w_contour, h_contour = cv2.boundingRect(contour)
                aspect_ratio = w_contour / h_contour if h_contour > 0 else 0

                # Critères ajustés pour Sisters (ballons plus grands)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                # Filtrer avec critères plus permissifs pour Sisters
                if 0.3 < aspect_ratio < 10.0 and circularity > 0.02:  # Plus permissif
                    potential_balloons.append({
                        'x': x,
                        'y': y,
                        'w': w_contour,
                        'h': h_contour,
                        'area': area,
                        'center_x': x + w_contour/2,
                        'center_y': y + h_contour/2,
                        'aspect_ratio': aspect_ratio,
                        'circularity': circularity,
                        'source': 'rules'
                    })(f"🔍 DEBUG FILENAME: filename='{filename}'")
        self.gt_file_pattern = f"dataset/labels/train/{filename}"
        print(f"🔍 DEBUG PATTERN: pattern='{self.gt_file_pattern}'")ti-BD - Analyse Détaillée avec KPIs
Script complet pour analyser la précision de détection avec métriques avancées
Supporte plusieurs bandes dessinées et pages
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
from PySide6.QtCore import QTimer, QRectF
from PySide6.QtGui import QImage
import fitz

# Importer notre application
from main import PdfYoloViewer, load_config, GLOBAL_CONFIG

class DetailedComicTest:
    def __init__(self, comic_name="tintin", page_number=5):
        print(f"🔍 DEBUG CONSTRUCTOR: comic_name={comic_name}, page_number={page_number}")
        self.comic_name = comic_name
        self.page_number = page_number
        self.app = None
        self.viewer = None
        self.test_results = {}
        self.ground_truth_panels = []
        self.ground_truth_balloons = []
        self.detected_panels = []
        self.detected_balloons = []

        # Configuration des chemins selon la BD
        self.pdf_paths = {
            "tintin": "./data/examples/Tintin - 161 - Le Lotus Bleu - .pdf",
            "sisters": "./dataset/pdfs/The Grémillet Sisters - 02 - Cassiopeia's Summer of Love (2020).pdf",
            "pinup": "./data/examples/La Pin-up du B24 - T01.pdf",
            "golden": "./data/examples/Golden City - T01 - Pilleurs d'épaves.pdf"
        }

        # Format spécifique selon la BD pour les fichiers ground truth
        if comic_name == "sisters":
            self.gt_file_pattern = f"dataset/labels/train/{comic_name}_p{page_number:03d}.json"
        elif comic_name in ["tintin", "pinup"]:
            self.gt_file_pattern = f"dataset/labels/train/{comic_name}_p{page_number:04d}.json"
        elif comic_name == "golden":
            self.gt_file_pattern = f"dataset/labels/val/p{page_number:04d}.json"
        else:
            # Format par défaut
            self.gt_file_pattern = f"dataset/labels/train/{comic_name}_p{page_number:04d}.json"
        
        print(f"🔍 DEBUG PATTERN: comic={comic_name}, page={page_number}, pattern='{self.gt_file_pattern}'")

    def load_ground_truth(self) -> Tuple[List[Dict], List[Dict], int, int]:
        """Charger les annotations de référence pour la page spécifiée"""
        gt_file = self.gt_file_pattern
        print(f"🔍 Tentative de chargement: {gt_file}")
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
                    'x': x_min,
                    'y': y_min,
                    'w': x_max - x_min,
                    'h': y_max - y_min,
                    'x_max': x_max,
                    'y_max': y_max,
                    'area': (x_max - x_min) * (y_max - y_min),
                    'center_x': (x_min + x_max) / 2,
                    'center_y': (y_min + y_max) / 2,
                    'aspect_ratio': (x_max - x_min) / (y_max - y_min) if y_max > y_min else 0
                })
            elif shape['label'] == 'balloon':
                points = shape['points']
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                balloons_gt.append({
                    'x': x_min,
                    'y': y_min,
                    'w': x_max - x_min,
                    'h': y_max - y_min,
                    'x_max': x_max,
                    'y_max': y_max,
                    'area': (x_max - x_min) * (y_max - y_min),
                    'center_x': (x_min + x_max) / 2,
                    'center_y': (y_min + y_max) / 2,
                    'aspect_ratio': (x_max - x_min) / (y_max - y_min) if y_max > y_min else 0
                })

        img_width = data['imageWidth']
        img_height = data['imageHeight']

        return panels_gt, balloons_gt, img_width, img_height

    def find_pdf(self) -> str:
        """Trouver le PDF de la bande dessinée spécifiée"""
        if self.comic_name in self.pdf_paths:
            pdf_path = self.pdf_paths[self.comic_name]
            if os.path.exists(pdf_path):
                return pdf_path

        # Recherche alternative
        search_patterns = [
            f"*{self.comic_name}*.pdf",
            f"dataset/pdfs/*{self.comic_name}*.pdf",
            f"data/examples/*{self.comic_name}*.pdf"
        ]

        for pattern in search_patterns:
            import glob
            matches = glob.glob(pattern)
            if matches:
                return matches[0]

        raise FileNotFoundError(f"PDF non trouvé pour {self.comic_name}")

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
            # Stocker les dimensions de l'image pour la conversion de coordonnées
            self.image_width = img_width
            self.image_height = img_height
            print(f"✅ {len(self.ground_truth_panels)} panels et {len(self.ground_truth_balloons)} ballons chargés")
            print(f"   📐 Dimensions: {img_width}x{img_height} pixels")
        except Exception as e:
            print(f"❌ Erreur chargement référence: {e}")
            return False

        # Étape 2: Trouver le PDF
        print(f"\n🔍 RECHERCHE DU PDF {self.comic_name.upper()}...")
        try:
            pdf_path = self.find_pdf()
        except FileNotFoundError:
            print(f"❌ PDF {self.comic_name} non trouvé")
            return False
        print(f"✅ PDF trouvé: {pdf_path}")

        # Étape 3: Initialiser l'application Qt
        print("\n🚀 INITIALISATION DE L'APPLICATION...")
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        # Étape 4: Créer la fenêtre viewer
        self.viewer = PdfYoloViewer()

        # Forcer le rechargement de la configuration
        print("   🔄 Rechargement de la configuration...")
        config_path = 'config/detect.yaml'
        if os.path.exists(config_path):
            config = load_config(config_path)
            # Mettre à jour la configuration globale
            GLOBAL_CONFIG.clear()
            GLOBAL_CONFIG.update(config)
            print(f"   ✅ Configuration rechargée: panel_conf={config.get('panel_conf', 'N/A')}")
        else:
            print(f"   ⚠️ Fichier de configuration non trouvé: {config_path}")

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

        # Étape 8: Extraire les résultats de détection
        print("\n📊 EXTRACTION DES RÉSULTATS...")
        self.extract_detection_results()

        # Étape 9: Analyser les résultats en détail
        print("\n🔬 ANALYSE DÉTAILLÉE...")
        analysis = self.analyze_detection_quality()

        # Étape 10: Afficher les résultats détaillés
        self.display_detailed_results(analysis)

        # Étape 11: Sauvegarder les résultats
        self.save_detailed_results(analysis)

        # Étape 12: Tester l'approche alternative par règles
        print("\n🔧 ÉTAPE 12: APPROCHE ALTERNATIVE PAR RÈGLES")
        try:
            rule_based_panels = self.run_rule_based_detection()
            print(f"✅ Détection par règles terminée: {len(rule_based_panels) if rule_based_panels else 0} panels")
        except Exception as e:
            print(f"⚠️ Erreur dans la détection par règles: {e}")

        # Étape 13: Tester l'approche hybride YOLO + Règles
        print("\n🔄 ÉTAPE 13: APPROCHE HYBRIDE YOLO + RÈGLES")
        try:
            hybrid_panels, hybrid_balloons = self.run_hybrid_detection()
            print(f"✅ Détection hybride terminée: {len(hybrid_panels)} panels, {len(hybrid_balloons)} ballons")
        except Exception as e:
            print(f"⚠️ Erreur dans la détection hybride: {e}")

        print("\n✅ ANALYSE TERMINÉE")
        return True
    def run_rule_based_detection(self):
        """Méthode alternative : détection basée sur des règles simples pour bandes dessinées"""
        print("\n🔧 APPROCHE ALTERNATIVE : DÉTECTION BASÉE SUR DES RÈGLES")
        print("=" * 60)

        # Charger l'image de la page
        from PIL import Image
        import numpy as np
        import cv2

        # Obtenir l'image depuis le viewer
        if self.viewer is None:
            print("❌ Viewer non initialisé")
            return

        if hasattr(self.viewer, 'qimage_current') and self.viewer.qimage_current:
            qimg = self.viewer.qimage_current
            # S'assurer que l'image est au bon format
            if qimg.format() != QImage.Format.Format_RGBA8888:
                qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)

            w, h = qimg.width(), qimg.height()
            ptr = qimg.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            print("❌ Impossible d'obtenir l'image de la page")
            return

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Détection des contours pour identifier les panels
        edges = cv2.Canny(gray, 50, 150)

        # Dilatation pour connecter les contours
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Trouver les contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrer les contours par taille
        page_area = w * h
        min_area = int(page_area * 0.02)  # 2% de la page
        max_area = int(page_area * 0.8)   # 80% de la page

        potential_panels = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w_contour, h_contour = cv2.boundingRect(contour)
                aspect_ratio = w_contour / h_contour if h_contour > 0 else 0

                # Filtrer par ratio d'aspect (éviter les éléments trop allongés)
                if 0.3 < aspect_ratio < 3.0:
                    potential_panels.append({
                        'x': x,
                        'y': y,
                        'w': w_contour,
                        'h': h_contour,
                        'area': area,
                        'center_x': x + w_contour/2,
                        'center_y': y + h_contour/2,
                        'aspect_ratio': aspect_ratio
                    })

        print(f"📦 Panels potentiels détectés par règles: {len(potential_panels)}")

        # Analyser les correspondances avec le ground truth
        if potential_panels:
            self.analyze_rule_based_results(potential_panels)

        return potential_panels

    def run_rule_based_balloon_detection(self):
        """Détection de ballons basée sur des règles simples"""
        print("🔍 DÉTECTION BALLONS PAR RÈGLES...")

        # Charger l'image de la page
        from PIL import Image
        import numpy as np
        import cv2

        # Obtenir l'image depuis le viewer
        if self.viewer is None:
            print("❌ Viewer non initialisé")
            return []

        if hasattr(self.viewer, 'qimage_current') and self.viewer.qimage_current:
            qimg = self.viewer.qimage_current
            if qimg.format() != QImage.Format.Format_RGBA8888:
                qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)

            w = qimg.width()
            h = qimg.height()
            ptr = qimg.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
            img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            print("❌ Impossible d'obtenir l'image de la page")
            return []

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Améliorer le contraste pour mieux détecter les ballons
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Détection des contours avec paramètres adaptés aux ballons
        edges = cv2.Canny(enhanced, 30, 100)  # Paramètres plus sensibles

        # Dilatation plus légère pour les ballons
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Trouver les contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"🔍 Contours trouvés: {len(contours)} (avant filtrage ballons)")

        # Filtrer pour les ballons (plus petits que les panels)
        page_area = w * h
        min_area = int(page_area * 0.0005)  # 0.05% de la page (encore plus petit)
        max_area = int(page_area * 0.08)    # 8% de la page (plus permissif)

        potential_balloons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w_contour, h_contour = cv2.boundingRect(contour)
                aspect_ratio = w_contour / h_contour if h_contour > 0 else 0

                # Critères spécifiques aux ballons - plus permissifs
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                # Filtrer par circularité (ballons souvent plus circulaires) - plus permissif
                if 0.05 < aspect_ratio < 8.0 and circularity > 0.05:
                    potential_balloons.append({
                        'x': x,
                        'y': y,
                        'w': w_contour,
                        'h': h_contour,
                        'area': area,
                        'center_x': x + w_contour/2,
                        'center_y': y + h_contour/2,
                        'aspect_ratio': aspect_ratio,
                        'circularity': circularity,
                        'source': 'rules'
                    })

        print(f"💬 Ballons potentiels détectés par règles: {len(potential_balloons)}")
        return potential_balloons

    def analyze_rule_based_results(self, detected_panels):
        """Analyser les résultats de la détection basée sur les règles"""
        print(f"\n📊 ANALYSE DES RÉSULTATS PAR RÈGLES:")
        print(f"   📦 Panels détectés: {len(detected_panels)}")
        print(f"   📦 Panels ground truth: {len(self.ground_truth_panels)}")

        # Calculer les métriques de base
        matches = 0
        total_iou = 0
        total_center_dist = 0

        for i, gt_panel in enumerate(self.ground_truth_panels):
            best_iou = 0
            best_dist = float('inf')
            best_match = None

            for j, det_panel in enumerate(detected_panels):
                # Calculer IoU
                iou = self.calculate_iou_rule_based(gt_panel, det_panel)
                if iou > best_iou:
                    best_iou = iou
                    center_dist = math.sqrt(
                        (gt_panel['center_x'] - det_panel['center_x'])**2 +
                        (gt_panel['center_y'] - det_panel['center_y'])**2
                    )
                    best_dist = center_dist
                    best_match = j

            if best_iou > 0.1:  # Seuil minimal pour considérer une correspondance
                matches += 1
                total_iou += best_iou
                total_center_dist += best_dist
                print(f"   ✅ Panel {i}: IoU={best_iou:.3f}, Dist={best_dist:.1f}px")

        if matches > 0:
            avg_iou = total_iou / matches
            avg_dist = total_center_dist / matches
            print(f"\n🏆 RÉSULTATS RÈGLES:")
            print(f"   📊 Correspondances: {matches}/{len(self.ground_truth_panels)}")
            print(f"   🎯 IoU moyen: {avg_iou:.3f}")
            print(f"   📍 Distance centre moyenne: {avg_dist:.1f}px")
        else:
            print("   ❌ Aucune correspondance trouvée")

    def calculate_iou_rule_based(self, panel1, panel2):
        """Calculer l'IoU entre deux panels pour la détection par règles"""
        # Convertir en rectangles
        r1 = QRectF(panel1['x'], panel1['y'], panel1['w'], panel1['h'])
        r2 = QRectF(panel2['x'], panel2['y'], panel2['w'], panel2['h'])

        inter = r1.intersected(r2)
        if inter.isEmpty():
            return 0.0

        inter_area = inter.width() * inter.height()
        union_area = r1.width()*r1.height() + r2.width()*r2.height() - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def run_hybrid_detection(self):
        """Approche hybride : combine YOLO + détection par règles"""
        print("\n🔄 APPROCHE HYBRIDE : YOLO + RÈGLES")
        print("=" * 50)

        # Étape 1: Obtenir les résultats YOLO
        yolo_panels = self.detected_panels.copy() if self.detected_panels else []
        yolo_balloons = self.detected_balloons.copy() if self.detected_balloons else []

        print(f"📊 YOLO - Panels: {len(yolo_panels)}, Ballons: {len(yolo_balloons)}")

        # Étape 2: Obtenir les résultats par règles
        rule_panels = []
        try:
            rule_panels = self.run_rule_based_detection()
            if rule_panels is None:
                rule_panels = []
        except Exception as e:
            print(f"⚠️ Erreur dans la détection par règles: {e}")
            rule_panels = []

        # Étape 3: Fusionner intelligemment les résultats
        hybrid_panels = self.merge_yolo_and_rules(yolo_panels, rule_panels, "panel")
        hybrid_balloons = self.merge_yolo_and_rules(yolo_balloons, [], "balloon")  # Pour l'instant, pas de règles pour ballons

        print(f"\n🎯 RÉSULTATS HYBRIDES:")
        print(f"   📦 Panels: {len(hybrid_panels)} (YOLO: {len(yolo_panels)}, Règles: {len(rule_panels)})")
        print(f"   💬 Ballons: {len(hybrid_balloons)} (YOLO: {len(yolo_balloons)}, Règles: 0)")

        # Étape 4: Essayer de détecter les ballons manqués par règles
        # Vérifier le rappel des ballons YOLO
        balloon_matches = 0
        for gt_balloon in self.ground_truth_balloons:
            for det_balloon in hybrid_balloons:
                if self.calculate_iou_hybrid(gt_balloon, det_balloon) > 0.1:
                    balloon_matches += 1
                    break

        balloon_recall = balloon_matches / len(self.ground_truth_balloons) if self.ground_truth_balloons else 1.0

        if balloon_recall < 0.5:  # Si rappel < 50%, essayer les règles
            print(f"\n🔍 TENTATIVE DE DÉTECTION BALLONS PAR RÈGLES (rappel YOLO: {balloon_recall:.1%})...")
            rule_balloons = self.run_rule_based_balloon_detection()
            if rule_balloons:
                # Fusionner les ballons YOLO avec ceux détectés par règles
                hybrid_balloons = self.merge_yolo_and_rules(hybrid_balloons, rule_balloons, "balloon")
                print(f"   💬 Ballons après règles: {len(hybrid_balloons)}")
        else:
            print(f"   💬 Rappel YOLO acceptable ({balloon_recall:.1%}), pas de détection par règles")

        # Étape 5: Analyser la qualité des résultats hybrides
        self.analyze_hybrid_results(hybrid_panels, hybrid_balloons)

        return hybrid_panels, hybrid_balloons

    def merge_yolo_and_rules(self, yolo_results, rule_results, element_type):
        """Fusionner les résultats YOLO et par règles de manière intelligente"""
        merged_results = []

        # Commencer par ajouter tous les résultats YOLO
        merged_results.extend(yolo_results)

        # Pour les panels, ajouter les résultats par règles qui ne se chevauchent pas trop
        if element_type == "panel" and rule_results:
            for rule_panel in rule_results:
                # Vérifier si ce panel par règles se chevauche avec un panel YOLO
                overlap_found = False
                for yolo_panel in yolo_results:
                    iou = self.calculate_iou_hybrid(yolo_panel, rule_panel)
                    if iou > 0.5:  # AUGMENTÉ de 0.3 → 0.5 pour être plus strict sur les doublons
                        overlap_found = True
                        break

                # Si pas de chevauchement significatif, ajouter le panel par règles
                if not overlap_found:
                    # Marquer comme venant des règles
                    rule_panel_copy = rule_panel.copy()
                    rule_panel_copy['source'] = 'rules'
                    merged_results.append(rule_panel_copy)

        return merged_results

    def calculate_iou_hybrid(self, panel1, panel2):
        """Calculer l'IoU pour la fusion hybride"""
        # S'assurer que les panels ont le bon format
        if 'x' not in panel1 or 'x' not in panel2:
            return 0.0

        # Calculer les coordonnées des rectangles
        x1_min = min(panel1['x'], panel1.get('x_max', panel1['x'] + panel1['w']))
        y1_min = min(panel1['y'], panel1.get('y_max', panel1['y'] + panel1['h']))
        x1_max = max(panel1['x'], panel1.get('x_max', panel1['x'] + panel1['w']))
        y1_max = max(panel1['y'], panel1.get('y_max', panel1['y'] + panel1['h']))

        x2_min = min(panel2['x'], panel2.get('x_max', panel2['x'] + panel2['w']))
        y2_min = min(panel2['y'], panel2.get('y_max', panel2['y'] + panel2['h']))
        x2_max = max(panel2['x'], panel2.get('x_max', panel2['x'] + panel2['w']))
        y2_max = max(panel2['y'], panel2.get('y_max', panel2['y'] + panel2['h']))

        # Calculer l'intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Calculer l'union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def analyze_hybrid_results(self, hybrid_panels, hybrid_balloons):
        """Analyser les résultats de l'approche hybride"""
        print(f"\n📊 ANALYSE HYBRIDE:")

        # Analyser les panels
        if hybrid_panels:
            panel_matches = 0
            for i, gt_panel in enumerate(self.ground_truth_panels):
                best_iou = 0
                for det_panel in hybrid_panels:
                    iou = self.calculate_iou_hybrid(gt_panel, det_panel)
                    if iou > best_iou:
                        best_iou = iou

                if best_iou > 0.1:
                    panel_matches += 1
                    print(f"   ✅ Panel {i}: IoU={best_iou:.3f}")

            panel_recall = panel_matches / len(self.ground_truth_panels) if self.ground_truth_panels else 0
            print(f"   📦 Panels: {panel_matches}/{len(self.ground_truth_panels)} ({panel_recall:.1%} rappel)")

        # Analyser les ballons
        if hybrid_balloons:
            balloon_matches = 0
            for i, gt_balloon in enumerate(self.ground_truth_balloons):
                best_iou = 0
                for det_balloon in hybrid_balloons:
                    iou = self.calculate_iou_hybrid(gt_balloon, det_balloon)
                    if iou > best_iou:
                        best_iou = iou

                if best_iou > 0.1:
                    balloon_matches += 1
                    print(f"   ✅ Ballon {i}: IoU={best_iou:.3f}")

            balloon_recall = balloon_matches / len(self.ground_truth_balloons) if self.ground_truth_balloons else 0
            print(f"   💬 Ballons: {balloon_matches}/{len(self.ground_truth_balloons)} ({balloon_recall:.1%} rappel)")

        print(f"\n🏆 RÉSULTATS HYBRIDES:")
        print(f"   📊 Score global: {len(hybrid_panels) + len(hybrid_balloons)} éléments détectés")
        print(f"   🎯 Amélioration potentielle par rapport à YOLO seul")

    def extract_detection_results(self):
        """Extraire les résultats de détection du viewer"""
        if self.viewer is None or not hasattr(self.viewer, 'dets') or self.viewer.dets is None or not self.viewer.dets:
            print("❌ Aucune détection trouvée")
            return

        # Séparer panels et ballons détectés
        detected_panels_raw = [d for d in self.viewer.dets if d.cls == 0]
        detected_balloons_raw = [d for d in self.viewer.dets if d.cls == 1]

        # Convertir en format dictionnaire pour l'analyse
        self.detected_panels = []
        for panel in detected_panels_raw:
            rect = panel.rect
            
            # Conversion des coordonnées PDF vers pixels d'image
            if hasattr(self, 'image_width') and hasattr(self, 'image_height') and self.image_width and self.image_height:
                pdf_to_pixel_x = self.image_width / 612.0
                pdf_to_pixel_y = self.image_height / 792.0
                
                x_pixel = rect.left() / pdf_to_pixel_x
                y_pixel = rect.top() / pdf_to_pixel_y
                w_pixel = rect.width() / pdf_to_pixel_x
                h_pixel = rect.height() / pdf_to_pixel_y
            else:
                x_pixel = rect.left()
                y_pixel = rect.top()
                w_pixel = rect.width()
                h_pixel = rect.height()
            
            self.detected_panels.append({
                'x': x_pixel,
                'y': y_pixel,
                'w': w_pixel,
                'h': h_pixel,
                'x_max': x_pixel + w_pixel,
                'y_max': y_pixel + h_pixel,
                'area': w_pixel * h_pixel,
                'center_x': x_pixel + w_pixel/2,
                'center_y': y_pixel + h_pixel/2,
                'aspect_ratio': w_pixel / h_pixel if h_pixel > 0 else 0,
                'confidence': panel.conf
            })

        self.detected_balloons = []
        for balloon in detected_balloons_raw:
            rect = balloon.rect
            
            # Conversion des coordonnées PDF vers pixels d'image
            # Les rect sont en points PDF, on les convertit en pixels
            if hasattr(self, 'image_width') and hasattr(self, 'image_height') and self.image_width and self.image_height:
                # Facteurs de conversion inverse (PDF -> pixels)
                pdf_to_pixel_x = self.image_width / 612.0  # A4 width = 612 pts
                pdf_to_pixel_y = self.image_height / 792.0  # A4 height = 792 pts
                
                x_pixel = rect.left() / pdf_to_pixel_x
                y_pixel = rect.top() / pdf_to_pixel_y
                w_pixel = rect.width() / pdf_to_pixel_x
                h_pixel = rect.height() / pdf_to_pixel_y
            else:
                # Fallback: utiliser les coordonnées telles quelles
                x_pixel = rect.left()
                y_pixel = rect.top()
                w_pixel = rect.width()
                h_pixel = rect.height()
            
            self.detected_balloons.append({
                'x': x_pixel,
                'y': y_pixel,
                'w': w_pixel,
                'h': h_pixel,
                'x_max': x_pixel + w_pixel,
                'y_max': y_pixel + h_pixel,
                'area': w_pixel * h_pixel,
                'center_x': x_pixel + w_pixel/2,
                'center_y': y_pixel + h_pixel/2,
                'aspect_ratio': w_pixel / h_pixel if h_pixel > 0 else 0,
                'confidence': balloon.conf
            })

    def display_detailed_results(self, analysis: Dict[str, Any]):
        """Afficher les résultats détaillés de l'analyse"""
        print("\n" + "=" * 100)
        print("📊 RÉSULTATS DÉTAILLÉS DE L'ANALYSE")
        print("=" * 100)

        # Résultats généraux
        print("\n🎯 RÉSULTATS GÉNÉRAUX:")
        print(f"   📦 Panels détectés: {analysis['panels']['detected']}/{analysis['panels']['ground_truth']}")
        print(f"   💬 Ballons détectés: {analysis['balloons']['detected']}/{analysis['balloons']['ground_truth']}")

        # Analyse des panels
        print("\n📦 ANALYSE DES PANELS:")
        p = analysis['panels']
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
        print("\n💬 ANALYSE DES BALLONS:")
        b = analysis['balloons']
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
            print("\n📍 STATISTIQUES DE CENTRAGE DÉTAILLÉES:")
            distances = centering['center_distances']
            relative_distances = centering['relative_distances']
            ious = centering['ious']

            print(f"   Distances centres (px): min={min(distances):.1f}, max={max(distances):.1f}, std={math.sqrt(sum((d - centering['avg_center_distance'])**2 for d in distances) / len(distances)):.1f}")
            print(f"   Distances relatives: min={min(relative_distances):.3f}, max={max(relative_distances):.3f}, std={math.sqrt(sum((d - centering['avg_relative_distance'])**2 for d in relative_distances) / len(relative_distances)):.3f}")
            print(f"   IoU: min={min(ious):.3f}, max={max(ious):.3f}, std={math.sqrt(sum((iou - centering['avg_iou'])**2 for iou in ious) / len(ious)):.3f}")

        # Résumé final
        print("\n🏆 RÉSUMÉ FINAL:")
        overall_score = (p['f1_score'] + b['precision'] + b['recall']) / 3
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
        print(f"\n💾 Résultats détaillés sauvegardés: {results_file}")

    def cleanup(self):
        """Nettoyer les ressources"""
        if self.viewer:
            self.viewer.close()
        if self.app:
            self.app.quit()

def main():
    """Fonction principale"""
    print("🚀 DÉMARRAGE DE L'ANALYSE DÉTAILLÉE...")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test de détection de panels et ballons')
    parser.add_argument('--comic', default='tintin', choices=['tintin', 'sisters', 'pinup', 'golden'],
                       help='Nom de la bande dessinée (tintin, sisters, pinup, golden)')
    parser.add_argument('--page', type=int, default=5,
                       help='Numéro de la page à tester')

    args = parser.parse_args()

    print(f"🚀 DÉMARRAGE DU TEST POUR {args.comic.upper()} PAGE {args.page}")
    print("=" * 80)
    print(f"🔍 DEBUG ARGS: comic={args.comic}, page={args.page}")

    test = DetailedComicTest(args.comic, args.page)

    try:
        success = test.run_automated_test()
        if success:
            print("\n🎉 ANALYSE RÉUSSIE !")
        else:
            print("\n❌ ANALYSE ÉCHOUÉE")
    except Exception as e:
        print(f"\n💥 ERREUR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.cleanup()
