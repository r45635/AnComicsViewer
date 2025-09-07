#!/usr/bin/env python3
"""
Comparaison précise annotations JSON vs détections actuelles
===========================================================
Compare les annotations ground truth (JSON LabelMe) avec les détections du programme
"""

import sys
import os
import json
import fitz
from pathlib import Path
from PySide6.QtGui import QImage
from PySide6.QtCore import QRectF

# Ajouter le chemin du projet
sys.path.insert(0, '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch')

def load_json_annotations(json_path):
    """Charge les annotations JSON LabelMe"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    panels = []
    balloons = []

    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle' and len(shape['points']) == 2:
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)

            if shape['label'] == 'panel':
                panels.append(rect)
            elif shape['label'] == 'balloon':
                balloons.append(rect)

    return panels, balloons, data['imageWidth'], data['imageHeight']

def extract_page_from_pdf(pdf_path, page_num):
    """Extrait une page du PDF et la convertit en QImage"""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # Les pages commencent à 0

    # Rendu haute résolution pour la détection
    zoom = 2.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    # Convertir en QImage
    img_data = pix.tobytes("png")
    qimg = QImage()
    qimg.loadFromData(img_data)

    doc.close()
    return qimg

def run_detection_on_image(qimg):
    """Exécute la détection sur une QImage en utilisant le code de main.py"""
    # Utiliser une approche plus simple : créer un script Python séparé
    import tempfile
    import os
    import subprocess

    # Sauvegarder temporairement l'image
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, "temp_page.png")
    qimg.save(img_path)

    # Créer un script de détection simple
    detect_script = '''
import sys
sys.path.insert(0, "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QRectF
import json

# Importer et exécuter le code de main.py
exec(open("/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/main.py").read(), globals())

# Créer l'application et le viewer
app = QApplication(sys.argv)
viewer = PdfYoloViewer()

# Charger l'image
qimg = QImage("''' + img_path + '''")
viewer._set_page_image(qimg)
viewer._run_detection()

# Collecter les résultats
panels = []
balloons = []

for det in viewer.dets:
    rect = det.rect
    item = {
        "x": float(rect.x()),
        "y": float(rect.y()),
        "width": float(rect.width()),
        "height": float(rect.height()),
        "conf": float(det.conf)
    }
    if det.cls == 0:
        panels.append(item)
    elif det.cls == 1:
        balloons.append(item)

result = {"panels": panels, "balloons": balloons}
print(json.dumps(result))
'''

    script_path = os.path.join(temp_dir, "detect.py")
    with open(script_path, 'w') as f:
        f.write(detect_script)

    # Exécuter le script
    try:
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0 and result.stdout.strip():
            detections = json.loads(result.stdout.strip())

            detected_panels = []
            detected_balloons = []

            for p in detections["panels"]:
                detected_panels.append(QRectF(p["x"], p["y"], p["width"], p["height"]))

            for b in detections["balloons"]:
                detected_balloons.append(QRectF(b["x"], b["y"], b["width"], b["height"]))

            return detected_panels, detected_balloons
        else:
            print(f"Erreur script: returncode={result.returncode}")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            return [], []

    except subprocess.TimeoutExpired:
        print("Timeout de la détection")
        return [], []
    except Exception as e:
        print(f"Erreur subprocess: {e}")
        return [], []
    finally:
        # Nettoyer
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def calculate_iou(rect1, rect2):
    """Calcule l'IoU entre deux rectangles"""
    inter = rect1.intersected(rect2)
    if inter.isEmpty():
        return 0.0

    inter_area = inter.width() * inter.height()
    union_area = rect1.width() * rect1.height() + rect2.width() * rect2.height() - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def evaluate_detection(ground_truth, detections, iou_threshold=0.5):
    """Évalue la qualité des détections par rapport au ground truth"""
    if not ground_truth:
        return {
            'precision': 1.0 if not detections else 0.0,
            'recall': 1.0,
            'f1_score': 1.0 if not detections else 0.0,
            'matched_gt': 0,
            'false_positives': len(detections),
            'missed_gt': 0
        }

    matched_gt = set()
    matched_det = set()

    # Pour chaque détection, trouver le meilleur match ground truth
    for i, det in enumerate(detections):
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt in enumerate(ground_truth):
            if j in matched_gt:
                continue

            iou = calculate_iou(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            matched_det.add(i)

    # Calculer les métriques
    true_positives = len(matched_det)
    false_positives = len(detections) - true_positives
    false_negatives = len(ground_truth) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'matched_gt': len(matched_gt),
        'false_positives': false_positives,
        'missed_gt': false_negatives
    }

def compare_page(json_file, pdf_path, page_num):
    """Compare une page spécifique"""
    print(f"\n🔍 ANALYSE DE {json_file}")
    print("-" * 50)

    # Charger les annotations ground truth
    json_path = f"/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/backup_annotations_20250822_182146/{json_file}"
    gt_panels, gt_balloons, img_w, img_h = load_json_annotations(json_path)

    print(f"📝 GROUND TRUTH:")
    print(f"   Panels: {len(gt_panels)}")
    print(f"   Balloons: {len(gt_balloons)}")
    print(f"   Dimensions: {img_w}x{img_h}")

    # Extraire la page du PDF
    try:
        qimg = extract_page_from_pdf(pdf_path, page_num)
        print(f"📄 PAGE EXTRAITE: {qimg.width()}x{qimg.height()}")
    except Exception as e:
        print(f"❌ ERREUR extraction PDF: {e}")
        return None

    # Exécuter la détection
    try:
        det_panels, det_balloons = run_detection_on_image(qimg)
        print(f"🤖 DÉTECTIONS:")
        print(f"   Panels: {len(det_panels)}")
        print(f"   Balloons: {len(det_balloons)}")
    except Exception as e:
        print(f"❌ ERREUR détection: {e}")
        return None

    # Évaluer les panels
    panel_metrics = evaluate_detection(gt_panels, det_panels)
    balloon_metrics = evaluate_detection(gt_balloons, det_balloons)

    print(f"\n📊 RÉSULTATS PANELS:")
    print(f"   Précision: {panel_metrics['precision']:.3f}")
    print(f"   Rappel: {panel_metrics['recall']:.3f}")
    print(f"   F1-Score: {panel_metrics['f1_score']:.3f}")
    print(f"   Correspondances: {panel_metrics['matched_gt']}/{len(gt_panels)}")
    print(f"   Faux positifs: {panel_metrics['false_positives']}")
    print(f"   Manqués: {panel_metrics['missed_gt']}")

    print(f"\n💬 RÉSULTATS BALLOONS:")
    print(f"   Précision: {balloon_metrics['precision']:.3f}")
    print(f"   Rappel: {balloon_metrics['recall']:.3f}")
    print(f"   F1-Score: {balloon_metrics['f1_score']:.3f}")
    print(f"   Correspondances: {balloon_metrics['matched_gt']}/{len(gt_balloons)}")
    print(f"   Faux positifs: {balloon_metrics['false_positives']}")
    print(f"   Manqués: {balloon_metrics['missed_gt']}")

    return {
        'panels': panel_metrics,
        'balloons': balloon_metrics,
        'gt_panels': len(gt_panels),
        'det_panels': len(det_panels),
        'gt_balloons': len(gt_balloons),
        'det_balloons': len(det_balloons)
    }

def main():
    """Fonction principale"""
    print("🔬 COMPARAISON PRÉCISE: ANNOTATIONS vs DÉTECTIONS")
    print("=" * 70)

    # Configuration des tests - correspondances correctes JSON -> PDF
    test_cases = [
        # Format: (json_file, pdf_path, page_number, description)
        ("tintin_p0001.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 1, "Tintin - Page 1"),
        ("tintin_p0002.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 2, "Tintin - Page 2"),
        ("tintin_p0003.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 3, "Tintin - Page 3"),
        ("tintin_p0005.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 5, "Tintin - Page 5"),
        ("tintin_p0006.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 6, "Tintin - Page 6"),
        ("tintin_p0007.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 7, "Tintin - Page 7"),
        ("tintin_p0008.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 8, "Tintin - Page 8"),
        ("tintin_p0009.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 9, "Tintin - Page 9"),
        ("tintin_p0012.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 12, "Tintin - Page 12"),
        ("tintin_p0013.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 13, "Tintin - Page 13"),
        ("tintin_p0015.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 15, "Tintin - Page 15"),
        ("tintin_p0016.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 16, "Tintin - Page 16"),
        ("tintin_p0018.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 18, "Tintin - Page 18"),
        ("tintin_p0019.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 19, "Tintin - Page 19"),
        ("tintin_p0020.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 20, "Tintin - Page 20"),
        ("tintin_p0021.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf", 21, "Tintin - Page 21"),
        ("pinup_p0001.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 1, "Pin-up - Page 1"),
        ("pinup_p0003.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 3, "Pin-up - Page 3"),
        ("pinup_p0005.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 5, "Pin-up - Page 5"),
        ("pinup_p0006.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 6, "Pin-up - Page 6"),
        ("pinup_p0007.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 7, "Pin-up - Page 7"),
        ("pinup_p0008.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 8, "Pin-up - Page 8"),
        ("pinup_p0009.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 9, "Pin-up - Page 9"),
        ("pinup_p0010.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 10, "Pin-up - Page 10"),
        ("pinup_p0011.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 11, "Pin-up - Page 11"),
        ("pinup_p0012.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 12, "Pin-up - Page 12"),
        ("pinup_p0013.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 13, "Pin-up - Page 13"),
        ("pinup_p0014.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 14, "Pin-up - Page 14"),
        ("pinup_p0015.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 15, "Pin-up - Page 15"),
        ("pinup_p0016.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 16, "Pin-up - Page 16"),
        ("pinup_p0017.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 17, "Pin-up - Page 17"),
        ("pinup_p0019.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 19, "Pin-up - Page 19"),
        ("pinup_p0020.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 20, "Pin-up - Page 20"),
        ("pinup_p0021.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 21, "Pin-up - Page 21"),
        ("pinup_p0022.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 22, "Pin-up - Page 22"),
        ("pinup_p0023.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 23, "Pin-up - Page 23"),
        ("pinup_p0024.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 24, "Pin-up - Page 24"),
        ("pinup_p0025.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 25, "Pin-up - Page 25"),
        ("pinup_p0026.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 26, "Pin-up - Page 26"),
        ("pinup_p0027.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 27, "Pin-up - Page 27"),
        ("pinup_p0029.json", "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf", 29, "Pin-up - Page 29"),
    ]

    results = []

    for json_file, pdf_path, page_num, description in test_cases:
        if os.path.exists(pdf_path):
            result = compare_page(json_file, pdf_path, page_num)
            if result:
                result['description'] = description
                results.append(result)
        else:
            print(f"⚠️ PDF manquant: {pdf_path}")

    # Synthèse des résultats
    if results:
        print(f"\n📈 SYNTHÈSE GLOBALE ({len(results)} pages testées)")
        print("=" * 60)

        total_gt_panels = sum(r['gt_panels'] for r in results)
        total_det_panels = sum(r['det_panels'] for r in results)
        total_gt_balloons = sum(r['gt_balloons'] for r in results)
        total_det_balloons = sum(r['det_balloons'] for r in results)

        avg_panel_precision = sum(r['panels']['precision'] for r in results) / len(results)
        avg_panel_recall = sum(r['panels']['recall'] for r in results) / len(results)
        avg_panel_f1 = sum(r['panels']['f1_score'] for r in results) / len(results)

        avg_balloon_precision = sum(r['balloons']['precision'] for r in results) / len(results)
        avg_balloon_recall = sum(r['balloons']['recall'] for r in results) / len(results)
        avg_balloon_f1 = sum(r['balloons']['f1_score'] for r in results) / len(results)

        print(f"PANELS:")
        print(f"   Ground truth total: {total_gt_panels}")
        print(f"   Détections total: {total_det_panels}")
        print(f"   Précision moyenne: {avg_panel_precision:.3f}")
        print(f"   Rappel moyen: {avg_panel_recall:.3f}")
        print(f"   F1-Score moyen: {avg_panel_f1:.3f}")

        print(f"\nBALLOONS:")
        print(f"   Ground truth total: {total_gt_balloons}")
        print(f"   Détections total: {total_det_balloons}")
        print(f"   Précision moyenne: {avg_balloon_precision:.3f}")
        print(f"   Rappel moyen: {avg_balloon_recall:.3f}")
        print(f"   F1-Score moyen: {avg_balloon_f1:.3f}")

        # Détail par page
        print(f"\n📋 DÉTAIL PAR PAGE:")
        for r in results:
            print(f"   {r['description']}:")
            print(f"     Panels: {r['det_panels']}/{r['gt_panels']} (F1={r['panels']['f1_score']:.3f})")
            print(f"     Balloons: {r['det_balloons']}/{r['gt_balloons']} (F1={r['balloons']['f1_score']:.3f})")

if __name__ == "__main__":
    main()
