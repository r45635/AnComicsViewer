#!/usr/bin/env python3
"""
Script de debug simple pour analyser les détections YOLO
"""

import sys
import os
import yaml
import numpy as np

# Imports
try:
    import fitz
    from ultralytics import YOLO
    from PIL import Image
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def load_config():
    """Charge la configuration depuis detect.yaml"""
    config_paths = ["config/detect.yaml", "detect.yaml"]
    for path in config_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
    return {}

def analyze_pdf(pdf_path, page_num=0):
    """Analyse simple d'un PDF"""
    print(f"📖 Analyse de {pdf_path}, page {page_num+1}")

    # Charger le PDF
    try:
        pdf = fitz.open(pdf_path)
        if page_num >= len(pdf):
            print(f"❌ Page {page_num+1} n'existe pas")
            return
        page = pdf[page_num]
    except Exception as e:
        print(f"❌ Erreur PDF: {e}")
        return

    # Extraire l'image
    try:
        # Essayer différentes méthodes pour extraire l'image
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        except AttributeError:
            pix = page.getPixmap(matrix=fitz.Matrix(2, 2), alpha=False)

        print(f"📏 Image: {pix.width}x{pix.height} pixels")

        # Convertir en PIL Image
        try:
            img = Image.open(fitz.io.BytesIO(pix.tobytes()))
        except AttributeError:
            img = Image.open(fitz.io.BytesIO(pix.getPNGData()))

        img_array = np.array(img)
        print(f"📷 Array shape: {img_array.shape}, dtype: {img_array.dtype}")

    except Exception as e:
        print(f"❌ Erreur image: {e}")
        pdf.close()
        return

    # Charger le modèle
    model_path = "anComicsViewer_v01.pt"
    if not os.path.exists(model_path):
        print(f"❌ Modèle {model_path} introuvable")
        pdf.close()
        return

    try:
        model = YOLO(model_path)
        print(f"🤖 Modèle chargé: {model_path}")
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        pdf.close()
        return

    # Config
    config = load_config()
    panel_conf = config.get('panel_conf', 0.12)
    balloon_conf = config.get('balloon_conf', 0.15)
    print(f"⚙️  Seuils: panel={panel_conf}, balloon={balloon_conf}")

    # Prédiction
    print("🔍 Détection en cours...")
    try:
        results = model.predict(
            source=img_array,
            imgsz=1024,
            conf=min(panel_conf, balloon_conf),
            iou=0.6,
            max_det=200,
            augment=False,
            verbose=False
        )

        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"✅ {len(boxes)} détections trouvées")

                panels = 0
                balloons = 0
                for i in range(len(boxes)):
                    box = boxes[i]
                    cls = int(box.cls.item() if hasattr(box.cls, 'item') else box.cls)
                    conf = float(box.conf.item() if hasattr(box.conf, 'item') else box.conf)

                    if cls == 0 and conf >= panel_conf:
                        panels += 1
                    elif cls == 1 and conf >= balloon_conf:
                        balloons += 1

                print(f"📊 Résultats: {panels} panels, {balloons} balloons")
                print("✅ SUCCÈS !")
            else:
                print("❌ Aucune détection")
        else:
            print("❌ Aucun résultat")

    except Exception as e:
        print(f"❌ Erreur prédiction: {e}")

    pdf.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_detection.py <pdf_path> [page_num]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    analyze_pdf(pdf_path, page_num)

if __name__ == "__main__":
    main()
