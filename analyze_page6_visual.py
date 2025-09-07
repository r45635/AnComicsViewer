#!/usr/bin/env python3
"""
Script pour analyser visuellement la page 6 et diagnostiquer les problèmes de détection
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au path pour importer main
sys.path.insert(0, str(Path(__file__).parent))

def analyze_page6_visual():
    """Analyse visuelle de la page 6 pour comprendre les problèmes de détection"""

    try:
        import fitz  # PyMuPDF

        # Ouvrir le PDF
        pdf_path = "data/examples/Golden City - T01 - Pilleurs d'épaves.pdf"
        if not os.path.exists(pdf_path):
            print(f"❌ PDF non trouvé: {pdf_path}")
            return

        pdf = fitz.open(pdf_path)
        page = pdf[6]  # Page 6 (0-based)

        # Convertir en image haute résolution
        dpi = 300
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

        # Convertir en numpy array
        img_data = pix.samples
        img = np.frombuffer(img_data, dtype=np.uint8)
        img = img.reshape(pix.h, pix.w, pix.n)

        print(f"📊 Analyse de la page 6:")
        print(f"   📏 Dimensions: {pix.w}x{pix.h}")
        print(f"   🎨 Format: {pix.n} canaux")

        # Analyse des canaux de couleur
        b, g, r = cv2.split(img)
        print("   📈 Statistiques des canaux:")
        print(f"      Rouge - min:{r.min()}, max:{r.max()}, mean:{r.mean():.1f}")
        print(f"      Vert  - min:{g.min()}, max:{g.max()}, mean:{g.mean():.1f}")
        print(f"      Bleu  - min:{b.min()}, max:{b.max()}, mean:{b.mean():.1f}")

        # Analyse du contraste
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        contrast = gray.std()
        print(f"   🔆 Contraste: {contrast:.2f}")

        # Détection de contours basique
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"   📦 Contours détectés: {len(contours)}")

        # Filtrer les contours par taille
        min_area = (pix.w * pix.h) * 0.005  # 0.5% de la page
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        print(f"   📦 Grands contours (>0.5%): {len(large_contours)}")

        # Sauvegarder l'image pour analyse
        output_dir = "debug_page6_analysis"
        os.makedirs(output_dir, exist_ok=True)

        # Image originale
        cv2.imwrite(f"{output_dir}/page6_original.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Image en niveaux de gris
        cv2.imwrite(f"{output_dir}/page6_gray.jpg", gray)

        # Contours
        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
        cv2.imwrite(f"{output_dir}/page6_contours.jpg", cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR))

        # Seuillage adaptatif pour voir les zones sombres/claires
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(f"{output_dir}/page6_threshold.jpg", thresh)

        print(f"   💾 Images sauvegardées dans: {output_dir}/")

        # Analyse des panels potentiels
        print("   🔍 Analyse des panels potentiels:")
        for i, contour in enumerate(large_contours[:10]):  # Top 10
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area_pct = (area / (pix.w * pix.h)) * 100

            print(f"      Panel {i}: ({x},{y}) {w}x{h}, area={area_pct:.2f}%, ratio={aspect_ratio:.2f}")

        pdf.close()

    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_page6_visual()
