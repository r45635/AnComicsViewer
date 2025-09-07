#!/usr/bin/env python3
"""
Diagnostic détaillé de la détection par règles sur la page 6
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

def diagnose_page6_detection():
    """Diagnostiquer les problèmes de détection par règles sur la page 6"""

    try:
        import fitz  # PyMuPDF

        # Ouvrir le PDF
        pdf_path = "data/examples/Golden City - T01 - Pilleurs d'épaves.pdf"
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

        print("🔍 DIAGNOSTIC DÉTAILLÉ DE LA PAGE 6")
        print("=" * 50)

        # Analyse préliminaire
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        page_area = pix.w * pix.h

        print(f"📊 Dimensions: {pix.w}x{pix.h} ({page_area} pixels)")
        print(f"📏 Zone par défaut: {page_area * 0.02:.0f} pixels (2%)")

        # Test de différents seuils de contour
        print("\n🔧 TEST DE SEUILS DE CONTOUR:")
        thresholds = [30, 50, 70, 100, 150]

        for thresh in thresholds:
            edges = cv2.Canny(gray, thresh, thresh * 2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filtrer par taille
            min_area_1pct = page_area * 0.01   # 1%
            min_area_2pct = page_area * 0.02   # 2%
            min_area_5pct = page_area * 0.05   # 5%

            contours_1pct = [c for c in contours if cv2.contourArea(c) > min_area_1pct]
            contours_2pct = [c for c in contours if cv2.contourArea(c) > min_area_2pct]
            contours_5pct = [c for c in contours if cv2.contourArea(c) > min_area_5pct]

            print(f"   Seuil {thresh:3d}: {len(contours):4d} contours → 1%: {len(contours_1pct):2d}, 2%: {len(contours_2pct):2d}, 5%: {len(contours_5pct):2d}")

        # Analyse des contours avec le seuil optimal
        print("\n🎯 ANALYSE AVEC SEUIL OPTIMAL (50):")
        edges = cv2.Canny(gray, 50, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrer par taille (2% comme dans la config)
        min_area = page_area * 0.02
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        print(f"   Contours totaux: {len(contours)}")
        print(f"   Contours >2%: {len(large_contours)}")

        # Analyser les caractéristiques des contours
        print("\n📋 CARACTÉRISTIQUES DES CONTOURS >2%:")
        for i, contour in enumerate(large_contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area_pct = (area / page_area) * 100

            # Calculer la compacité (aire/périmètre²)
            perimeter = cv2.arcLength(contour, True)
            compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

            print(f"   {i+1:2d}: ({x:4d},{y:4d}) {w:4d}x{h:4d} | Aire:{area_pct:5.2f}% | Ratio:{aspect_ratio:.2f} | Compact:{compactness:.3f}")

            # Vérifier si c'est un panel potentiel
            is_panel_like = (
                0.3 <= aspect_ratio <= 3.0 and  # Ratio d'aspect raisonnable
                area_pct >= 2.0 and             # Taille minimum
                area_pct <= 80.0 and            # Taille maximum
                compactness > 0.1               # Forme pas trop complexe
            )

            status = "✅ PANEL" if is_panel_like else "❌ REJETÉ"
            print(f"       {status}")

        # Test avec seuillage adaptatif
        print("\n🔧 TEST SEUILLAGE ADAPTATIF:")
        for block_size in [11, 21, 31]:
            for c in [2, 5, 8]:
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, c
                )

                # Inverser pour détecter les zones sombres
                thresh_inv = cv2.bitwise_not(thresh)
                contours_adapt, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                large_contours_adapt = [c for c in contours_adapt if cv2.contourArea(c) > min_area]

                print(f"   Block{block_size:2d} C{c}: {len(contours_adapt):4d} contours → >2%: {len(large_contours_adapt):2d}")

        # Recommandations
        print("\n💡 RECOMMANDATIONS:")
        print("   1. La page 6 a un contraste plus faible (62.81 vs ~70-80)")
        print("   2. Seuils de contour plus bas pourraient aider (30-50 au lieu de 50-100)")
        print("   3. Seuillage adaptatif pourrait mieux détecter les panels")
        print("   4. Les panels de cette page sont plus grands que la moyenne")

        pdf.close()

    except Exception as e:
        print(f"❌ Erreur lors du diagnostic: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_page6_detection()
