#!/usr/bin/env python3
"""
Script simple pour analyser les détections sans interface graphique
"""

import sys
import os
import yaml

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the detection logic
import main
PdfYoloViewer = main.PdfYoloViewer

def analyze_detections(pdf_path):
    """Analyse les détections sans GUI"""
    print(f"Analyse des détections pour: {pdf_path}")

    # Create a viewer instance (without GUI)
    viewer = PdfYoloViewer.__new__(PdfYoloViewer)  # Create without __init__ to avoid GUI

    # Initialize necessary attributes
    viewer.model = None
    viewer.class_names = ["panel", "balloon"]
    viewer.conf_thres = 0.15
    viewer.iou_thres = 0.6
    viewer.max_det = 200
    viewer.show_panels = True
    viewer.show_balloons = True

    # Load model
    try:
        import torch
        from ultralytics import YOLO

        # Try to load the improved model
        model_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/models/multibd_enhanced_v2.pt"
        if os.path.exists(model_path):
            viewer.model = YOLO(model_path)
            print("✅ Loaded enhanced model")
        else:
            # Fallback to local model
            local_model = os.path.join(os.path.dirname(__file__), "anComicsViewer_v01.pt")
            if os.path.exists(local_model):
                viewer.model = YOLO(local_model)
                print("✅ Loaded local model")
            else:
                print("❌ No model found")
                return
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load PDF
    try:
        import fitz
        viewer.pdf = fitz.open(pdf_path)
        viewer.page_index = 0
        print(f"✅ Loaded PDF: {len(viewer.pdf)} pages")
    except Exception as e:
        print(f"❌ Failed to load PDF: {e}")
        return

    # Load page
    try:
        page = viewer.pdf[0]
        dpi = 300
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        from PySide6.QtGui import QImage
        viewer.qimage_current = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888).copy()
        print(f"✅ Loaded page: {viewer.qimage_current.width()}x{viewer.qimage_current.height()}")
    except Exception as e:
        print(f"❌ Failed to load page: {e}")
        return

    # Run detection
    print("\n🔍 Running detection...")
    viewer._run_detection()

    # Print detailed results
    if hasattr(viewer, 'dets'):
        panels = [d for d in viewer.dets if d.cls == 0]
        balloons = [d for d in viewer.dets if d.cls == 1]

        print("\n📊 RÉSULTATS DÉTAILLÉS:")
        print(f"   Panels détectés: {len(panels)}")
        print(f"   Balloons détectés: {len(balloons)}")

        print("\n🔲 PANELS DÉTECTÉS:")
        for i, p in enumerate(panels):
            rect = p.rect
            area_px = rect.width() * rect.height()
            area_pct = area_px / (viewer.qimage_current.width() * viewer.qimage_current.height()) * 100
            print(f"     Panel {i+1}: conf={p.conf:.2f}")
            print(f"     Position: x={rect.x():.0f}, y={rect.y():.0f}")
            print(f"     Taille: {rect.width():.0f}x{rect.height():.0f} px")
            print(f"     Aire: {area_pct:.2f}% de la page")

        if balloons:
            print("\n💬 BALLOONS DÉTECTÉS:")
            for i, b in enumerate(balloons):
                rect = b.rect
                area_px = rect.width() * rect.height()
                area_pct = area_px / (viewer.qimage_current.width() * viewer.qimage_current.height()) * 100
                print(f"     Balloon {i+1}: conf={b.conf:.2f}")
                print(f"     Position: x={rect.x():.0f}, y={rect.y():.0f}")
                print(f"     Taille: {rect.width():.0f}x{rect.height():.0f} px")
                print(f"     Aire: {area_pct:.2f}% de la page")

        # Analyse de la distribution
        if panels:
            print("\n📈 ANALYSE DE DISTRIBUTION:")
            widths = [p.rect.width() for p in panels]
            heights = [p.rect.height() for p in panels]
            areas = [p.rect.width() * p.rect.height() for p in panels]

            print(f"     Nombre total de panels: {len(panels)}")
            print(f"     Aire moyenne: {sum(areas)/len(areas):.0f} px²")
            print(f"     Largeur moyenne: {sum(widths)/len(widths):.0f} px")
            print(f"     Hauteur moyenne: {sum(heights)/len(heights):.0f} px")

            # Vérifier si les panels sont réalistes pour une BD
            page_width = viewer.qimage_current.width()
            page_height = viewer.qimage_current.height()
            avg_panel_width_pct = (sum(widths)/len(widths)) / page_width * 100
            avg_panel_height_pct = (sum(heights)/len(heights)) / page_height * 100

            print(f"     Largeur moyenne: {avg_panel_width_pct:.1f}% de la page")
            print(f"     Hauteur moyenne: {avg_panel_height_pct:.1f}% de la page")

            # Pour une BD typique, les panels font souvent 20-40% de la largeur de page
            if avg_panel_width_pct < 10:
                print("⚠️  ATTENTION: Panels très étroits (<10% de la page)")
            elif avg_panel_width_pct > 60:
                print("⚠️  ATTENTION: Panels très larges (>60% de la page)")
            else:
                print("✅ Largeur des panels semble réaliste")

    print("\n✅ Analyse terminée")

if __name__ == "__main__":
    # Test with a sample PDF
    pdf_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf"

    if os.path.exists(pdf_path):
        analyze_detections(pdf_path)
    else:
        print(f"PDF not found: {pdf_path}")
        # List available PDFs
        examples_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples"
        if os.path.exists(examples_dir):
            pdfs = [f for f in os.listdir(examples_dir) if f.endswith('.pdf')]
            print("Available PDFs:")
            for pdf in pdfs:
                print(f"  {pdf}")
        else:
            print("No examples directory found")
