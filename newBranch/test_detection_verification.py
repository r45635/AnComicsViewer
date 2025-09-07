#!/usr/bin/env python3
"""
Test script to verify panel detection on a sample PDF
"""

import sys
import os
import yaml
from pathlib import Path

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the detection logic
from main import PdfYoloViewer

def test_detection(pdf_path):
    """Test detection on a PDF file"""
    print(f"Testing detection on: {pdf_path}")

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

    # Print results
    if hasattr(viewer, 'dets'):
        panels = [d for d in viewer.dets if d.cls == 0]
        balloons = [d for d in viewer.dets if d.cls == 1]
        print("\n📊 DETECTION RESULTS:")
        print(f"   Panels: {len(panels)}")
        print(f"   Balloons: {len(balloons)}")

        if panels:
            areas = [d.rect.width() * d.rect.height() for d in panels]
            print(f"   Panel areas: min={min(areas):.0f}, max={max(areas):.0f}, avg={sum(areas)/len(areas):.0f}")

        if balloons:
            print(f"   Balloons found!")
        else:
            print("   No balloons detected")

    print("\n✅ Test completed")

if __name__ == "__main__":
    # Test with a sample PDF
    pdf_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf"

    if os.path.exists(pdf_path):
        test_detection(pdf_path)
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
