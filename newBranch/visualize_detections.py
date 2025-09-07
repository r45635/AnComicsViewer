#!/usr/bin/env python3
"""
Script pour visualiser les détections et sauvegarder une image avec les overlays
"""

import sys
import os
import yaml
from pathlib import Path

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the detection logic
import main
PdfYoloViewer = main.PdfYoloViewer

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor, QFont

def visualize_detections(pdf_path, output_path="detection_visualization.png"):
    """Visualise les détections et sauvegarde une image"""
    print(f"Visualisation des détections pour: {pdf_path}")

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

    # Create visualization
    if hasattr(viewer, 'qimage_current') and hasattr(viewer, 'dets'):
        # Create a copy of the image for drawing
        vis_image = viewer.qimage_current.copy()
        painter = QPainter(vis_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Set up fonts and pens
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)

        # Draw detections
        for d in viewer.dets:
            if d.cls == 0:  # panel
                color = QColor(35, 197, 83, 180)  # Green with transparency
                pen_color = QColor(35, 197, 83)  # Solid green
                label = "PANEL"
            else:  # balloon
                color = QColor(41, 121, 255, 180)  # Blue with transparency
                pen_color = QColor(41, 121, 255)  # Solid blue
                label = "BALLOON"

            # Draw rectangle
            pen = QPen(pen_color)
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(color)
            painter.drawRect(d.rect.toRect())

            # Draw label
            painter.setPen(QColor(255, 255, 255))  # White text
            painter.setBrush(QColor(0, 0, 0, 150))  # Semi-transparent black background
            label_rect = painter.fontMetrics().boundingRect(f"{label} {d.conf:.2f}")
            label_rect.adjust(-5, -5, 5, 5)
            label_rect.moveTopLeft(d.rect.topLeft().toPoint())
            painter.drawRect(label_rect)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, f"{label} {d.conf:.2f}")

        painter.end()

        # Save the image
        if vis_image.save(output_path):
            print(f"✅ Visualization saved to: {output_path}")
            print(f"   Image size: {vis_image.width()}x{vis_image.height()}")
        else:
            print("❌ Failed to save visualization")

        # Print detection details
        panels = [d for d in viewer.dets if d.cls == 0]
        balloons = [d for d in viewer.dets if d.cls == 1]

        print("\n📊 DETECTIONS:")
        print(f"   Panels: {len(panels)}")
        for i, p in enumerate(panels):
            rect = p.rect
            print(f"     Panel {i+1}: x={rect.x():.0f}, y={rect.y():.0f}, w={rect.width():.0f}, h={rect.height():.0f}, conf={p.conf:.2f}")

        print(f"   Balloons: {len(balloons)}")
        for i, b in enumerate(balloons):
            rect = b.rect
            print(f"     Balloon {i+1}: x={rect.x():.0f}, y={rect.y():.0f}, w={rect.width():.0f}, h={rect.height():.0f}, conf={b.conf:.2f}")

    print("\n✅ Visualization completed")

if __name__ == "__main__":
    # Test with a sample PDF
    pdf_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/Tintin - 161 - Le Lotus Bleu - .pdf"
    output_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/detection_visualization.png"

    if os.path.exists(pdf_path):
        visualize_detections(pdf_path, output_path)
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
