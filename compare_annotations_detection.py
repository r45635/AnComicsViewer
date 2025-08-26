#!/usr/bin/env python3
"""
Comparaison visuelle annotations vs détections YOLO
===================================================
Compare les annotations ground truth avec les détections du modèle
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QFont
from PySide6.QtCore import Qt, QRectF

def load_yolo_annotations(annotation_file):
    """Charge les annotations YOLO (format: class cx cy w h normalisés)"""
    annotations = []
    if os.path.exists(annotation_file):
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    annotations.append({
                        'class': cls,
                        'cx': cx, 'cy': cy,
                        'w': w, 'h': h
                    })
    return annotations

def yolo_to_pixels(annotation, img_width, img_height):
    """Convertit annotation YOLO normalisée en pixels"""
    cx_px = annotation['cx'] * img_width
    cy_px = annotation['cy'] * img_height
    w_px = annotation['w'] * img_width
    h_px = annotation['h'] * img_height
    
    x1 = cx_px - w_px / 2
    y1 = cy_px - h_px / 2
    
    return QRectF(x1, y1, w_px, h_px)

def compare_annotations_vs_detection():
    """Compare annotations ground truth vs détections modèle"""
    
    if not QApplication.instance():
        app = QApplication(sys.argv)
    
    print("🔍 COMPARAISON ANNOTATIONS vs DÉTECTIONS")
    print("=" * 60)
    
    # Test avec plusieurs images annotées
    test_cases = [
        ("p0004", "Page avec 7 panels"),
        ("p0015", "Page avec 6 panels"), 
        ("p0026", "Page complexe"),
        ("p0029", "Page simple")
    ]
    
    for page_id, description in test_cases:
        print(f"\n📋 TEST: {page_id} - {description}")
        print("=" * 50)
        
        # Chemins
        img_path = f"dataset/yolo_single_class/images/val/{page_id}.png"
        ann_path = f"dataset/yolo_single_class/labels/val/{page_id}.txt"
        
        if not os.path.exists(img_path) or not os.path.exists(ann_path):
            print(f"❌ Fichiers manquants pour {page_id}")
            continue
            
        # Charger image
        qimg = QImage(img_path)
        if qimg.isNull():
            print(f"❌ Impossible de charger {img_path}")
            continue
            
        print(f"🖼️ Image: {qimg.width()}x{qimg.height()}")
        
        # Charger annotations ground truth
        annotations = load_yolo_annotations(ann_path)
        print(f"📝 Annotations GT: {len(annotations)} panels")
        
        # Convertir annotations en rectangles pixels
        gt_rects = []
        for i, ann in enumerate(annotations):
            rect = yolo_to_pixels(ann, qimg.width(), qimg.height())
            gt_rects.append(rect)
            print(f"   GT {i+1}: pos=({rect.x():.0f},{rect.y():.0f}) taille={rect.width():.0f}x{rect.height():.0f}")
        
        # Détection avec notre modèle
        try:
            from src.ancomicsviewer.detectors.robust_yolo_detector import RobustYoloDetector
            detector = RobustYoloDetector()
            detected_panels = detector.detect_panels(qimg)
            print(f"🤖 Détections modèle: {len(detected_panels)} panels")
            
            for i, panel in enumerate(detected_panels):
                print(f"   DET {i+1}: pos=({panel.x():.0f},{panel.y():.0f}) taille={panel.width():.0f}x{panel.height():.0f}")
        
        except Exception as e:
            print(f"❌ Erreur détection: {e}")
            detected_panels = []
        
        # Analyse comparative
        print(f"\n📊 ANALYSE:")
        print(f"   Ground Truth: {len(gt_rects)} panels")
        print(f"   Détections: {len(detected_panels)} panels")
        
        if len(detected_panels) < len(gt_rects):
            print(f"   ⚠️ SOUS-DÉTECTION: {len(gt_rects) - len(detected_panels)} panels manqués")
        elif len(detected_panels) > len(gt_rects):
            print(f"   ⚠️ SUR-DÉTECTION: {len(detected_panels) - len(gt_rects)} faux positifs")
        else:
            print(f"   ✅ NOMBRE CORRECT")
        
        # Créer image de comparaison visuelle
        comparison_img = qimg.copy()
        painter = QPainter(comparison_img)
        
        # Dessiner ground truth en vert
        painter.setPen(QPen(QColor(0, 255, 0), 3))  # Vert pour GT
        for i, rect in enumerate(gt_rects):
            painter.drawRect(rect)
            painter.drawText(int(rect.x() + 5), int(rect.y() + 20), f"GT{i+1}")
        
        # Dessiner détections en rouge
        painter.setPen(QPen(QColor(255, 0, 0), 2))  # Rouge pour détections
        for i, rect in enumerate(detected_panels):
            painter.drawRect(rect)
            painter.drawText(int(rect.x() + 5), int(rect.y() + 35), f"DET{i+1}")
        
        # Légende
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawText(10, qimg.height() - 40, f"VERT=Ground Truth ({len(gt_rects)})")
        painter.drawText(10, qimg.height() - 20, f"ROUGE=Détections ({len(detected_panels)})")
        
        painter.end()
        
        # Sauvegarder comparaison
        comp_path = f"COMPARISON_{page_id}_gt_vs_det.png"
        comparison_img.save(comp_path)
        print(f"💾 Comparaison sauvée: {comp_path}")
        
        # Calcul IoU approximatif
        if len(detected_panels) > 0 and len(gt_rects) > 0:
            print(f"\n🎯 ANALYSE IoU:")
            for i, det_rect in enumerate(detected_panels):
                best_iou = 0
                best_gt = -1
                
                for j, gt_rect in enumerate(gt_rects):
                    # Calcul IoU simple
                    inter = det_rect.intersected(gt_rect)
                    if not inter.isEmpty():
                        inter_area = inter.width() * inter.height()
                        union_area = (det_rect.width() * det_rect.height() + 
                                    gt_rect.width() * gt_rect.height() - inter_area)
                        iou = inter_area / union_area if union_area > 0 else 0
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = j
                
                if best_iou > 0.1:
                    print(f"   DET{i+1} ↔ GT{best_gt+1}: IoU={best_iou:.3f}")
                else:
                    print(f"   DET{i+1}: SANS CORRESPONDANCE (IoU<0.1)")
    
    print(f"\n🎉 COMPARAISON TERMINÉE!")
    print(f"📁 Vérifiez les images COMPARISON_*.png")

if __name__ == "__main__":
    compare_annotations_vs_detection()
