#!/usr/bin/env python3
"""
Compare Pin-up Annotations vs Current Detection Results
======================================================
Compare expected panel counts from JSON annotations with actual detection results
"""

import sys
import os
import json
import fitz
from pathlib import Path
import yaml

# Add project paths
sys.path.insert(0, '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer')

def load_json_annotation(json_path):
    """Load annotation from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    panels = []
    balloons = []

    for shape in data['shapes']:
        if shape['label'] == 'panel':
            # Convert from [[x1,y1],[x2,y2]] to QRectF
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            panels.append({
                'x': x1, 'y': y1,
                'width': x2 - x1,
                'height': y2 - y1
            })
        elif shape['label'] == 'balloon':
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            balloons.append({
                'x': x1, 'y': y1,
                'width': x2 - x1,
                'height': y2 - y1
            })

    return {
        'panels': panels,
        'balloons': balloons,
        'image_width': data['imageWidth'],
        'image_height': data['imageHeight']
    }

def run_detection_on_page(pdf_path, page_num):
    """Run detection on a specific page using the full pipeline logic"""
    try:
        # Import detection modules
        from ultralytics import YOLO
        import numpy as np
        from PySide6.QtGui import QImage
        from PySide6.QtCore import QRectF
        import yaml

        # Load PDF page with higher DPI
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        dpi = 300; zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888).copy()
        doc.close()

        # Load config
        config = {}
        config_path = '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}

        # Load model
        model_path = '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/anComicsViewer_v01.pt'
        if not os.path.exists(model_path):
            return None, "Model not found"

        model = YOLO(model_path)

        # Convert to numpy array
        img = np.frombuffer(bytes(qimg.constBits()), dtype=np.uint8)
        img = img.reshape(qimg.height(), qimg.bytesPerLine())[:, :qimg.width()*3]
        img = img.reshape(qimg.height(), qimg.width(), 3)[:, :, :3].copy()

        # Get raw detections
        # Run detection with proper thresholds
        results = model.predict(
            source=img,
            conf=min(config.get('panel_conf', 0.12), config.get('balloon_conf', 0.15)),
            iou=0.6,
            max_det=config.get('max_det', 400),
            augment=False,
            verbose=False,
            device="mps" if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "1") else None
        )[0]

        # Convert to QRectF format for post-processing
        raw_dets = []
        if results and hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                rect = QRectF(float(x1), float(y1), float(x2-x1), float(y2-y1))
                raw_dets.append((cls, conf, rect))

        # Apply post-processing similar to main.py
        W, H = qimg.width(), qimg.height()
        PAGE_AREA = float(W * H)

        # Separate panels and balloons
        all_panels = [(c, p, r) for (c, p, r) in raw_dets if c == 0]
        all_balloons = [(c, p, r) for (c, p, r) in raw_dets if c == 1]

        # Filter panels
        panels = []
        for c, p, r in all_panels:
            if p < config.get('panel_conf', 0.12):
                continue
            if r.width() * r.height() < config.get('panel_area_min_pct', 0.01) * PAGE_AREA:
                continue
            panels.append((c, p, r))

        # Filter balloons
        balloons = []
        for c, p, r in all_balloons:
            if p < config.get('balloon_conf', 0.15):
                continue
            if r.width() * r.height() < config.get('balloon_area_min_pct', 0.0003) * PAGE_AREA:
                continue
            if r.width() < config.get('balloon_min_w', 30) or r.height() < config.get('balloon_min_h', 22):
                continue
            balloons.append((c, p, r))

        # Apply merging if enabled
        if config.get('enable_panel_merge', True) and panels:
            panel_rects = [r for (_, _, r) in panels]

            # Apply IoU-based merging
            merged_rects = []
            used = [False] * len(panel_rects)
            
            for i, rect1 in enumerate(panel_rects):
                if used[i]:
                    continue
                current_rect = QRectF(rect1)
                used[i] = True
                
                # Check all other rectangles for merging
                changed = True
                while changed:
                    changed = False
                    for j, rect2 in enumerate(panel_rects):
                        if used[j]:
                            continue
                        
                        # Calculate IoU
                        inter = current_rect.intersected(rect2)
                        if inter.isEmpty():
                            continue
                        inter_area = inter.width() * inter.height()
                        union_area = (current_rect.width() * current_rect.height() + rect2.width() * rect2.height() - inter_area)
                        iou = inter_area / union_area if union_area > 0 else 0
                        
                        # Calculate center distance
                        dx = abs(current_rect.center().x() - rect2.center().x())
                        dy = abs(current_rect.center().y() - rect2.center().y())
                        dist = (dx**2 + dy**2)**0.5
                        max_dim = max(W, H)
                        dist_thresh_px = config.get('panel_merge_dist', 0.02) * max_dim
                        
                        # Merge if IoU or distance criteria are met
                        if iou > config.get('iou_merge', 0.25) or dist < dist_thresh_px:
                            current_rect = current_rect.united(rect2)
                            used[j] = True
                            changed = True
                
                merged_rects.append(current_rect)
            
            panel_rects = merged_rects

            # Apply row-based merging if enabled
            if config.get('enable_row_merge', True):
                def v_overlap(a, b):
                    inter_h = max(0.0, min(a.bottom(), b.bottom()) - max(a.top(), b.top()))
                    return inter_h / max(1e-6, min(a.height(), b.height()))
                
                rows = []
                used = [False] * len(panel_rects)
                
                for i, r in enumerate(panel_rects):
                    if used[i]:
                        continue
                    bucket = [r]
                    used[i] = True
                    
                    for j, s in enumerate(panel_rects):
                        if used[j]:
                            continue
                        if v_overlap(r, s) >= config.get('panel_row_overlap', 0.35):
                            bucket.append(s)
                            used[j] = True
                    
                    rows.append(bucket)
                
                merged_rows = []
                max_gap = W * config.get('panel_row_gap_pct', 0.02)
                
                for bucket in rows:
                    bucket = sorted(bucket, key=lambda rr: rr.left())
                    cur = bucket[0]
                    for nx in bucket[1:]:
                        if (nx.left() - cur.right()) <= max_gap:
                            cur = cur.united(nx)
                        else:
                            merged_rows.append(cur)
                            cur = nx
                    merged_rows.append(cur)
                
                panel_rects = merged_rows

            # Apply containment-based filtering
            kept_rects = []
            for i, ri in enumerate(panel_rects):
                drop = False
                for j, rj in enumerate(panel_rects):
                    if i == j:
                        continue
                    # Calculate containment
                    inter = ri.intersected(rj)
                    if not inter.isEmpty():
                        inter_area = inter.width() * inter.height()
                        ri_area = ri.width() * ri.height()
                        containment = inter_area / ri_area if ri_area > 0 else 0
                        if containment > config.get('panel_containment_merge', 0.55):
                            drop = True
                            break
                if not drop:
                    kept_rects.append(ri)
            
            panel_rects = kept_rects

            # Reconstruct panels with merged rectangles
            best_conf = max([p for (_, p, _) in panels], default=0.5)
            panels = [(0, best_conf, rr) for rr in panel_rects]

        # Apply full-page detection
        if panels:
            def _area(r):
                return max(0.0, r.width() * r.height())
                
            largest_panel = max(panels, key=lambda t: _area(t[2]))
            largest_area = _area(largest_panel[2])
            
            if largest_area / max(1e-6, PAGE_AREA) >= config.get('full_page_panel_pct', 0.93):
                # This is a full-page panel
                page_rect = QRectF(0, 0, W, H)
                panels = [(0, largest_panel[1], page_rect)]
                
                # Filter balloons that overlap with the full-page panel
                if config.get('full_page_keep_balloons', True):
                    filtered_balloons = []
                    for c, p, r in balloons:
                        inter = r.intersected(page_rect)
                        if not inter.isEmpty():
                            inter_area = inter.width() * inter.height()
                            r_area = r.width() * r.height()
                            overlap = inter_area / r_area if r_area > 0 else 0
                            if overlap >= config.get('full_page_balloon_overlap_pct', 0.12):
                                filtered_balloons.append((c, p, r))
                    balloons = filtered_balloons
                else:
                    balloons = []

        # Convert back to detection format
        detected_panels = []
        detected_balloons = []
        
        for c, p, r in panels:
            detected_panels.append({
                'x': r.x(), 'y': r.y(),
                'width': r.width(), 'height': r.height(),
                'conf': p
            })
        
        for c, p, r in balloons:
            detected_balloons.append({
                'x': r.x(), 'y': r.y(),
                'width': r.width(), 'height': r.height(),
                'conf': p
            })

        return {
            'panels': detected_panels,
            'balloons': detected_balloons,
            'image_width': qimg.width(),
            'image_height': qimg.height()
        }, None

    except Exception as e:
        return None, str(e)

def compare_page(pdf_path, page_num, annotation_data, config):
    """Compare expected vs detected for a specific page"""
    print(f"\n🔍 ANALYSE PAGE {page_num}")
    print("=" * 50)

    # Expected from annotations
    expected_panels = len(annotation_data['panels'])
    expected_balloons = len(annotation_data['balloons'])

    print(f"📋 ATTENDU: {expected_panels} panels, {expected_balloons} balloons")
    print(f"   Image: {annotation_data['image_width']}x{annotation_data['image_height']}")

    # Run detection
    detected, error = run_detection_on_page(pdf_path, page_num)

    if error:
        print(f"❌ ERREUR DÉTECTION: {error}")
        return

    detected_panels = len(detected['panels'])
    detected_balloons = len(detected['balloons'])

    print(f"🤖 DÉTECTÉ: {detected_panels} panels, {detected_balloons} balloons")
    print(f"   Image: {detected['image_width']}x{detected['image_height']}")

    # Analysis
    print(f"\n📊 ANALYSE:")
    if detected_panels == expected_panels:
        print("   ✅ PANELS: Nombre correct")
    else:
        diff = abs(detected_panels - expected_panels)
        if detected_panels < expected_panels:
            print(f"   ⚠️ PANELS: {diff} manquants")
        else:
            print(f"   ⚠️ PANELS: {diff} faux positifs")

    if detected_balloons == expected_balloons:
        print("   ✅ BALLOONS: Nombre correct")
    else:
        diff = abs(detected_balloons - expected_balloons)
        if detected_balloons < expected_balloons:
            print(f"   ⚠️ BALLOONS: {diff} manquants")
        else:
            print(f"   ⚠️ BALLOONS: {diff} faux positifs")

    # Show details
    if detected_panels > 0:
        print("\n📏 PANELS DÉTECTÉS:")
        page_area = detected['image_width'] * detected['image_height']
        min_area = page_area * config.get('panel_area_min_pct', 0.01)
        print(f"   Page area: {page_area}, Min panel area: {min_area:.0f}")
        for i, panel in enumerate(detected['panels']):
            area = panel['width'] * panel['height']
            status = "✅" if area >= min_area else "❌ TOO SMALL"
            print(f"   DET {i+1}: pos=({panel['x']:.0f},{panel['y']:.0f}) taille={panel['width']:.0f}x{panel['height']:.0f} area={area:.0f} conf={panel['conf']:.2f} {status}")

    if expected_panels > 0:
        print("\n📏 PANELS ATTENDUS:")
        expected_page_area = annotation_data['image_width'] * annotation_data['image_height']
        for i, panel in enumerate(annotation_data['panels']):
            area = panel['width'] * panel['height']
            print(f"   EXP {i+1}: pos=({panel['x']:.0f},{panel['y']:.0f}) taille={panel['width']:.0f}x{panel['height']:.0f} area={area:.0f}")

def main():
    pdf_path = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/data/examples/La Pin-up du B24 - T01.pdf"
    annotations_dir = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/backup_annotations_20250822_182146"

    # Load config
    config = {}
    config_path = '/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/newBranch/config/detect.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

    print("🔍 COMPARAISON PIN-UP: Annotations vs Détection Actuelle")
    print("=" * 70)

    # Test multiple pages
    test_pages = [
        (1, "pinup_p0001.json", "Page 1 - 1 panel attendu"),
        (3, "pinup_p0003.json", "Page 3 - 2 panels attendus"),
        (5, "pinup_p0005.json", "Page 5 - 6 panels attendus"),
        (6, "pinup_p0006.json", "Page 6 - 4 panels attendus"),
    ]

    for page_num, json_file, description in test_pages:
        json_path = os.path.join(annotations_dir, json_file)

        if not os.path.exists(json_path):
            print(f"\n❌ Annotation manquante: {json_file}")
            continue

        try:
            annotation_data = load_json_annotation(json_path)
            compare_page(pdf_path, page_num, annotation_data, config)
        except Exception as e:
            print(f"\n❌ Erreur chargement {json_file}: {e}")

    print("\n🎉 COMPARAISON TERMINÉE!")

if __name__ == "__main__":
    main()
