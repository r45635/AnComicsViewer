#!/usr/bin/env python3
"""
Two-Detector Pipeline for Comic Detection
Pipeline: Panels First → Crop → Balloons Detection
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwoDetectorPipeline:
    """Two-stage detection pipeline: Panels → Balloons"""

    def __init__(self, panel_model_path: str, balloon_model_path: str):
        self.panel_model = None
        self.balloon_model = None

        # Load models
        if os.path.exists(panel_model_path):
            self.panel_model = YOLO(panel_model_path)
            logger.info(f"Loaded panel model: {panel_model_path}")
        else:
            logger.warning(f"Panel model not found: {panel_model_path}")

        if os.path.exists(balloon_model_path):
            self.balloon_model = YOLO(balloon_model_path)
            logger.info(f"Loaded balloon model: {balloon_model_path}")
        else:
            logger.warning(f"Balloon model not found: {balloon_model_path}")

    def detect_panels_first(self, image_path: str, conf_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """First stage: Detect panels in full image"""
        if not self.panel_model:
            logger.error("Panel model not loaded")
            return []

        # Load image
        image = Image.open(image_path)
        img_width, img_height = image.size

        # Run panel detection
        results = self.panel_model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=0.3,  # Class-aware NMS
            max_det=50,
            verbose=False
        )[0]

        panels = []
        if results.boxes is not None:
            for i, (box, conf, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
                if int(cls) == 0:  # Panel class
                    x1, y1, x2, y2 = box.cpu().numpy()
                    panel = {
                        'id': i,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'class': 'panel',
                        'area': (x2 - x1) * (y2 - y1),
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                    }
                    panels.append(panel)

        # Sort panels by position (top-left to bottom-right)
        panels.sort(key=lambda p: (p['center'][1], p['center'][0]))

        logger.info(f"Detected {len(panels)} panels in {Path(image_path).name}")
        return panels

    def crop_panel_region(self, image: Image.Image, panel_bbox: List[float],
                         margin_pct: float = 0.1) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """Crop panel region with margin"""
        x1, y1, x2, y2 = panel_bbox
        img_width, img_height = image.size

        # Add margin
        margin_x = (x2 - x1) * margin_pct
        margin_y = (y2 - y1) * margin_pct

        crop_x1 = max(0, int(x1 - margin_x))
        crop_y1 = max(0, int(y1 - margin_y))
        crop_x2 = min(img_width, int(x2 + margin_x))
        crop_y2 = min(img_height, int(y2 + margin_y))

        # Crop image
        cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        return cropped, (crop_x1, crop_y1, crop_x2, crop_y2)

    def detect_balloons_in_panel(self, panel_crop: Image.Image, panel_bbox: List[float],
                                crop_coords: Tuple[int, int, int, int],
                                conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Second stage: Detect balloons within panel crop"""
        if not self.balloon_model:
            logger.error("Balloon model not loaded")
            return []

        # Run balloon detection on crop
        results = self.balloon_model.predict(
            source=np.array(panel_crop),
            conf=conf_threshold,
            iou=0.3,  # Class-aware NMS
            max_det=20,
            verbose=False
        )[0]

        balloons = []
        if results.boxes is not None:
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords

            for i, (box, conf, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
                if int(cls) == 1:  # Balloon class
                    bx1, by1, bx2, by2 = box.cpu().numpy()

                    # Convert crop coordinates back to original image coordinates
                    orig_x1 = crop_x1 + bx1
                    orig_y1 = crop_y1 + by1
                    orig_x2 = crop_x1 + bx2
                    orig_y2 = crop_y1 + by2

                    balloon = {
                        'id': i,
                        'bbox': [float(orig_x1), float(orig_y1), float(orig_x2), float(orig_y2)],
                        'confidence': float(conf),
                        'class': 'balloon',
                        'panel_id': None,  # Will be assigned later
                        'area': (orig_x2 - orig_x1) * (orig_y2 - orig_y1),
                        'center': [(orig_x1 + orig_x2) / 2, (orig_y1 + orig_y2) / 2]
                    }
                    balloons.append(balloon)

        return balloons

    def assign_balloons_to_panels(self, panels: List[Dict], balloons: List[Dict],
                                 iou_threshold: float = 0.1) -> List[Dict]:
        """Assign balloons to their containing panels"""
        assigned_balloons = []

        for balloon in balloons:
            bx1, by1, bx2, by2 = balloon['bbox']
            balloon_center = balloon['center']

            best_panel = None
            best_iou = 0

            for panel in panels:
                px1, py1, px2, py2 = panel['bbox']

                # Calculate IoU
                inter_x1 = max(bx1, px1)
                inter_y1 = max(by1, py1)
                inter_x2 = min(bx2, px2)
                inter_y2 = min(by2, py2)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    balloon_area = (bx2 - bx1) * (by2 - by1)
                    panel_area = (px2 - px1) * (py2 - py1)
                    union_area = balloon_area + panel_area - inter_area

                    iou = inter_area / union_area if union_area > 0 else 0

                    if iou > best_iou:
                        best_iou = iou
                        best_panel = panel

            if best_iou >= iou_threshold and best_panel is not None:
                balloon['panel_id'] = best_panel['id']
                assigned_balloons.append(balloon)

        logger.info(f"Assigned {len(assigned_balloons)} balloons to panels")
        return assigned_balloons

    def run_pipeline(self, image_path: str) -> Dict[str, Any]:
        """Run complete two-detector pipeline"""
        logger.info(f"Running two-detector pipeline on {Path(image_path).name}")

        # Stage 1: Detect panels
        panels = self.detect_panels_first(image_path)

        if not panels:
            logger.warning("No panels detected, skipping balloon detection")
            return {'panels': [], 'balloons': []}

        # Load image for cropping
        image = Image.open(image_path)

        # Stage 2: Detect balloons in each panel
        all_balloons = []
        for panel in panels:
            panel_crop, crop_coords = self.crop_panel_region(image, panel['bbox'])
            panel_balloons = self.detect_balloons_in_panel(panel_crop, panel['bbox'], crop_coords)
            all_balloons.extend(panel_balloons)

        # Assign balloons to panels
        assigned_balloons = self.assign_balloons_to_panels(panels, all_balloons)

        result = {
            'panels': panels,
            'balloons': assigned_balloons,
            'total_panels': len(panels),
            'total_balloons': len(assigned_balloons),
            'image_path': image_path
        }

        logger.info(f"Pipeline completed: {len(panels)} panels, {len(assigned_balloons)} balloons")
        return result

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save detection results to JSON"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

def main():
    """Main function for testing the pipeline"""
    logger.info("🚀 Starting Two-Detector Pipeline Test")

    # Model paths (adjust as needed)
    panel_model = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/anComicsViewer_v01.pt"
    balloon_model = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/anComicsViewer_v01.pt"  # Same model for now

    # Initialize pipeline
    pipeline = TwoDetectorPipeline(panel_model, balloon_model)

    # Test on a sample image
    test_image = "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/images/train/p0011.png"

    if os.path.exists(test_image):
        results = pipeline.run_pipeline(test_image)
        pipeline.save_results(results, "two_detector_results.json")

        print("\n📊 RESULTS:")
        print(f"   Panels detected: {results['total_panels']}")
        print(f"   Balloons detected: {results['total_balloons']}")
    else:
        logger.error(f"Test image not found: {test_image}")

if __name__ == "__main__":
    main()
