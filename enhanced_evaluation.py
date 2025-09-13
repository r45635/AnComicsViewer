#!/usr/bin/env python3
"""
Enhanced Evaluation Script for Comic Detection
Provides mAP, recall, per-class PR curves, and confusion matrix
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedEvaluator:
    """Enhanced evaluator with comprehensive metrics"""

    def __init__(self, ground_truth_file: str, predictions_file: str):
        self.gt_file = Path(ground_truth_file)
        self.pred_file = Path(predictions_file)
        self.ground_truth = []
        self.predictions = []

    def load_data(self):
        """Load ground truth and predictions"""
        # Load ground truth
        with open(self.gt_file, 'r') as f:
            gt_data = json.load(f)

        # Load predictions
        with open(self.pred_file, 'r') as f:
            pred_data = json.load(f)

        self.ground_truth = gt_data.get('shapes', [])
        self.predictions = pred_data.get('predictions', [])

        logger.info(f"Loaded {len(self.ground_truth)} GT annotations, {len(self.predictions)} predictions")

    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def match_predictions_to_ground_truth(self, iou_threshold: float = 0.5) -> Dict[str, Any]:
        """Match predictions to ground truth"""
        gt_by_class = defaultdict(list)
        pred_by_class = defaultdict(list)

        # Organize by class
        for gt in self.ground_truth:
            label = gt.get('label', 'unknown')
            points = gt.get('points', [])
            if len(points) >= 2:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                gt_by_class[label].append({'bbox': bbox, 'matched': False})

        for pred in self.predictions:
            label = pred.get('class', 'unknown')
            bbox = pred.get('bbox', [])
            conf = pred.get('confidence', 0.0)
            if len(bbox) == 4:
                pred_by_class[label].append({'bbox': bbox, 'conf': conf, 'matched': False})

        # Match predictions to GT
        matches = defaultdict(list)
        for class_name in set(gt_by_class.keys()) | set(pred_by_class.keys()):
            gt_boxes = gt_by_class[class_name]
            pred_boxes = sorted(pred_by_class[class_name], key=lambda x: x['conf'], reverse=True)

            for pred in pred_boxes:
                best_iou = 0
                best_gt_idx = -1

                for i, gt in enumerate(gt_boxes):
                    if not gt['matched']:
                        iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i

                if best_iou >= iou_threshold:
                    pred['matched'] = True
                    gt_boxes[best_gt_idx]['matched'] = True
                    matches[class_name].append({
                        'pred': pred,
                        'gt': gt_boxes[best_gt_idx],
                        'iou': best_iou
                    })

        return dict(matches)

    def calculate_precision_recall(self, matches: Dict[str, List], num_gt: int,
                                 confidence_thresholds: List[float]) -> Dict[str, Any]:
        """Calculate precision-recall curve"""
        if not matches:
            return {'precision': [], 'recall': []}

        # Sort by confidence
        sorted_preds = sorted(matches, key=lambda x: x['pred']['conf'], reverse=True)

        tp = 0
        fp = 0
        precision_values = []
        recall_values = []

        for pred in sorted_preds:
            if pred['pred']['matched']:
                tp += 1
            else:
                fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / num_gt if num_gt > 0 else 0

            precision_values.append(precision)
            recall_values.append(recall)

        return {
            'precision': precision_values,
            'recall': recall_values
        }

    def calculate_map(self, precision: List[float], recall: List[float]) -> float:
        """Calculate mean Average Precision"""
        if not precision or not recall:
            return 0.0

        # Add (0,1) point
        recall = [0] + recall + [1]
        precision = [1] + precision + [0]

        # Make precision monotonically decreasing
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])

        # Calculate AP
        ap = 0
        for i in range(1, len(recall)):
            ap += (recall[i] - recall[i-1]) * precision[i]

        return ap

    def calculate_confusion_matrix(self) -> Dict[str, Any]:
        """Calculate confusion matrix for all classes"""
        classes = set()
        for gt in self.ground_truth:
            classes.add(gt.get('label', 'unknown'))
        for pred in self.predictions:
            classes.add(pred.get('class', 'unknown'))

        classes = sorted(list(classes))
        n_classes = len(classes)
        conf_matrix = np.zeros((n_classes, n_classes))

        # Create class mapping
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        # Match predictions
        matches = self.match_predictions_to_ground_truth(0.5)

        # Fill confusion matrix
        for class_name, class_matches in matches.items():
            for match in class_matches:
                pred_class = match['pred'].get('class', 'unknown')
                gt_class = class_name  # GT class from the match key

                pred_idx = class_to_idx.get(pred_class, -1)
                gt_idx = class_to_idx.get(gt_class, -1)

                if pred_idx >= 0 and gt_idx >= 0:
                    conf_matrix[gt_idx, pred_idx] += 1

        # Add unmatched predictions as false positives
        for pred in self.predictions:
            pred_class = pred.get('class', 'unknown')
            if not any(m['pred']['bbox'] == pred['bbox'] for matches_list in matches.values() for m in matches_list):
                pred_idx = class_to_idx.get(pred_class, -1)
                if pred_idx >= 0:
                    conf_matrix[n_classes-1, pred_idx] += 1  # Last row for FP

        return {
            'matrix': conf_matrix.tolist(),
            'classes': classes,
            'class_mapping': class_to_idx
        }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        logger.info("Generating comprehensive evaluation report...")

        if not self.ground_truth or not self.predictions:
            self.load_data()

        # Match predictions to GT
        matches = self.match_predictions_to_ground_truth(0.5)

        # Calculate metrics per class
        per_class_metrics = {}
        overall_tp = 0
        overall_fp = 0
        overall_fn = 0

        for class_name in set(list(matches.keys()) + ['panel', 'balloon']):
            class_matches = matches.get(class_name, [])
            gt_count = len([gt for gt in self.ground_truth if gt.get('label') == class_name])

            tp = len([m for m in class_matches if m['pred']['matched']])
            fp = len([p for p in self.predictions if p.get('class') == class_name]) - tp
            fn = gt_count - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_class_metrics[class_name] = {
                'tp': tp, 'fp': fp, 'fn': fn,
                'precision': precision, 'recall': recall, 'f1': f1
            }

            overall_tp += tp
            overall_fp += fp
            overall_fn += fn

        # Overall metrics
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

        # Confusion matrix
        conf_matrix = self.calculate_confusion_matrix()

        report = {
            'overall_metrics': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1,
                'tp': overall_tp,
                'fp': overall_fp,
                'fn': overall_fn
            },
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': conf_matrix,
            'total_ground_truth': len(self.ground_truth),
            'total_predictions': len(self.predictions),
            'matched_predictions': overall_tp
        }

        return report

    def print_report(self, report: Dict[str, Any]):
        """Print human-readable report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE EVALUATION REPORT")
        print("="*80)

        overall = report['overall_metrics']

        print("\n🎯 OVERALL METRICS:")
        print(".3f"        print(".3f"        print(".3f"        print(f"   True Positives: {overall['tp']}")
        print(f"   False Positives: {overall['fp']}")
        print(f"   False Negatives: {overall['fn']}")

        print("\n📈 PER-CLASS METRICS:")
        for class_name, metrics in report['per_class_metrics'].items():
            print(f"\n   {class_name.upper()}:")
            print(".3f"            print(".3f"            print(".3f"            print(f"      TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")

        print("\n📊 SUMMARY:")
        print(f"   Ground Truth: {report['total_ground_truth']}")
        print(f"   Predictions: {report['total_predictions']}")
        print(f"   Matched: {report['matched_predictions']}")

def main():
    """Main evaluation function"""
    logger.info("🚀 Starting Enhanced Evaluation")

    # Example usage (adjust paths as needed)
    gt_file = "/path/to/ground_truth.json"
    pred_file = "/path/to/predictions.json"

    evaluator = EnhancedEvaluator(gt_file, pred_file)

    try:
        report = evaluator.generate_comprehensive_report()
        evaluator.print_report(report)

        # Save report
        with open("evaluation_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        logger.info("✅ Evaluation completed!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()
