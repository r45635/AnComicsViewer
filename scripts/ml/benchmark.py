"""Benchmark and evaluation utilities for ML panel detectors.

This module provides tools to compare ML-based detectors with the heuristic approach
and evaluate detection performance on various metrics.
"""

import json
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage

try:
    from .models import BaseMLDetector, YOLODetector, HybridDetector
except ImportError:
    # Handle case where ML dependencies aren't installed
    BaseMLDetector = None
    YOLODetector = None
    HybridDetector = None


class PanelDetectionBenchmark:
    """Benchmark suite for panel detection methods."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def run_detection_benchmark(self, 
                               detectors: Dict[str, Any], 
                               test_images: List[Tuple[QImage, Tuple[int, int]]],
                               ground_truth: Optional[List[List[QRectF]]] = None) -> Dict[str, Any]:
        """Run detection benchmark on multiple detectors.
        
        Args:
            detectors: Dict of detector_name -> detector_instance
            test_images: List of (qimage, page_size) tuples
            ground_truth: Optional ground truth annotations
            
        Returns:
            Benchmark results dictionary
        """
        results = {}
        
        for name, detector in detectors.items():
            print(f"Benchmarking {name}...")
            
            detector_results = {
                "total_time": 0.0,
                "avg_time_per_image": 0.0,
                "total_panels_detected": 0,
                "avg_panels_per_image": 0.0,
                "success_rate": 0.0,
                "detections": []
            }
            
            successful_detections = 0
            
            for i, (qimage, page_size) in enumerate(test_images):
                start_time = time.time()
                
                try:
                    panels = detector.detect_panels(qimage, page_size)
                    end_time = time.time()
                    
                    detection_time = end_time - start_time
                    detector_results["total_time"] += detection_time
                    detector_results["total_panels_detected"] += len(panels)
                    
                    detector_results["detections"].append({
                        "image_index": i,
                        "num_panels": len(panels),
                        "detection_time": detection_time,
                        "panels": [self._qrectf_to_dict(rect) for rect in panels]
                    })
                    
                    successful_detections += 1
                    
                except Exception as e:
                    print(f"Detection failed for {name} on image {i}: {e}")
                    detector_results["detections"].append({
                        "image_index": i,
                        "error": str(e),
                        "detection_time": 0.0,
                        "num_panels": 0
                    })
            
            # Calculate averages
            num_images = len(test_images)
            detector_results["avg_time_per_image"] = (
                detector_results["total_time"] / num_images if num_images > 0 else 0.0
            )
            detector_results["avg_panels_per_image"] = (
                detector_results["total_panels_detected"] / num_images if num_images > 0 else 0.0
            )
            detector_results["success_rate"] = (
                successful_detections / num_images if num_images > 0 else 0.0
            )
            
            results[name] = detector_results
        
        # Save results
        self._save_results("detection_benchmark", results)
        return results
    
    def calculate_accuracy_metrics(self, 
                                 predicted: List[QRectF], 
                                 ground_truth: List[QRectF],
                                 iou_threshold: float = 0.5) -> Dict[str, float]:
        """Calculate accuracy metrics for panel detection.
        
        Args:
            predicted: Predicted panel rectangles
            ground_truth: Ground truth panel rectangles
            iou_threshold: IoU threshold for considering a match
            
        Returns:
            Dictionary with precision, recall, F1 metrics
        """
        if not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        if not predicted:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Calculate IoU matrix
        matches = 0
        matched_gt = set()
        
        for pred_rect in predicted:
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_rect in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue
                    
                iou = self._calculate_iou(pred_rect, gt_rect)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                matches += 1
                matched_gt.add(best_gt_idx)
        
        # Calculate metrics
        precision = matches / len(predicted) if predicted else 0.0
        recall = matches / len(ground_truth) if ground_truth else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matches": matches,
            "predicted_count": len(predicted),
            "ground_truth_count": len(ground_truth)
        }
    
    def _calculate_iou(self, rect1: QRectF, rect2: QRectF) -> float:
        """Calculate Intersection over Union for two rectangles."""
        # Get intersection rectangle
        x1 = max(rect1.x(), rect2.x())
        y1 = max(rect1.y(), rect2.y())
        x2 = min(rect1.x() + rect1.width(), rect2.x() + rect2.width())
        y2 = min(rect1.y() + rect1.height(), rect2.y() + rect2.height())
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate areas
        intersection = (x2 - x1) * (y2 - y1)
        area1 = rect1.width() * rect1.height()
        area2 = rect2.width() * rect2.height()
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _qrectf_to_dict(self, rect: QRectF) -> Dict[str, float]:
        """Convert QRectF to dictionary."""
        return {
            "x": rect.x(),
            "y": rect.y(),
            "width": rect.width(),
            "height": rect.height()
        }
    
    def _save_results(self, benchmark_name: str, results: Dict[str, Any]):
        """Save benchmark results to JSON file."""
        output_file = self.output_dir / f"{benchmark_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {output_file}")


def run_quick_benchmark():
    """Run a quick benchmark with synthetic test data."""
    print("Running quick ML detection benchmark...")
    
    # Check if ML dependencies are available
    if BaseMLDetector is None:
        print("ML dependencies not installed. Install with:")
        print("pip install -r requirements-ml.txt")
        return
    
    # Import the heuristic detector
    try:
        from AnComicsViewer import PanelDetector
        heuristic_detector = PanelDetector(debug=True)
    except ImportError:
        print("Could not import heuristic detector")
        return
    
    # Create synthetic test images (you would replace this with real data)
    from AnComicsViewer import make_test_qimage  # From smoke_test.py
    
    test_images = [
        (make_test_qimage(800, 1200), (800, 1200)),
        (make_test_qimage(600, 900), (600, 900)),
        (make_test_qimage(1000, 1400), (1000, 1400))
    ]
    
    # Setup detectors to test
    detectors = {
        "heuristic": heuristic_detector
    }
    
    # Add YOLO detector if available
    if YOLODetector is not None:
        try:
            yolo_detector = YOLODetector()
            if yolo_detector.load_model():
                detectors["yolo"] = yolo_detector
        except Exception as e:
            print(f"Could not load YOLO detector: {e}")
    
    # Run benchmark
    benchmark = PanelDetectionBenchmark()
    results = benchmark.run_detection_benchmark(detectors, test_images)
    
    # Print summary
    print("\n=== Benchmark Results ===")
    for detector_name, result in results.items():
        print(f"\n{detector_name.upper()}:")
        print(f"  Average time per image: {result['avg_time_per_image']:.3f}s")
        print(f"  Average panels per image: {result['avg_panels_per_image']:.1f}")
        print(f"  Success rate: {result['success_rate']:.1%}")


if __name__ == "__main__":
    run_quick_benchmark()
