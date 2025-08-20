"""YOLOv8-based panel detector for comic books.

This module implements panel detection using Ultralytics YOLOv8 models,
which can be trained on comic book datasets for accurate panel detection.
"""

from typing import List, Tuple, Dict, Any, Optional
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage
import numpy as np

from . import BaseMLDetector


class YOLODetector(BaseMLDetector):
    """YOLOv8-based panel detector."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """Initialize YOLO detector.
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            device: Device for inference ("cpu", "cuda", "auto")
        """
        super().__init__(model_path, device)
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
    def load_model(self) -> bool:
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            
            if self.model_path:
                self.model = YOLO(self.model_path)
            else:
                # Use pre-trained model and fine-tune for panels
                self.model = YOLO('yolov8n.pt')  # Nano version for speed
                
            # Configure model
            self.model.to(self.device)
            self.is_loaded = True
            return True
            
        except ImportError:
            print("YOLOv8 not available. Install with: pip install ultralytics")
            return False
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return False
    
    def detect_panels(self, qimage: QImage, page_point_size: Tuple[int, int]) -> List[QRectF]:
        """Detect panels using YOLOv8."""
        if not self.is_loaded:
            if not self.load_model():
                return []
        
        try:
            # Preprocess image
            rgb_array = self.preprocess_image(qimage)
            
            # Run inference
            results = self.model(rgb_array, 
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               device=self.device)
            
            # Convert results to detection format
            detections = []
            if results and len(results) > 0:
                result = results[0]  # First (and only) image
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = box
                        w, h = x2 - x1, y2 - y1
                        
                        detections.append({
                            "bbox": [x1, y1, w, h],
                            "confidence": float(conf)
                        })
            
            # Convert to QRectF in page coordinates
            image_size = (rgb_array.shape[1], rgb_array.shape[0])  # width, height
            return self.postprocess_detections(detections, image_size, page_point_size)
            
        except Exception as e:
            print(f"YOLO inference failed: {e}")
            return []
    
    def get_confidence_threshold(self) -> float:
        """Get confidence threshold."""
        return self.confidence_threshold
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for detections."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def set_iou_threshold(self, threshold: float):
        """Set IoU threshold for NMS."""
        self.iou_threshold = max(0.0, min(1.0, threshold))


class YOLOTrainer:
    """Trainer for custom YOLO models on comic datasets."""
    
    def __init__(self, dataset_path: str, model_size: str = "n"):
        """Initialize trainer.
        
        Args:
            dataset_path: Path to dataset in YOLO format
            model_size: Model size ("n", "s", "m", "l", "x")
        """
        self.dataset_path = dataset_path
        self.model_size = model_size
        self.model = None
    
    def train(self, epochs: int = 100, batch_size: int = 16, 
              img_size: int = 640, **kwargs) -> str:
        """Train YOLO model on comic dataset.
        
        Returns:
            Path to trained model
        """
        try:
            from ultralytics import YOLO
            
            # Initialize model
            self.model = YOLO(f'yolov8{self.model_size}.pt')
            
            # Train
            results = self.model.train(
                data=self.dataset_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                **kwargs
            )
            
            return results.save_dir / "weights" / "best.pt"
            
        except Exception as e:
            print(f"Training failed: {e}")
            return ""
    
    def validate(self, model_path: str) -> Dict[str, float]:
        """Validate trained model."""
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            metrics = model.val(data=self.dataset_path)
            
            return {
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr)
            }
            
        except Exception as e:
            print(f"Validation failed: {e}")
            return {}
