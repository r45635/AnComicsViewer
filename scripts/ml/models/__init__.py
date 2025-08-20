"""Base class for ML-based panel detectors.

This module defines the interface that all ML panel detection models should implement
to integrate seamlessly with the existing AnComicsViewer application.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage
import numpy as np


class BaseMLDetector(ABC):
    """Base class for ML-based panel detection models."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """Initialize the ML detector.
        
        Args:
            model_path: Path to the trained model file
            device: Device to run inference on ("cpu", "cuda", "auto")
        """
        self.model_path = model_path
        self.device = self._select_device(device)
        self.model = None
        self.is_loaded = False
        
        # Compatibility with heuristic detector interface
        self.debug = False
        self.last_debug = {"v": [], "h": []}
        
    def _select_device(self, device: str) -> str:
        """Select the best available device."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the ML model. Returns True if successful."""
        pass
    
    @abstractmethod
    def detect_panels(self, qimage: QImage, page_point_size: Tuple[int, int]) -> List[QRectF]:
        """Detect panels in the given image.
        
        Args:
            qimage: Input image as QImage
            page_point_size: Page size in points (width, height)
            
        Returns:
            List of detected panel rectangles in page coordinate system
        """
        pass
    
    def preprocess_image(self, qimage: QImage) -> np.ndarray:
        """Convert QImage to numpy array for ML processing."""
        from AnComicsViewer import qimage_to_numpy_rgba8888
        
        arr = qimage_to_numpy_rgba8888(qimage)
        if arr is None:
            raise ValueError("Failed to convert QImage to numpy array")
        
        # Convert RGBA to RGB
        rgb = arr[:, :, :3]
        return rgb
    
    def postprocess_detections(self, detections: List[Dict[str, Any]], 
                             image_size: Tuple[int, int],
                             page_point_size: Tuple[int, int]) -> List[QRectF]:
        """Convert ML detection results to QRectF in page coordinates.
        
        Args:
            detections: Raw ML model detections
            image_size: Original image size (width, height)
            page_point_size: Page size in points
            
        Returns:
            List of QRectF objects in page coordinate system
        """
        rects = []
        img_w, img_h = image_size
        page_w, page_h = page_point_size
        
        for det in detections:
            # Assume detection format: {"bbox": [x, y, w, h], "confidence": float}
            bbox = det.get("bbox", [])
            confidence = det.get("confidence", 0.0)
            
            if len(bbox) != 4 or confidence < self.get_confidence_threshold():
                continue
                
            x, y, w, h = bbox
            
            # Convert from image pixels to page points
            x_pt = (x / img_w) * page_w
            y_pt = (y / img_h) * page_h
            w_pt = (w / img_w) * page_w
            h_pt = (h / img_h) * page_h
            
            rects.append(QRectF(x_pt, y_pt, w_pt, h_pt))
            
        return rects
    
    def get_confidence_threshold(self) -> float:
        """Get the minimum confidence threshold for detections."""
        return 0.5
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "model_type": self.__class__.__name__
        }


class HybridDetector(BaseMLDetector):
    """Hybrid detector that combines ML and heuristic methods."""
    
    def __init__(self, ml_detector: BaseMLDetector, heuristic_detector, 
                 ml_weight: float = 0.7):
        """Initialize hybrid detector.
        
        Args:
            ml_detector: ML-based detector instance
            heuristic_detector: Heuristic detector instance
            ml_weight: Weight for ML results (0.0 = only heuristic, 1.0 = only ML)
        """
        super().__init__()
        self.ml_detector = ml_detector
        self.heuristic_detector = heuristic_detector
        self.ml_weight = ml_weight
        
    def load_model(self) -> bool:
        """Load the ML model."""
        return self.ml_detector.load_model()
    
    def detect_panels(self, qimage: QImage, page_point_size: Tuple[int, int]) -> List[QRectF]:
        """Detect panels using both ML and heuristic methods."""
        ml_rects = []
        heuristic_rects = []
        
        # Get ML detections
        if self.ml_detector.is_loaded:
            try:
                ml_rects = self.ml_detector.detect_panels(qimage, page_point_size)
            except Exception as e:
                print(f"ML detection failed: {e}")
        
        # Get heuristic detections
        try:
            heuristic_rects = self.heuristic_detector.detect_panels(qimage, page_point_size)
        except Exception as e:
            print(f"Heuristic detection failed: {e}")
        
        # Combine results based on weight
        if self.ml_weight >= 0.5 and ml_rects:
            return ml_rects
        elif heuristic_rects:
            return heuristic_rects
        else:
            return ml_rects  # Fallback to ML even if weight is low
