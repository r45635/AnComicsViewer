"""Page style classifier for panel detection.

Classifies pages as:
- classic_franco_belge: Traditional European comics with clear black gutters
- modern: Watercolor, tinted backgrounds, complex layouts

Uses a combination of heuristics and a simple feature-based classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import pickle
import os

from PySide6.QtCore import QRectF

from .utils import pdebug, HAS_CV2, HAS_NUMPY

if HAS_CV2:
    import cv2
if HAS_NUMPY:
    import numpy as np
    from numpy.typing import NDArray


@dataclass
class PageFeatures:
    """Features extracted from a page for classification."""
    rect_count: int
    small_ratio: float      # Ratio of small rects
    margin_ratio: float     # Ratio of rects in margins
    gutter_count: int       # Number of detected gutters
    bg_tint_dist: float     # Distance from pure white background
    coverage: float         # Page coverage by detected rects
    aspect_variance: float  # Variance in aspect ratios
    
    def to_vector(self) -> List[float]:
        """Convert to feature vector for ML classifier."""
        return [
            self.rect_count / 12.0,  # Normalize to typical max
            self.small_ratio,
            self.margin_ratio,
            self.gutter_count / 10.0,  # Normalize
            min(self.bg_tint_dist / 50.0, 1.0),  # Cap at 50
            self.coverage,
            min(self.aspect_variance, 1.0),  # Cap variance
        ]


class PageStyleClassifier:
    """Classifier for determining page style (classic vs modern).
    
    Uses a combination of:
    1. Rule-based heuristics for clear cases
    2. Simple feature scoring for ambiguous cases
    """
    
    # Thresholds for classic detection
    CLASSIC_MIN_GUTTERS = 3
    CLASSIC_MAX_SMALL_RATIO = 0.30
    CLASSIC_MIN_RECT_COUNT = 3
    CLASSIC_MAX_RECT_COUNT = 12
    CLASSIC_MAX_BG_TINT = 18.0
    CLASSIC_MIN_COVERAGE = 0.50
    
    def __init__(self):
        """Initialize classifier."""
        self._model = None
        self._model_path = os.path.join(
            os.path.dirname(__file__), 
            "classifier_model.pkl"
        )
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model if available."""
        if os.path.exists(self._model_path):
            try:
                with open(self._model_path, "rb") as f:
                    self._model = pickle.load(f)
                pdebug(f"[Classifier] Loaded ML model from {self._model_path}")
            except Exception as e:
                pdebug(f"[Classifier] Failed to load model: {e}")
                self._model = None
    
    def extract_features(
        self, 
        img_bgr: NDArray, 
        rects: List[QRectF], 
        w: int, 
        h: int
    ) -> PageFeatures:
        """Extract features from page for classification.
        
        Args:
            img_bgr: Image in BGR format
            rects: Detected rectangles in page points
            w, h: Image dimensions
            
        Returns:
            PageFeatures object
        """
        page_area = w * h
        rect_count = len(rects)
        
        # Small rect ratio
        small_count = sum(1 for r in rects 
                         if (r.width() * r.height()) < 0.01 * page_area)
        small_ratio = small_count / rect_count if rect_count > 0 else 0.0
        
        # Margin rect ratio
        margin_count = 0
        for rect in rects:
            in_top = rect.top() < 0.06 * h
            in_bottom = (rect.top() + rect.height()) > 0.94 * h
            if (in_top or in_bottom) and rect.height() < 0.12 * h:
                margin_count += 1
        margin_ratio = margin_count / rect_count if rect_count > 0 else 0.0
        
        # Gutter count (quick estimation)
        gutter_count = self._estimate_gutter_count(img_bgr, w, h)
        
        # Background tint distance
        bg_tint_dist = self._compute_bg_tint_distance(img_bgr)
        
        # Coverage
        coverage = self._compute_coverage(rects, w, h)
        
        # Aspect ratio variance
        aspect_variance = self._compute_aspect_variance(rects)
        
        return PageFeatures(
            rect_count=rect_count,
            small_ratio=small_ratio,
            margin_ratio=margin_ratio,
            gutter_count=gutter_count,
            bg_tint_dist=bg_tint_dist,
            coverage=coverage,
            aspect_variance=aspect_variance,
        )
    
    def _estimate_gutter_count(self, img_bgr: NDArray, w: int, h: int) -> int:
        """Quick estimation of gutter count using projections."""
        if not HAS_CV2 or not HAS_NUMPY:
            return 0
        
        # Downscale for speed
        scale = 0.5
        img_small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        # Projections
        h_proj = np.mean(gray_small, axis=1)
        v_proj = np.mean(gray_small, axis=0)
        
        # Find bright stripes (gutters)
        h_threshold = np.percentile(h_proj, 85)
        v_threshold = np.percentile(v_proj, 85)
        
        h_gutters = np.where(h_proj > h_threshold)[0]
        v_gutters = np.where(v_proj > v_threshold)[0]
        
        # Count runs
        def count_runs(arr):
            if len(arr) == 0:
                return 0
            runs = 1
            for i in range(1, len(arr)):
                if arr[i] - arr[i-1] > 3:
                    runs += 1
            return runs
        
        return count_runs(h_gutters) + count_runs(v_gutters)
    
    def _compute_bg_tint_distance(self, img_bgr: NDArray) -> float:
        """Compute distance of background from pure white."""
        if not HAS_CV2 or not HAS_NUMPY:
            return 0.0
        
        # Sample borders
        border = 5
        h, w = img_bgr.shape[:2]
        
        border_pixels = np.vstack([
            img_bgr[:border, :].reshape(-1, 3),
            img_bgr[-border:, :].reshape(-1, 3),
            img_bgr[:, :border].reshape(-1, 3),
            img_bgr[:, -border:].reshape(-1, 3),
        ])
        
        # Convert to Lab
        border_lab = cv2.cvtColor(
            border_pixels.reshape(1, -1, 3).astype(np.uint8), 
            cv2.COLOR_BGR2Lab
        ).reshape(-1, 3).astype(np.float32)
        
        # Mean background
        bg_lab = np.mean(border_lab, axis=0)
        
        # Distance from pure white (Lab: [100, 0, 0])
        white_lab = np.array([100.0, 128.0, 128.0])  # OpenCV Lab ranges
        return float(np.linalg.norm(bg_lab - white_lab))
    
    def _compute_coverage(self, rects: List[QRectF], w: int, h: int) -> float:
        """Compute page coverage by rectangles."""
        if not rects or not HAS_NUMPY:
            return 0.0
        
        page_area = w * h
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for rect in rects:
            x = int(max(0, min(rect.left(), w - 1)))
            y = int(max(0, min(rect.top(), h - 1)))
            x2 = int(max(0, min(rect.left() + rect.width(), w)))
            y2 = int(max(0, min(rect.top() + rect.height(), h)))
            
            if x2 > x and y2 > y:
                mask[y:y2, x:x2] = 1
        
        return float(np.sum(mask)) / page_area if page_area > 0 else 0.0
    
    def _compute_aspect_variance(self, rects: List[QRectF]) -> float:
        """Compute variance in aspect ratios."""
        if len(rects) < 2 or not HAS_NUMPY:
            return 0.0
        
        aspects = []
        for r in rects:
            if r.height() > 0:
                aspects.append(r.width() / r.height())
        
        if len(aspects) < 2:
            return 0.0
        
        return float(np.var(aspects))
    
    def classify(
        self, 
        img_bgr: NDArray, 
        rects: List[QRectF], 
        w: int, 
        h: int,
        debug: bool = False
    ) -> str:
        """Classify page style.
        
        Args:
            img_bgr: Image in BGR format
            rects: Detected rectangles
            w, h: Image dimensions
            debug: Enable debug logging
            
        Returns:
            "classic_franco_belge" or "modern"
        """
        features = self.extract_features(img_bgr, rects, w, h)
        
        # Use ML model if available
        if self._model is not None:
            try:
                vec = np.array([features.to_vector()])
                pred = self._model.predict(vec)[0]
                if debug:
                    pdebug(f"[Classifier] ML prediction: {pred}")
                return pred
            except Exception as e:
                if debug:
                    pdebug(f"[Classifier] ML error: {e}, falling back to rules")
        
        # Rule-based classification
        return self._classify_by_rules(features, debug)
    
    def _classify_by_rules(self, features: PageFeatures, debug: bool) -> str:
        """Rule-based classification for classic vs modern."""
        # Score-based approach
        classic_score = 0.0
        modern_score = 0.0
        
        # Gutter count
        if features.gutter_count >= self.CLASSIC_MIN_GUTTERS:
            classic_score += 2.0
        elif features.gutter_count <= 2:
            modern_score += 1.5
        
        # Small ratio
        if features.small_ratio < self.CLASSIC_MAX_SMALL_RATIO:
            classic_score += 1.0
        else:
            modern_score += 1.5
        
        # Rect count
        if self.CLASSIC_MIN_RECT_COUNT <= features.rect_count <= self.CLASSIC_MAX_RECT_COUNT:
            classic_score += 1.0
        elif features.rect_count > self.CLASSIC_MAX_RECT_COUNT:
            modern_score += 0.5
        
        # Background tint
        if features.bg_tint_dist < self.CLASSIC_MAX_BG_TINT:
            classic_score += 1.5
        else:
            modern_score += 2.0
        
        # Coverage
        if features.coverage >= self.CLASSIC_MIN_COVERAGE:
            classic_score += 0.5
        
        # Aspect variance (high variance suggests dynamic layouts)
        if features.aspect_variance > 0.5:
            modern_score += 0.5
        
        result = "classic_franco_belge" if classic_score >= modern_score else "modern"
        
        if debug:
            pdebug(f"[Classifier] Features: rect_count={features.rect_count}, "
                  f"small_ratio={features.small_ratio:.3f}, "
                  f"gutter_count={features.gutter_count}, "
                  f"bg_tint_dist={features.bg_tint_dist:.1f}, "
                  f"coverage={features.coverage:.3f}")
            pdebug(f"[Classifier] Scores: classic={classic_score:.1f}, modern={modern_score:.1f} => {result}")
        
        return result


# Global instance for reuse
_classifier: Optional[PageStyleClassifier] = None


def get_classifier() -> PageStyleClassifier:
    """Get or create global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = PageStyleClassifier()
    return _classifier


def classify_page_style(
    img_bgr: NDArray, 
    rects: List[QRectF], 
    w: int, 
    h: int,
    debug: bool = False
) -> str:
    """Convenience function to classify page style.
    
    Args:
        img_bgr: Image in BGR format
        rects: Detected rectangles
        w, h: Image dimensions
        debug: Enable debug logging
        
    Returns:
        "classic_franco_belge" or "modern"
    """
    return get_classifier().classify(img_bgr, rects, w, h, debug)
