"""Panel detection engine for AnComicsViewer.

Optimized heuristic-based comic panel detection using OpenCV.

This package provides a modular architecture for panel detection:
- base.py: Core PanelDetector class with main detection flow
- adaptive.py: Adaptive threshold detection route
- gutter.py: Gutter-based detection for white/light separations
- freeform.py: Watershed segmentation for complex layouts
- filters.py: Post-processing filters (title rows, empty panels, etc.)
- classifier.py: Page style classification (classic vs modern)
- utils.py: Shared utilities and data structures
"""

from __future__ import annotations

# Re-export main classes for backward compatibility
from .base import PanelDetector
from .utils import PanelRegion, DebugInfo

__all__ = ["PanelDetector", "PanelRegion", "DebugInfo"]
