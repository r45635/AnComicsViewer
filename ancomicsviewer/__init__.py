"""AnComicsViewer - PDF comics reader with heuristic panel detection.

A modular, performant comics/manga viewer combining PySide6 and OpenCV
for intelligent panel detection and navigation.
"""

__version__ = "2.0.0"
__author__ = "AnComicsViewer Contributors"

from .config import DetectorConfig, AppConfig
from .detector import PanelDetector
from .main_window import ComicsView

__all__ = [
    "DetectorConfig",
    "AppConfig",
    "PanelDetector",
    "ComicsView",
]
