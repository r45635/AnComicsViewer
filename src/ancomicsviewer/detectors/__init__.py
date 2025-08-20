"""
Détecteurs de panels pour AnComicsViewer
=======================================

Collection d'algorithmes de détection de cases/panels pour bandes dessinées:

- Détecteur heuristique (OpenCV): Rapide, basé sur les contours et la morphologie
- Détecteur YOLO: Apprentissage automatique pour segmentation
- Détecteur Multi-BD Enhanced v2.0: Modèle optimisé multi-styles avec MPS

Classes principales:
- BasePanelDetector: Interface de base
- MultiBDPanelDetector: Détecteur ML optimisé (recommandé)
"""

from .base import BasePanelDetector
from .multibd_detector import MultiBDPanelDetector

# Import conditionnel des détecteurs ML
try:
    from .yolo_seg import YoloSegPanelDetector
    __all__ = ["BasePanelDetector", "MultiBDPanelDetector", "YoloSegPanelDetector"]
except ImportError:
    __all__ = ["BasePanelDetector", "MultiBDPanelDetector"]
