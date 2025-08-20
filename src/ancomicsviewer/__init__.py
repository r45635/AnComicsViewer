"""
AnComicsViewer - Lecteur PDF intelligent pour bandes dessinées
============================================================

Un lecteur PDF moderne avec détection intelligente de cases utilisant
des algorithmes heuristiques et l'apprentissage automatique.

Modules principaux:
- detectors: Algorithmes de détection de panels (heuristique, YOLO, Multi-BD)
- ui: Interface utilisateur PySide6
- utils: Utilitaires et cache amélioré

Version: 2.0.0+
Auteur: Vincent Cruvellier
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Vincent Cruvellier"
__license__ = "MIT"

# Imports principaux pour l'API publique
from .main_app import main

__all__ = ["main", "__version__", "__author__", "__license__"]
