"""
Utilitaires AnComicsViewer
=========================

Modules utilitaires pour le cache, les conversions d'images,
et autres fonctions d'aide.
"""

# Import conditionnel du cache amélioré
try:
    from .enhanced_cache import PanelCacheManager
    __all__ = ["PanelCacheManager"]
except ImportError:
    __all__ = []
