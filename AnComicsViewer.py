#!/usr/bin/env python3
"""AnComicsViewer — PDF comics reader with heuristic panel detection.

This is the main entry point that provides backward compatibility.
The application has been refactored into a modular package structure
for better maintainability and performance.

Usage:
    python AnComicsViewer.py          # Run the application
    python -m ancomicsviewer          # Alternative: run as module

For development/testing, the modular package in ancomicsviewer/ provides:
    - ancomicsviewer.config: Configuration dataclasses
    - ancomicsviewer.detector: Panel detection engine
    - ancomicsviewer.pdf_view: Custom PDF view widget
    - ancomicsviewer.dialogs: Tuning dialogs
    - ancomicsviewer.main_window: Main application window
    - ancomicsviewer.cache: LRU caching for panel results

See README.md for installation and usage instructions.
"""

import sys
import os

# Environment tweaks for macOS and Qt
if sys.platform == "darwin":
    if os.environ.get("QT_MAC_WANTS_LAYER") is None:
        os.environ["QT_MAC_WANTS_LAYER"] = "1"

import warnings
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

# Add package to path if running as script
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Import from the modular package
from ancomicsviewer import ComicsView, PanelDetector, DetectorConfig, AppConfig
from ancomicsviewer.image_utils import pdebug, qimage_to_numpy_rgba as qimage_to_numpy_rgba8888
from ancomicsviewer.pdf_view import PannablePdfView
from ancomicsviewer.dialogs import PanelTuningDialog
from ancomicsviewer.config import PRESETS

from PySide6.QtWidgets import QApplication


def main() -> int:
    """Application entry point."""
    app = QApplication(sys.argv)
    window = ComicsView()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
