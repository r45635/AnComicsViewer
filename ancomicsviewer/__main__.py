"""Entry point for running AnComicsViewer as a module.

Usage:
    python -m ancomicsviewer
"""

import sys
import os

# Environment tweaks for macOS
if sys.platform == "darwin":
    if os.environ.get("QT_MAC_WANTS_LAYER") is None:
        os.environ["QT_MAC_WANTS_LAYER"] = "1"

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

from PySide6.QtWidgets import QApplication
from .main_window import ComicsView


def main() -> int:
    """Application entry point."""
    app = QApplication(sys.argv)
    window = ComicsView()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
