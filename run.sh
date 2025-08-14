#!/usr/bin/env bash
# Small helper to run the app with the correct Qt plugin path on macOS.
PY=$(python -c "import sys; print(sys.executable)")
PLUGS=$(python -c "import PySide6, os; print(os.path.join(os.path.dirname(PySide6.__file__), 'Qt', 'plugins', 'platforms'))")

QT_QPA_PLATFORM_PLUGIN_PATH="$PLUGS" "$PY" AnComicsViewer.py
