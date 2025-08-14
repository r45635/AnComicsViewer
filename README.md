# AnComicsViewer

A lightweight PDF/manga/comics reader with heuristic panel detection and an interactive tuning dialog.

This repository contains a single-file PySide6 + OpenCV-based viewer (`AnComicsViewer.py`) that
detects rectangular panels on comic book pages using a heuristic pipeline. It includes an interactive
Panel Tuning dialog so you can adjust detection parameters and immediately re-run detection.

Features
- Heuristic panel detection using adaptive thresholding, morphological cleanup, light/gutter splitting,
  and recursive projection splitting.
- Interactive `Panel Tuning` dialog exposing DPI-stable fractional thresholds, projection smoothing,
  and title-row heuristics (two-scenario rule: many small OR a few big title boxes).
- Runtime parameter snapshot logging (printed to stdout) to help reproduce detection settings.

Quick start (recommended: use the included virtualenv)

1) Create and activate a virtual environment (macOS / zsh):

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install runtime dependencies:

```bash
pip install -r requirements.txt
```

3) Run the app (macOS note below):

```bash
# macOS: ensure Qt can find the cocoa platform plugin
QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import PySide6, os; print(os.path.join(os.path.dirname(PySide6.__file__), 'Qt', 'plugins', 'platforms'))") \
  .venv/bin/python AnComicsViewer.py
```

What to expect
- The app prints parameter snapshots and per-page detection debug messages to the terminal. When you
  open the `Panel Tuning` dialog, change values, and click `Apply`, the app clears the panel cache,
  re-runs detection at the page DPI, and prints a new parameter snapshot followed by detection logs.

Notes and troubleshooting
- If Qt complains about the `cocoa` plugin on macOS, set `QT_QPA_PLATFORM_PLUGIN_PATH` as shown above.
- If you have issues with the most recent PySide6 builds, a known working version in this repo's
  environment is `PySide6==6.8.3` (installed in the development venv I tested with). You can adjust the
  pinned version in `requirements.txt` if necessary.

Developer tips
- The main detection logic and tunables live in `AnComicsViewer.py` within the `PanelDetector` class.
- The `PanelTuningDialog` writes tunable values back to the detector; `_apply_panel_tuning` now clears
  the panel cache and forces re-detection.

License
- This repo contains example code; add a license file if you want to make terms explicit.

If you want, I can also add a small smoke-test script that programmatically runs detection on a bundled
sample page and writes a JSON snapshot of the detected panels for CI or regression testing.
# AnComicsViewer

Minimal standalone PDF comics reader (PySide6) with a heuristic panel detector (OpenCV).

This README explains how to create a fresh virtualenv, install runtime dependencies, and run the app on macOS (zsh), including the small env-var fix required when Qt cannot find the `cocoa` platform plugin.

## Prerequisites
- Python 3.11+ (recommended). Use the system Python or Homebrew Python on macOS.
- zsh (default on modern macOS)

## Fresh install (recommended)
Open a terminal in the repository root and run:

```bash
# create a venv and activate it
python3 -m venv .venv
source .venv/bin/activate

# upgrade pip and install deps
pip install --upgrade pip
pip install -r requirements.txt
```

## Run (macOS / zsh)

On macOS, Qt sometimes cannot find the `cocoa` platform plugin when using a venv. Use the following small env-var command to locate PySide6's plugins and run the app in one line (recommended):

```bash
export QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import PySide6, os; print(os.path.join(os.path.dirname(PySide6.__file__), 'Qt', 'plugins', 'platforms'))")
.venv/bin/python AnComicsViewer.py
```

Or as a single inline prefix (no export):

```bash
QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import PySide6, os; print(os.path.join(os.path.dirname(PySide6.__file__), 'Qt', 'plugins', 'platforms'))") .venv/bin/python AnComicsViewer.py
```

If the first command prints an empty path or errors, ensure PySide6 is installed in the active environment (`pip install PySide6`) and retry.

## Quick troubleshooting
- Error "Could not find the Qt platform plugin \"cocoa\"": confirm the env-var above points to a directory that contains files like `libqcocoa.dylib` (or similar). If not present, reinstall PySide6 in the venv: `pip install --force-reinstall PySide6`.
- If OpenCV wheels fail to install via pip on your macOS setup, try installing a binary wheel (arm64/x86) via pip from PyPI or install OpenCV system-wide and use the `opencv-python` wheel.

## Notes
- The app uses PySide6, numpy and OpenCV for panel detection. `requirements.txt` pins reasonable defaults but you can adjust versions if needed.
- To run without a venv, adapt the `QT_QPA_PLATFORM_PLUGIN_PATH` resolution to the Python you will use.

If you want, I can try launching the app here (I will set the env var and run it to capture logs), or I can add a short helper script to automate the env-var detection and launch.
# AnComicsViewer

AnComicsViewer is a simple Python-based application for viewing comics. This project is in its initial setup phase.

## Features
- Comic viewing (to be implemented)

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/r45635/AnComicsViewer.git
   ```
2. Run the main script:
   ```bash
   python AnComicsViewer.py
   ```

## Requirements
- Python 3.7+

## License
See [LICENSE](LICENSE) for details.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Author
- Vincent Cruvellier
