<div align="center">

![AnComicsViewer Logo](logo.png)

# AnComicsViewer

**A lightweight PDF/manga/comics reader with intelligent panel detection**

[![Windows CI](https://github.com/r45635/AnComicsViewer/actions/workflows/windows-smoke.yml/badge.svg)](https://github.com/r45635/AnComicsViewer/actions/workflows/windows-smoke.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## Overview

AnComicsViewer is a modular PySide6 + OpenCV-based comics viewer that detects rectangular panels using a sophisticated heuristic pipeline. It features an interactive Panel Tuning dialog for real-time parameter adjustment and includes optimizations like LRU caching and thread-safe detection.

## Features

- **Heuristic panel detection** using adaptive thresholding, morphological cleanup, light/gutter splitting, and recursive projection splitting
- **Interactive Panel Tuning dialog** with DPI-stable fractional thresholds, projection smoothing, and title-row heuristics
- **Modular architecture** with clean separation of concerns for maintainability
- **LRU caching** for efficient panel detection results
- **Thread-safe detection** with proper locking mechanisms
- **Preset configurations** for Franco-Belge, Manga, and Newspaper comics
- **Cross-platform support** (macOS, Windows, Linux)
- **Modern Qt-based UI** with panel overlay visualization

## Quick Start

### Prerequisites
- Python 3.9+ (tested with 3.11)
- Virtual environment (recommended)

### Installation

1. **Clone and setup:**
   ```bash
   git clone https://github.com/r45635/AnComicsViewer.git
   cd AnComicsViewer
   python -m venv .venv
   ```

2. **Activate environment:**
   ```bash
   # macOS/Linux
   source .venv/bin/activate

   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Using helper scripts (recommended):

```bash
# macOS/Linux
chmod +x run.sh
./run.sh

# Windows
.\run_win.ps1
```

#### As a Python module:
```bash
python -m ancomicsviewer
```

#### Direct execution:
```bash
python AnComicsViewer.py
```

## Usage

1. **Launch the app** using one of the methods above
2. **Open a PDF** via File menu or drag & drop
3. **Navigate pages** using toolbar buttons or keyboard shortcuts
4. **Toggle panel mode** with Ctrl+2 to see detected panels
5. **Navigate panels** with N (next) and Shift+N (previous)
6. **Access Panel Tuning** via the gear button in the toolbar
7. **Apply presets** for different comic styles (Franco-Belge, Manga, Newspaper)

### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Open PDF | Ctrl+O |
| First page | Home |
| Last page | End |
| Previous page | Page Up |
| Next page | Page Down |
| Zoom in | Ctrl++ |
| Zoom out | Ctrl+- |
| Fit to width | Ctrl+1 |
| Fit to page | Ctrl+0 |
| Toggle panels | Ctrl+2 |
| Next panel | N |
| Previous panel | Shift+N |
| Cycle framing | F |

## Project Structure

```
AnComicsViewer/
├── AnComicsViewer.py          # Legacy entry point (backward compatible)
├── ancomicsviewer/            # Main package
│   ├── __init__.py           # Package exports
│   ├── __main__.py           # Module entry point
│   ├── config.py             # Configuration dataclasses & presets
│   ├── detector.py           # Panel detection engine
│   ├── image_utils.py        # QImage/NumPy conversions
│   ├── pdf_view.py           # Custom PDF view widget
│   ├── dialogs.py            # Tuning dialogs
│   ├── cache.py              # LRU cache implementation
│   └── main_window.py        # Main application window
├── smoke_test.py             # Integration test
├── requirements.txt          # Python dependencies
├── run.sh                    # macOS/Linux launcher
└── run_win.ps1               # Windows launcher
```

## Architecture

### Key Components

| Module | Description |
|--------|-------------|
| `config.py` | `DetectorConfig` and `AppConfig` dataclasses with 40+ tunable parameters |
| `detector.py` | `PanelDetector` class with multi-route detection (Adaptive, LAB, Canny) |
| `cache.py` | Thread-safe `LRUCache` and `PanelCache` for efficient caching |
| `pdf_view.py` | `PannablePdfView` with pan/zoom and overlay support |
| `dialogs.py` | `PanelTuningDialog` for interactive parameter adjustment |
| `main_window.py` | `ComicsView` main application window |

### Detection Pipeline

```
PDF Page → QImage → NumPy (RGBA)
    ↓
Grayscale + LAB L-channel (CLAHE)
    ↓
┌─────────────────────────────────────┐
│ Route 1: Adaptive Threshold         │
│ Route 2: LAB L-channel (fallback)   │
│ Route 3: Canny edges (optional)     │
└─────────────────────────────────────┘
    ↓
Contours → Bounding Boxes
    ↓
Base Filters (area%, fill ratio, size)
    ↓
IoU-based Merge
    ↓
Recursive Light Split (gutter detection)
    ↓
Title Row Filter
    ↓
Reading Order Sort (LTR/RTL)
    ↓
Panel QRectF[] (page points, 72 DPI)
```

### Performance Optimizations

- **LRU Cache**: Stores detection results per page with configurable size
- **Config Hash Validation**: Automatically invalidates cache when parameters change
- **Thread Safety**: Proper locking for concurrent access
- **Contiguous Arrays**: NumPy arrays optimized for OpenCV operations
- **Lazy Loading**: Dependencies loaded only when needed

## Testing

Run the smoke test to verify installation:

```bash
python smoke_test.py
```

Expected output:
```json
{
  "count": 2,
  "rects": [
    {"x": 33.0, "y": 83.0, "w": 735.0, "h": 335.0},
    {"x": 43.0, "y": 433.0, "w": 715.0, "h": 535.0}
  ]
}
```

## Troubleshooting

### macOS Issues
- **Qt cocoa plugin not found**: Use the provided `run.sh` script
- **PySide6 compatibility**: Tested with `PySide6==6.8.3`

### Windows Issues
- **Platform plugin errors**: Use the `run_win.ps1` PowerShell script
- **Path issues**: Ensure Python and pip are in your PATH

### General Issues
- **Import errors**: Activate the virtual environment first
- **Missing dependencies**: Run `pip install -r requirements.txt`

## API Usage

```python
from ancomicsviewer import PanelDetector, DetectorConfig
from PySide6.QtCore import QSizeF
from PySide6.QtGui import QImage

# Create detector with custom config
config = DetectorConfig(
    adaptive_block=51,
    min_fill_ratio=0.6,
    reading_rtl=True,  # Manga mode
)
detector = PanelDetector(config)

# Detect panels
qimage = QImage("page.png")
page_size = QSizeF(612, 792)  # Letter size in points
panels = detector.detect_panels(qimage, page_size)

for i, panel in enumerate(panels):
    print(f"Panel {i+1}: ({panel.x()}, {panel.y()}) {panel.width()}x{panel.height()}")
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the smoke test
5. Submit a pull request

## Status

- Core panel detection working
- Interactive tuning dialog
- Modular architecture (v2.0)
- LRU caching and performance optimizations
- Cross-platform support (macOS, Windows)
- CI/CD pipeline

---

<div align="center">
Made with care for comic and manga readers everywhere

**Author:** Vincent Cruvellier
</div>
