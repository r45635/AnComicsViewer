<div align="center">

![AnComicsViewer Logo](logo.png)

# AnComicsViewer

**A lightweight PDF/manga/comics reader with intelligent panel detection**

[![Windows CI](https://github.com/r45635/AnComicsViewer/actions/workflows/windows-smoke.yml/badge.svg)](https://github.com/r45635/AnComicsViewer/actions/workflows/windows-smoke.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

## Overview

AnComicsViewer is a single-file PySide6 + OpenCV-based viewer that detects rectangular panels on comic book pages using a sophisticated heuristic pipeline. It features an interactive Panel Tuning dialog for real-time parameter adjustment and immediate re-detection.

## ✨ Features

- 🎯 **Heuristic panel detection** using adaptive thresholding, morphological cleanup, light/gutter splitting, and recursive projection splitting
- ⚙️ **Interactive Panel Tuning dialog** with DPI-stable fractional thresholds, projection smoothing, and title-row heuristics
- 📊 **Two-scenario title detection** (many small OR few big title boxes)
- 🔍 **Runtime parameter logging** printed to stdout for reproducible detection settings
- 🖥️ **Cross-platform support** (macOS, Windows, Linux)
- 📱 **Modern Qt-based UI** with panel overlay visualization

## 🚀 Quick Start

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

#### macOS (recommended method):
```bash
chmod +x run.sh
./run.sh
```

#### Windows:
```powershell
.\run_win.ps1
```

#### Manual (any platform):
```bash
# macOS: ensure Qt can find the cocoa platform plugin
QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import PySide6, os; print(os.path.join(os.path.dirname(PySide6.__file__), 'Qt', 'plugins', 'platforms'))") \
  .venv/bin/python AnComicsViewer.py
```

## 🎮 Usage

1. **Launch the app** using one of the methods above
2. **Open a PDF** via File menu or drag & drop
3. **Navigate pages** using toolbar buttons or keyboard shortcuts
4. **Access Panel Tuning** via the ⚙️ button in the toolbar
5. **Adjust parameters** and click "Apply" to see immediate results
6. **Monitor detection logs** in the terminal for debugging

### Expected Output
The app prints parameter snapshots and per-page detection debug messages to the terminal:
```
[Panels] params: ab=51 C=5 mk=7 mi=2 min_area=0.015 max_area=0.95 min_fill=0.55 min_px=80 ... psk=17
[Panels] Converted to gray: 1200x1858
[Panels] Adaptive route -> 5 rects
[Panels] [title-row] y=0.14 h=0.070 n=2 medW=0.26 L=0.66 -> KEEP (many_small=False few_big=True)
```

## 🔧 Troubleshooting

### macOS Issues
- **Qt cocoa plugin not found**: Use the provided `run.sh` script or set `QT_QPA_PLATFORM_PLUGIN_PATH` as shown above
- **PySide6 compatibility**: Tested with `PySide6==6.8.3`. Adjust in `requirements.txt` if needed

### Windows Issues  
- **Platform plugin errors**: Use the `run_win.ps1` PowerShell script
- **Path issues**: Ensure Python and pip are in your PATH

### General Issues
- **Import errors**: Activate the virtual environment: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)
- **Missing dependencies**: Run `pip install -r requirements.txt`

## 🧪 Testing

Run the smoke test to verify installation:
```bash
python smoke_test.py
```

Expected output:
```json
{
  "count": 3,
  "rects": [
    {"x": 50.0, "y": 100.0, "w": 700.0, "h": 300.0},
    ...
  ]
}
```

## 👨‍💻 Development

### Project Structure
- `AnComicsViewer.py` - Main application (single file)
- `PanelDetector` class - Detection pipeline and tunables
- `PanelTuningDialog` - Interactive parameter adjustment UI
- `ComicsView` - Main window with PDF viewer and panel overlay

### Key Classes
- **PanelDetector**: Core detection logic with configurable parameters
- **PanelTuningDialog**: UI for real-time parameter tuning
- **ComicsView**: Main application window and PDF handling

### Detection Pipeline
1. **Adaptive thresholding** - Convert to binary mask
2. **Morphological operations** - Clean up noise
3. **Contour detection** - Find rectangular regions
4. **Light/gutter splitting** - Recursive panel separation
5. **Title-row filtering** - Remove text-heavy regions
6. **Panel merging** - Combine adjacent regions

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the smoke test
5. Submit a pull request

## 📊 Status

- ✅ Core panel detection working
- ✅ Interactive tuning dialog
- ✅ Cross-platform support (macOS, Windows)  
- ✅ CI/CD pipeline
- ⏳ Additional detection algorithms (future)
- ⏳ Batch processing support (future)

---

<div align="center">
Made with ❤️ for comic and manga readers everywhere
</div>

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
