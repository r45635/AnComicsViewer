# AnComicsViewer Standalone Package - File Inventory

## 📦 Core Application Files

### 🚀 **Main Application**
- `main.py` (949 lines) - Complete viewer with AI detection, GUI, metrics, auto-reload

### 🔧 **Startup Scripts**  
- `run.sh` - macOS/Linux startup script (auto-installs dependencies)
- `run.bat` - Windows startup script (auto-installs dependencies)

### 📋 **Dependencies**
- `requirements_standalone.txt` - Complete dependency list including:
  - PySide6 (GUI framework)
  - PyMuPDF (PDF processing) 
  - ultralytics (YOLO models)
  - numpy, PyYAML, torch, torchvision
  - Optional: Pillow, opencv-python

### 🧠 **AI Model**
- `anComicsViewer_v01.pt` (23MB) - Pre-trained YOLO model for panel/balloon detection

### ⚙️ **Configuration Files**
- `config/detect_refined.yaml` - **RECOMMENDED** - Strict thresholds, cover page rules
- `config/detect_with_merge.yaml` - Standard configuration with merging
- `config/detect_strict.yaml` - Very conservative detection

### 📚 **Documentation**
- `README_STANDALONE.md` - Complete setup and usage guide

## 🎯 **Key Features Included**

### ✅ **Core Functionality**
- PDF rendering at 300 DPI
- YOLO-based panel/balloon detection
- Smart post-processing with 14+ configurable parameters
- Cover page handling (avoids phantom panels)
- Auto-reload last file and page position
- Quality metrics computation and JSON export

### ✅ **User Interface**
- Modern PySide6 GUI with graphics view
- Keyboard shortcuts (←/→ pages, ↑/↓ panels, Space, B, R, D)
- Toolbar with open, model load, navigation controls
- Status bar with real-time feedback

### ✅ **AI Detection Pipeline**
- Multi-scale tiled inference for large images
- Class-specific confidence thresholds
- Advanced NMS (Non-Maximum Suppression)
- Panel↔balloon attachment validation
- Size and margin filtering
- Overlap detection and quality scoring

### ✅ **Configuration System**
- YAML-based parameter management
- Real-time config reloading
- Debug mode with overlay generation
- Metrics export for analysis

## 🚀 **Usage Instructions**

### **Simple Launch**
```bash
# macOS/Linux
./run.sh

# Windows  
run.bat

# Manual
python main.py
```

### **Advanced Usage**
```bash
# Open specific file
python main.py --pdf "comic.pdf" --page 5

# Use specific config
python main.py --config config/detect_refined.yaml

# Debug mode
python main.py --pdf "comic.pdf" --debug-detect --save-debug-overlays debug/

# Export metrics
python main.py --metrics-out analysis.json
```

## 📁 **Directory Structure After Extraction**

```
AnComicsViewer_Standalone/
├── main.py                    # 🚀 Main application (949 lines)
├── requirements_standalone.txt # 📋 Dependencies
├── README_STANDALONE.md       # 📚 Documentation  
├── run.sh                     # 🔧 macOS/Linux launcher
├── run.bat                    # 🔧 Windows launcher
├── anComicsViewer_v01.pt      # 🧠 YOLO model (23MB)
└── config/                    # ⚙️ Configuration files
    ├── detect_refined.yaml    #   └─ Recommended (strict)
    ├── detect_with_merge.yaml #   └─ Standard
    └── detect_strict.yaml     #   └─ Conservative
```

## 🎯 **Total Package Size**: ~24MB
- Main code: ~50KB
- YOLO model: ~23MB  
- Config files: ~5KB
- Documentation: ~10KB
- Scripts: ~2KB

## ✅ **Self-Contained & Portable**
- No external model downloads required
- Auto-installs Python dependencies
- Cross-platform (Windows/macOS/Linux)
- No internet connection needed after setup
- Ready to run on any PDF comic collection
