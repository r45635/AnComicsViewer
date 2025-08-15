# AnComicsViewer - Complete Setup Guide

This guide provides step-by-step instructions to set up the AnComicsViewer environment, including ML capabilities and dataset preparation for comic panel detection.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Application Installation](#application-installation)
- [ML Environment Setup](#ml-environment-setup)
- [Dataset Creation](#dataset-creation)
- [Annotation Setup](#annotation-setup)
- [Testing & Validation](#testing--validation)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **macOS**: 11.0+ (Big Sur or later)
- **Python**: 3.11 or 3.13 (tested with 3.13)
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space for dependencies and datasets

### Required Tools
- Git
- Python 3.11+ with pip
- Homebrew (for macOS dependencies)

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/r45635/AnComicsViewer.git
cd AnComicsViewer
```

### 2. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Verify activation (should show .venv path)
which python
```

### 3. Install Core Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PySide6 (specific version for stability)
pip install 'PySide6==6.8.3'

# Install core dependencies
pip install -r requirements.txt
```

**Note**: We use PySide6 6.8.3 specifically to avoid Qt plugin issues on macOS.

## Application Installation

### 1. Install Base Requirements

```bash
# Install OpenCV and NumPy for heuristic detection
pip install opencv-python numpy

# Install additional dependencies
pip install Pillow  # For image processing
```

### 2. Set Environment Variables (macOS)

```bash
# Add to your shell profile (.zshrc, .bash_profile, etc.)
export QT_MAC_WANTS_LAYER=1

# For current session
export QT_MAC_WANTS_LAYER=1
```

### 3. Test Basic Installation

```bash
# Test the application
python AnComicsViewer.py
```

**Expected behavior**: Application window opens without Qt plugin errors.

## ML Environment Setup

### 1. Create ML Branch

```bash
# Create and switch to experimental ML branch
git checkout -b experimental-ml
```

### 2. Install ML Dependencies

```bash
# Install YOLOv8 and ML dependencies
pip install ultralytics==8.2.0
pip install torch torchvision torchaudio

# Install additional ML tools
pip install scikit-learn matplotlib seaborn
```

### 3. Fix PyTorch Loading Issues

The ML detector automatically handles PyTorch security changes:

```python
# This is handled automatically in detectors/yolo_seg.py
import torch
# Patches torch.load to use weights_only=False for YOLO models
```

### 4. Test ML Environment

```bash
# Test YOLO import with proper matplotlib backend
python -c "import matplotlib; matplotlib.use('Agg'); from ultralytics import YOLO; print('YOLO OK')"
```

### 5. Download Pre-trained Model

```bash
# Download YOLOv8 nano segmentation model for testing
python -c "import matplotlib; matplotlib.use('Agg'); from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"
```

## Dataset Creation

### 1. Prepare Dataset Structure

```bash
# Create dataset directories
mkdir -p dataset/images/train
mkdir -p dataset/images/val
mkdir -p dataset/labels/train
mkdir -p dataset/labels/val
```

### 2. Extract Comic Pages

```bash
# Example with Golden City comic (replace with your PDF)
python tools/export_pdf_pages.py "Your Comic.pdf" --out dataset/images/train --dpi 300
```

**Important**: Ensure PDF file has proper permissions (chmod 644 if needed).

### 3. Create Train/Validation Split

```bash
# Create train/val split (90/10)
python - <<'EOF'
import os, random, shutil, glob
random.seed(42)
imgs = sorted(glob.glob('dataset/images/train/*.png'))
n_val = max(5, len(imgs)//10)   # ~10% validation (minimum 5)
val = sorted(random.sample(imgs, n_val))
os.makedirs('dataset/images/val', exist_ok=True)
for p in val:
    shutil.move(p, 'dataset/images/val/'+os.path.basename(p))
print(f"Moved {len(val)} images to validation set")
EOF
```

### 4. Define Classes

```bash
# Create class definitions
echo "panel" > dataset/predefined_classes.txt
echo "text" >> dataset/predefined_classes.txt
```

## Annotation Setup

### 1. Install Annotation Tool

```bash
# Install LabelMe (more compatible than LabelImg)
pip install labelme
```

### 2. Launch Annotation Interface

```bash
# Start annotation (creates GUI window)
python start_annotation.py
```

**Alternative manual launch**:
```bash
labelme dataset/images/train --output dataset/labels/train --nodata
```

### 3. Annotation Guidelines

**Classes to annotate**:
- **panel**: Comic panel boundaries (main story panels)
- **text**: Text regions (speech bubbles, titles, captions)

**Best practices**:
- Use polygon tool for precise boundaries
- Skip page borders and margins
- Include speech bubbles as 'text' class
- Save frequently (Ctrl+S)
- Use Ctrl+D for next image

### 4. Convert Annotations

After annotation, convert to YOLO format:

```bash
# Convert LabelMe JSON to YOLO format
python post_annotation_processing.py
```

## Testing & Validation

### 1. Test Heuristic Detector

```bash
# Launch application
python AnComicsViewer.py

# In the app:
# 1. Open PDF (Ctrl+O)
# 2. Toggle panels (Ctrl+2)
# 3. Navigate panels (N/Shift+N)
# 4. Try presets: Franco-Belge, Manga, Newspaper
```

### 2. Test ML Detector

```bash
# In AnComicsViewer:
# 1. Settings (⚙️) → Detector → Load ML weights...
# 2. Select yolov8n-seg.pt
# 3. Settings (⚙️) → Detector → YOLOv8 Seg (ML)
# 4. Test panel detection
```

### 3. VS Code Integration

```bash
# Use VS Code tasks
# Ctrl+Shift+P → "Tasks: Run Task" → "Run App"
# F5 → Debug with "Run ComicsView" configuration
```

## Project Structure

After complete setup:

```
AnComicsViewer/
├── .venv/                     # Python virtual environment
├── .vscode/                   # VS Code configuration
│   ├── tasks.json            # Build and run tasks
│   └── launch.json           # Debug configuration
├── detectors/                 # ML detector modules
│   ├── base.py               # Abstract detector interface
│   └── yolo_seg.py           # YOLOv8 implementation
├── tools/                     # Utility scripts
│   ├── export_pdf_pages.py   # PDF page extraction
│   ├── overlay_predictions.py # Visualization tool
│   └── golden_city_analyzer.py # Comic analysis
├── ml/                        # ML training configuration
│   ├── dataset.yaml          # YOLO dataset config
│   └── README-ml.md          # ML documentation
├── dataset/                   # Training dataset
│   ├── images/train/         # Training images
│   ├── images/val/           # Validation images
│   ├── labels/train/         # Training labels
│   └── labels/val/           # Validation labels
├── AnComicsViewer.py         # Main application
├── requirements.txt          # Python dependencies
├── requirements-ml.txt       # ML-specific dependencies
└── README.md                 # Project documentation
```

## Troubleshooting

### Common Issues

#### 1. Qt Plugin Errors (macOS)
```bash
# Error: "qt.qpa.plugin: Could not find the Qt platform plugin 'cocoa'"
# Solution: Set environment variable
export QT_MAC_WANTS_LAYER=1

# Add to shell profile for persistence
echo 'export QT_MAC_WANTS_LAYER=1' >> ~/.zshrc
```

#### 2. PDF Loading Errors
```bash
# Error: "Error loading PDF: Error.None_"
# Solution: Check file permissions
chmod 644 "Your PDF File.pdf"

# Verify file exists and is readable
file "Your PDF File.pdf"
```

#### 3. YOLO Model Loading Errors
```bash
# Error: "WeightsUnpickler error" or "weights_only" errors
# Solution: Already handled in detectors/yolo_seg.py
# The code automatically patches torch.load for YOLO compatibility
```

#### 4. Matplotlib Backend Issues
```bash
# Error: Font cache building or GUI backend errors
# Solution: Set non-interactive backend
python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot"
```

### Performance Optimization

#### 1. Detection Performance
- Use Detection DPI 150-200 for good speed/quality balance
- Try different presets based on comic style:
  - **Franco-Belge**: Traditional European comics
  - **Manga**: Japanese comics (RTL reading)
  - **Newspaper**: Comic strips

#### 2. Memory Usage
- Close unused applications during annotation
- Process large PDFs in batches
- Use validation set for testing, not training

### Verification Checklist

- [ ] Python virtual environment activated
- [ ] PySide6 6.8.3 installed and working
- [ ] AnComicsViewer launches without Qt errors
- [ ] PDF files can be opened and displayed
- [ ] Panel detection works with heuristic detector
- [ ] ML dependencies installed (ultralytics, torch)
- [ ] YOLOv8 model loads successfully
- [ ] ML detector integration functional
- [ ] Dataset structure created correctly
- [ ] Annotation tool (LabelMe) launches
- [ ] Train/validation split completed
- [ ] VS Code tasks and debug configuration working

## Next Steps

After completing this setup:

1. **Annotation**: Use LabelMe to annotate comic panels and text regions
2. **Training**: Train custom YOLO models with your annotated data
3. **Integration**: Test trained models in the AnComicsViewer interface
4. **Deployment**: Use the application for comic reading with enhanced panel detection

## Support

For issues not covered in this guide:

1. Check the main [README.md](README.md) for additional information
2. Review the [ML documentation](ml/README-ml.md) for training details
3. Examine log output for specific error messages
4. Verify all dependencies are correctly installed

## Version Information

- **AnComicsViewer**: Latest experimental-ml branch
- **PySide6**: 6.8.3 (recommended for stability)
- **YOLOv8**: 8.2.0 via ultralytics
- **Python**: 3.11+ (tested with 3.13)
- **Platform**: macOS 11.0+ (adaptable to Linux/Windows)

This setup guide ensures a complete, functional environment for comic panel detection research and development.
