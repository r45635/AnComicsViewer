# AnComicsViewer - Standalone Edition

An advanced PDF comic book viewer with AI-powered panel and balloon detection using YOLO models.

## Features

- **PDF Rendering**: High-quality page rendering at 300 DPI
- **AI Detection**: YOLO-based panel and balloon detection  
- **Smart Post-processing**: Refined detection with configurable thresholds
- **Cover Page Handling**: Special rules for cover pages to avoid phantom panels
- **Quality Metrics**: Per-page analysis with overlap detection
- **Reading Mode**: Guided sequential reading of panels/balloons
- **Auto-reload**: Remembers last opened file and page position
- **Debug Overlays**: Visual debugging of detection results

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_standalone.txt
```

### 2. Run the Application
```bash
python main.py
```

### 3. Command Line Usage
```bash
# Open specific PDF and page
python main.py --pdf "your_comic.pdf" --page 5

# Use specific configuration
python main.py --config config/detect_refined.yaml

# Debug mode with overlays
python main.py --pdf "comic.pdf" --debug-detect --save-debug-overlays debug/

# Export quality metrics
python main.py --pdf "comic.pdf" --metrics-out metrics.json
```

## Configuration Files

- `config/detect_refined.yaml` - **Recommended**: Strict thresholds, good for covers
- `config/detect_with_merge.yaml` - Standard detection with merging
- `config/detect_strict.yaml` - Very conservative detection

## Model Files

Place YOLO model files (`.pt`) in the root directory. The app will auto-detect:
- `anComicsViewer_v01.pt` (included)
- `best.pt` from training runs
- Standard YOLO models (`yolov8s.pt`, etc.)

## Key Keyboard Shortcuts

- **←/→**: Previous/Next page  
- **↑/↓**: Previous/Next panel (reading mode)
- **Space**: Toggle panel visibility
- **B**: Toggle balloon detection
- **R**: Toggle reading mode
- **D**: Run detection on current page

## Directory Structure

```
AnComicsViewer/
├── main.py                    # Main application
├── requirements_standalone.txt # Dependencies
├── anComicsViewer_v01.pt      # YOLO model
├── config/                    # Configuration files
│   ├── detect_refined.yaml    # Recommended config
│   ├── detect_with_merge.yaml # Standard config
│   └── detect_strict.yaml     # Conservative config
├── debug/                     # Debug overlay outputs
└── outputs/                   # Metrics outputs
```

## Configuration Options

Key parameters in YAML config files:

```yaml
# Detection thresholds
panel_conf: 0.42        # Panel confidence threshold
balloon_conf: 0.38      # Balloon confidence threshold

# Cover page rules
cover_rule_enable: true
cover_pages: [0, 1]     # Pages treated as covers
cover_max_panels_before_force: 5

# Post-processing
max_panels: 10          # Max panels per page
max_balloons: 20        # Max balloons per page
panel_area_min_pct: 0.04  # Min panel size (4% of page)
```

## Troubleshooting

**No YOLO model found**: Place a `.pt` model file in the root directory

**Poor detection**: Try different config files or adjust thresholds

**GUI issues**: Ensure PySide6 is properly installed

## License

[Your license here]
