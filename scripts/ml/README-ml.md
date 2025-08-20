# ML Detection System

This directory contains the machine learning components for enhanced panel detection using YOLOv8 segmentation.

## Architecture

- `detectors/base.py`: Abstract base class for all panel detectors
- `detectors/yolo_seg.py`: YOLOv8-based segmentation detector for panels, balloons, and titles
- `tools/export_pdf_pages.py`: Utility to extract pages from PDF comics for dataset creation
- `tools/overlay_predictions.py`: Visualization tool to overlay detection results on images

## YOLOv8 Segmentation Detector

The `YoloSegPanelDetector` uses instance segmentation to detect:
- **Class 0**: Panels (comic book panels)
- **Class 1**: Balloons (speech/thought bubbles)  
- **Class 2**: Titles (text overlays, often chapter/page titles)

Key features:
- IoU-based title overlap filtering (removes titles that significantly overlap with panels)
- Reading order panel sorting (left-to-right, top-to-bottom)
- DPI-aware coordinate scaling
- QImage to RGB conversion for YOLO compatibility

## Usage

```python
from detectors.yolo_seg import YoloSegPanelDetector
from PySide6.QtGui import QImage
from PySide6.QtCore import QSizeF

detector = YoloSegPanelDetector("path/to/model.pt")
panels = detector.detect_panels(qimage, page_size)
```

## Training Data Preparation

1. Use `tools/export_pdf_pages.py` to extract pages:
   ```bash
   python tools/export_pdf_pages.py comic.pdf --out dataset/images/train --dpi 300
   ```

2. Annotate using tools like Roboflow or LabelStudio for segmentation masks

3. Train using the dataset configuration in `dataset.yaml`

## Visualization

Preview detection results:
```bash
python tools/overlay_predictions.py model.pt image1.png image2.png --out predictions/
```
