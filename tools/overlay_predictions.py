#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys; sys.path.append(".."); from detectors.yolo_seg import YoloSegPanelDetector

def overlay_predictions(img_path, model, out):
    detector = YoloSegPanelDetector(model)
    img = cv2.imread(str(img_path))
    assert img is not None, f"Can't load {img_path}"
    from PySide6.QtGui import QImage
    from PySide6.QtCore import QSizeF
    h,w,c = img.shape
    qimg = QImage(img.data, w, h, c*w, QImage.Format.Format_BGR888)
    page_size = QSizeF(w, h)
    panels = detector.detect_panels(qimg, page_size)
    for i,p in enumerate(panels):
        x1,y1,x2,y2 = int(p.left()),int(p.top()),int(p.right()),int(p.bottom())
        cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img, f"#{i+1}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imwrite(str(out), img)
    print(f"=> {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="pred_overlay")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for i in a.images:
        p = Path(i)
        o = Path(a.out) / f"{p.stem}_pred{p.suffix}"
        overlay_predictions(p, a.model, o)

if __name__ == "__main__":
    main()
