#!/bin/bash
set -e
echo '🚀 Starting optimized YOLO training...'
yolo train --model "yolov8m.pt" --data "/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/dataset/multibd_enhanced.yaml" --epochs 250 --imgsz 1024 --batch 4 --nms --iou 0.3 --conf 0.25 --mosaic 0.3 --mixup 0.0 --device "mps" --workers 4 --cache --amp --optimizer "AdamW" --lr0 0.001 --lrf 0.01 --momentum 0.937 --weight_decay 0.0005 --hsv_h 0.015 --hsv_s 0.7 --hsv_v 0.4 --degrees 5.0 --translate 0.1 --scale 0.5 --shear 2.0 --flipud 0.0 --fliplr 0.5 --patience 50 --save --save_period 25 --plots --verbose --name "ancomics_class_aware_nms_tiling_250ep" --seed 42 --deterministic
echo '✅ Training completed!'
