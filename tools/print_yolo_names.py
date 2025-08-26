from ultralytics import YOLO
m = YOLO("runs/multibd_enhanced_v2/yolov8s-mps-1280/weights/best.pt")
print("names:", m.names)
