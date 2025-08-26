from ultralytics import YOLO
import cv2, sys
import numpy as np
from collections import Counter

if len(sys.argv) < 3:
    print("Usage: python quickcheck.py model.pt image.png")
    sys.exit(1)

pt = sys.argv[1]  # chemin .pt
img_path = sys.argv[2]  # PNG/JPG exporté d'une page
m = YOLO(pt)
print("model.names =", m.names)
img = cv2.imread(img_path)  # BGR, mais YOLO s'en moque
if img is None:
    print("Erreur: impossible de charger l'image", img_path)
    sys.exit(1)

r = m.predict(img, conf=0.25, iou=0.6, imgsz=1280, verbose=False, classes=None)
b = r[0].boxes
if b is None or b.cls is None or len(b.cls)==0:
    print("Aucune box brute")
else:
    # Conversion sécurisée pour tensor YOLO
    if hasattr(b.cls, 'cpu'):
        cls = b.cls.cpu().numpy().astype(int)
    else:
        cls = np.array(b.cls).astype(int)
    print("cls ids  :", Counter(cls))
    print("cls names:", Counter([str(m.names[int(i)]).strip().lower() for i in cls]))
