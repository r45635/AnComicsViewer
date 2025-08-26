# detectors/ultra_panel_detector.py
from ultralytics import YOLO
import numpy as np
import logging

log = logging.getLogger("Panels")

# Accepte ces noms normalisés (à adapter si besoin)
ACCEPTED = {"panel", "panel_inset", "balloon"}

def _norm(s: str) -> str:
    return (s or "").strip().lower().replace("-", "_").replace(" ", "_")

def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Standardise l'image pour YOLO: HxWx3, RGB, uint8, contiguë."""
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 2:  # grayscale -> RGB
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] == 4:  # RGBA/BGRA -> drop alpha, on garde les 3 premiers canaux
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)

class UltraPanelDetector:
    def __init__(self, weights_path: str, device: str | None = None):
        self.model = YOLO(weights_path)
        if device:
            try:
                self.model.to(device)
            except Exception:
                pass

        # id->name + name->id normalisé
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            self.id2name = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            self.id2name = {i: str(v) for i, v in enumerate(names)}
        else:
            self.id2name = {}
        self.name2id = {_norm(v): k for k, v in self.id2name.items()}
        log.info(f"[Panels] model.names = {self.id2name}")

    def _parse(self, results):
        if not results or results[0].boxes is None or results[0].boxes.cls is None:
            return []
        b = results[0].boxes
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls  = b.cls.cpu().numpy().astype(int)
        out = []
        for (x1,y1,x2,y2), p, c in zip(xyxy, conf, cls):
            name = _norm(self.id2name.get(int(c), ""))
            out.append(dict(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                            conf=float(p), name=name, cls=int(c)))
        return out

    def detect(self, img_rgb_np: np.ndarray,
               conf: float = 0.25, iou: float = 0.60,
               imgsz: int | None = None, max_det: int = 400):
        img = ensure_rgb_uint8(img_rgb_np)
        H, W = img.shape[:2]
        if imgsz is None:
            # multiple de 32, borné 640--1536
            imgsz = int(np.clip(max(H, W), 640, 1536) // 32 * 32)

        # NE PAS passer classes=... (on filtre après par noms)
        r = self.model.predict(
            source=img, conf=conf, iou=iou, imgsz=imgsz,
            max_det=max_det, agnostic_nms=False, verbose=False, classes=None
        )
        dets = self._parse(r)
        log.info(f"[Panels] raw={len(dets)} by={_count_by(dets)}")

        keep = [d for d in dets if d["name"] in ACCEPTED]
        if not keep:
            # Fallback permissif
            r2 = self.model.predict(
                source=img,
                conf=max(0.05, conf*0.5),
                iou=min(0.75, iou+0.1),
                imgsz=max(imgsz, 1536),
                max_det=max(600, max_det),
                agnostic_nms=False, verbose=False, classes=None
            )
            dets2 = self._parse(r2)
            log.info(f"[Panels] fb-raw={len(dets2)} by={_count_by(dets2)}")
            keep = [d for d in dets2 if d["name"] in ACCEPTED]
        log.info(f"[Panels] keep={len(keep)}")
        return keep

def _count_by(dets):
    from collections import Counter
    return dict(Counter([d["name"] for d in dets]))
