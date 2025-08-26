# acv/detect/yolo_panels.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics YOLOv8 introuvable. Installe: pip install ultralytics") from e

def _norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", s.lower())

TARGET_SYNONYMS = {
    "panel": {"panel", "case", "cadre", "frame"},
    "panel_inset": {"panelinset", "inset", "insetpanel"},
    # Note: balloons exclus car ce ne sont pas des panels de navigation
    # "balloon": {"balloon", "bubble", "bulle", "speech", "dialog", "dialogue"},
}

@dataclass
class DetectCfg:
    weights: str
    device: str = "mps"         # "cpu" | "cuda" | "mps"
    imgsz: int = 1280
    conf_panel: float = 0.20
    conf_inset: float = 0.20
    conf_balloon: float = 0.30
    iou: float = 0.60
    drop_fullpage_ratio: float = 0.88
    min_area_ratio: float = 0.002
    verbose: bool = True

class PanelDetector:
    def __init__(self, cfg: DetectCfg):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        try:
            self.model.to(cfg.device)
        except Exception:
            pass

        raw_names: Dict[int, str] = getattr(self.model, "names", {})
        self.class_map: Dict[int, str] = {int(i): _norm(n) for i, n in raw_names.items()}

        allowed_idx = set()
        for syns in TARGET_SYNONYMS.values():
            for i, nm in self.class_map.items():
                if nm in syns:
                    allowed_idx.add(i)
        
        # Exclure explicitement les balloons pour la navigation
        balloon_classes = set()
        for i, nm in self.class_map.items():
            if nm in {"balloon", "bubble", "bulle", "speech", "dialog", "dialogue"}:
                balloon_classes.add(i)
        
        allowed_idx = allowed_idx - balloon_classes  # Retirer les balloons
        
        if not allowed_idx:
            # Fallback: autoriser toutes les classes sauf balloons
            allowed_idx = set(self.class_map.keys()) - balloon_classes
            
        self.allowed_idx = sorted(allowed_idx)

        if cfg.verbose:
            print("[Panels] 🔤 model.names:", raw_names)
            print("[Panels] 🎯 allowed class indices:", self.allowed_idx)

        self.per_class_conf: Dict[int, float] = {}
        for i, nm in self.class_map.items():
            if i not in self.allowed_idx:
                continue
            if nm in TARGET_SYNONYMS["panel"]:
                self.per_class_conf[i] = cfg.conf_panel
            elif nm in TARGET_SYNONYMS["panel_inset"]:
                self.per_class_conf[i] = cfg.conf_inset
            # Note: balloons exclus de la navigation
            else:
                # Classes non reconnues = confidence par défaut panel
                self.per_class_conf[i] = cfg.conf_panel

    def detect(
        self,
        image: np.ndarray,
        content_size: Optional[Tuple[int, int]] = None,
        *,
        override_conf: Optional[float] = None,
        override_iou: Optional[float] = None,
        tta: bool = False,
    ) -> List[Dict[str, Any]]:
        cfg = self.cfg
        iou = override_iou if override_iou is not None else cfg.iou
        global_conf = override_conf if override_conf is not None else min(self.per_class_conf.values() or [cfg.conf_panel])

        res = self.model.predict(
            source=image,
            imgsz=cfg.imgsz,
            conf=global_conf,
            iou=iou,
            device=cfg.device,
            verbose=False,
            augment=bool(tta),
        )
        if not res:
            return []
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        boxes = r.boxes
        # Conversion sécurisée tensor/numpy
        if hasattr(boxes.cls, 'cpu'):
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy().astype(float)
            xyxy = boxes.xyxy.cpu().numpy().astype(float)
        else:
            cls = np.array(boxes.cls).astype(int)
            conf = np.array(boxes.conf).astype(float)
            xyxy = np.array(boxes.xyxy).astype(float)

        keep = np.isin(cls, np.array(self.allowed_idx, dtype=int))
        if keep.sum() == 0:
            if cfg.verbose:
                print(f"[Panels] ⚠️ no dets after class filter (allowed={self.allowed_idx}) → disabling filter this page")
            keep = np.ones_like(cls, dtype=bool)
        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]

        mask_conf = np.ones_like(conf, dtype=bool)
        for i in range(len(conf)):
            thr = self.per_class_conf.get(int(cls[i]), global_conf)
            if conf[i] < thr:
                mask_conf[i] = False
        xyxy, conf, cls = xyxy[mask_conf], conf[mask_conf], cls[mask_conf]
        if xyxy.size == 0:
            return []

        H, W = image.shape[:2]
        cont_w, cont_h = (content_size if content_size else (W, H))
        cont_area = float(cont_w * cont_h)

        out = []
        for b, p, c in zip(xyxy, conf, cls):
            x1, y1, x2, y2 = map(float, b.tolist())
            w, h = (x2 - x1), (y2 - y1)
            area = w * h
            if cont_area > 0 and area / cont_area < self.cfg.min_area_ratio:
                continue
            out.append([x1, y1, x2, y2, float(p), int(c)])

        if len(out) > 1 and cont_area > 0:
            keep2 = []
            dropped = 0
            for x1, y1, x2, y2, p, c in out:
                area = (x2 - x1) * (y2 - y1)
                if area / cont_area > self.cfg.drop_fullpage_ratio:
                    dropped += 1
                    continue
                keep2.append([x1, y1, x2, y2, p, c])
            if dropped and self.cfg.verbose:
                print(f"[Panels] 🧹 dropped {dropped} near-full-page boxes")
            out = keep2

        dets: List[Dict[str, Any]] = []
        for x1, y1, x2, y2, p, c in out:
            name = self.model.names.get(int(c), str(int(c)))
            dets.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(p),
                "cls": int(c),
                "name": name,
            })
        return dets

    def detect_all_classes(
        self,
        image: np.ndarray,
        content_size: Optional[Tuple[int, int]] = None,
        *,
        override_conf: Optional[float] = None,
        override_iou: Optional[float] = None,
        tta: bool = False,
    ) -> List[Dict[str, Any]]:
        """Détecte toutes les classes (panels + balloons) sans filtrage de classe."""
        cfg = self.cfg
        iou = override_iou if override_iou is not None else cfg.iou
        global_conf = override_conf if override_conf is not None else 0.20

        res = self.model.predict(
            source=image,
            imgsz=cfg.imgsz,
            conf=global_conf,
            iou=iou,
            device=cfg.device,
            verbose=cfg.verbose,
            augment=tta,
        )[0]

        if res.boxes is None:
            return []

        xyxy = res.boxes.xyxy
        conf = res.boxes.conf
        cls = res.boxes.cls

        if xyxy.size == 0:
            return []

        # Pas de filtrage par classe, on garde tout
        dets: List[Dict[str, Any]] = []
        for x1, y1, x2, y2, p, c in zip(xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3], conf, cls):
            name = self.model.names.get(int(c), str(int(c)))
            dets.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(p),
                "cls": int(c),
                "name": name,
            })
        return dets
