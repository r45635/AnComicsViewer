#!/usr/bin/env python3
"""
Panel Detector - Module de détection de panels BD standalone
Remplace l'ancien système intégré pour plus de clarté et robustesse.
"""

from ultralytics import YOLO
import numpy as np
import logging
from typing import Optional

log = logging.getLogger("Panels")

ACCEPT = {"panel", "panel_inset"}

def _norm(s: str) -> str:
    """Normalise un nom de classe."""
    return (s or "").strip().lower().replace("-", "_").replace(" ", "_")

def _ensure_rgb_uint8(img):
    """
    Assure que l'image est en format RGB uint8 compatible YOLO.
    img: np.ndarray HxWx{3,4} quelconque
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim != 3:
        # grayscale -> RGB
        img = np.stack([img]*3, axis=-1)
    if img.shape[2] == 4:
        # drop alpha
        img = img[:, :, :3]
    # YOLO accepte RGB ou BGR; ici on reste en RGB, ça marche très bien.
    return np.ascontiguousarray(img)

class PanelDetector:
    """Détecteur de panels BD utilisant YOLO."""
    
    def __init__(self, weights_path: str, device: Optional[str] = None):
        """
        Initialise le détecteur.
        
        Args:
            weights_path: Chemin vers le fichier .pt YOLO
            device: Device ('cpu', 'cuda', 'mps', etc.)
        """
        log.info(f"[Panels] weights path loaded = {weights_path}")
        
        self.model = YOLO(weights_path)
        if device:
            try: 
                self.model.to(device)
                log.info(f"[Panels] device set to {device}")
            except Exception as e:
                log.warning(f"[Panels] failed to set device {device}: {e}")
        
        # Mapping id→nom et nom_normalisé→id
        names = getattr(self.model, "names", None)
        log.info(f"[Panels] model.names = {names}")
        
        if isinstance(names, dict):
            self.id2name = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            self.id2name = {i: str(v) for i, v in enumerate(names)}
        else:
            self.id2name = {}
            log.error("[Panels] No valid names found in model!")
        
        self.name2id = {_norm(v): k for k, v in self.id2name.items()}
        log.info(f"[Panels] class mapping: {self.id2name}")
        log.info(f"[Panels] normalized mapping: {self.name2id}")

    def detect(self, img_rgb_np: np.ndarray,
               conf: float = 0.25, iou: float = 0.60,
               imgsz: Optional[int] = None, max_det: int = 400):
        """
        Détecte les panels dans une image.
        
        Args:
            img_rgb_np: Image RGB en numpy array
            conf: Seuil de confiance
            iou: Seuil IoU pour NMS
            imgsz: Taille d'image pour YOLO (auto si None)
            max_det: Nombre max de détections
            
        Returns:
            List[dict]: Liste des panels détectés
        """
        img = _ensure_rgb_uint8(img_rgb_np)
        H, W = img.shape[:2]
        
        if imgsz is None:
            # Taille multiple de 32, bornée entre 640 et 1536
            imgsz = int(np.clip(max(H, W), 640, 1536) // 32 * 32)
            log.debug(f"[Panels] auto imgsz = {imgsz} for image {H}x{W}")

        log.info(f"[Panels] starting detection with conf={conf}, iou={iou}, imgsz={imgsz}")

        # IMPORTANT : **pas** de classes=... ici
        res = self.model.predict(
            source=img, 
            conf=conf, 
            iou=iou, 
            imgsz=imgsz,
            max_det=max_det, 
            augment=False, 
            agnostic_nms=False,
            verbose=False, 
            classes=None  # ⚠️ JAMAIS de filtre classes ici
        )

        dets = self._to_dets(res)
        log.info(f"[Panels] raw preds = {len(dets)}; by class = {self._counts_by_name(dets)}")

        # Filtrage par nom de classe
        keep = np.isin(dets[:, 5], [_norm(c) for c in ACCEPT])
        panels = dets[keep]
        log.info(f"[Panels] after filter = {len(panels)}")

        if not len(panels):
            # Fallback permissif
            log.info("[Panels] trying fallback with relaxed parameters...")
            res2 = self.model.predict(
                source=img, 
                conf=max(0.05, conf*0.4), 
                iou=min(0.70, iou+0.10),
                imgsz=max(imgsz, 1536), 
                max_det=max(600, max_det),
                augment=False, 
                agnostic_nms=False, 
                verbose=False, 
                classes=None
            )
            dets = self._to_dets(res2)
            log.info(f"[Panels] fallback raw = {len(dets)}; by class = {self._counts_by_name(dets)}")
            keep = np.isin(dets[:, 5], [_norm(c) for c in ACCEPT])
            panels = dets[keep]
            log.info(f"[Panels] fallback after filter = {len(panels)}")

        # Conversion au format attendu
        result = []
        for x1, y1, x2, y2, p, n in panels:
            result.append({
                'x1': float(x1), 
                'y1': float(y1), 
                'x2': float(x2), 
                'y2': float(y2),
                'conf': float(p), 
                'name': str(n)
            })
        
        log.info(f"[Panels] final result: {len(result)} panels detected")
        return result

    def _to_dets(self, results):
        """Convertit les résultats YOLO en array numpy."""
        if not results or results[0].boxes is None or results[0].boxes.cls is None:
            return np.empty((0, 6), dtype=object)
        
        b = results[0].boxes
        try:
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
        except Exception as e:
            log.error(f"[Panels] Error extracting boxes: {e}")
            return np.empty((0, 6), dtype=object)
        
        # Conversion des classes en noms normalisés
        names = np.array([_norm(self.id2name.get(i, f"unknown_{i}")) for i in cls], dtype=object)
        
        n = xyxy.shape[0]
        out = np.empty((n, 6), dtype=object)
        out[:, 0:4] = xyxy
        out[:, 4] = conf
        out[:, 5] = names
        return out

    @staticmethod
    def _counts_by_name(dets):
        """Compte les détections par nom de classe."""
        if len(dets) == 0:
            return {}
        from collections import Counter
        return dict(Counter(list(dets[:, 5])))

    def get_model_names_signature(self):
        """Retourne une signature des noms de classes pour le cache."""
        return ",".join(sorted(map(_norm, self.id2name.values())))
