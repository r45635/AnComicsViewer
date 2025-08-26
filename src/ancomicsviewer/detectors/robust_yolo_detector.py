"""
Détecteur YOLO Robuste selon les AR (Acceptance Requirements)
==========================================================
Implémentation complète des spécifications AR pour la détection robuste et générique.
"""

import os
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage
from src.ancomicsviewer.detect.yolo_panels import PanelDetector, DetectCfg

log = logging.getLogger("Panels")

def qimage_to_rgb_array(qimage: QImage) -> np.ndarray:
    """
    Conversion robuste QImage -> RGB array selon AR.
    """
    log.info(f"[Panels] 🔄 qimage_to_rgb_array: input QImage {qimage.width()}x{qimage.height()}, format={qimage.format()}")
    
    if qimage.isNull():
        log.error("[Panels] ❌ QImage is null!")
        raise ValueError("QImage is null")
    
    # Force conversion vers RGBA8888 pour standardiser
    original_format = qimage.format()
    if qimage.format() != QImage.Format.Format_RGBA8888:
        log.info(f"[Panels] 🔄 Converting from {original_format} to RGBA8888...")
        qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
        log.info(f"[Panels] ✅ Converted to RGBA8888")
    
    w, h = qimage.width(), qimage.height()
    bpl = qimage.bytesPerLine()
    log.info(f"[Panels] 📏 Image dimensions: {w}x{h}, bytesPerLine={bpl}")
    
    ptr = qimage.constBits()
    log.info(f"[Panels] 🔗 Got constBits pointer")
    
    # Conversion buffer sécurisée
    buffer_size = bpl * h
    log.info(f"[Panels] 📦 Buffer size: {buffer_size} bytes")
    buffer = bytes(ptr)[:buffer_size]
    log.info(f"[Panels] ✅ Buffer extracted: {len(buffer)} bytes")
    
    # Reshape en array RGBA
    log.info(f"[Panels] 🔄 Creating numpy array...")
    arr = np.frombuffer(buffer, dtype=np.uint8).reshape(h, bpl)[:, :w*4]
    log.info(f"[Panels] ✅ Numpy array created: shape={arr.shape}")
    
    rgba = arr.reshape(h, w, 4)
    log.info(f"[Panels] ✅ RGBA array: shape={rgba.shape}")
    
    # Extraire RGB (drop alpha)
    rgb = rgba[:, :, :3]
    log.info(f"[Panels] ✅ RGB extracted: shape={rgb.shape}")
    
    # Assurer la contiguïté mémoire (CRITICAL pour YOLO)
    result = np.ascontiguousarray(rgb)
    log.info(f"[Panels] ✅ Contiguous array: shape={result.shape}, dtype={result.dtype}, contiguous={result.flags.c_contiguous}")
    
    return result

class RobustYoloDetector:
    """
    Détecteur YOLO robuste conforme aux AR (Acceptance Requirements).
    
    Features AR:
    - Détection adaptative panel/panel_inset/balloon
    - Retry automatique avec seuils plus permissifs  
    - Suppression faux positifs pleine page
    - Cache sécurisé (pas de résultats vides)
    - Logs explicites avec préfixe [Panels]
    """
    
    def __init__(self):
        """Initialise le détecteur robuste selon AR."""
        # Paramètres AR par défaut SYNCHRONISÉS avec main_app.py
        self.model_path = "runs/multibd_enhanced_v2/yolov8s-mps-1280/weights/best.pt"
        self.render_dpi = 130  # AR: DPI = 130 (évite fragmentation, sync avec main_app._det_dpi)
        
        log.info("[Panels] 🔥 RobustYoloDetector: Initialisation AR-compliant...")
        
        # Configuration AR-compliant OPTIMISÉE pour haute résolution
        self._cfg = DetectCfg(
            weights=self.model_path,
            device=self._get_best_device(),  # AR: mps si dispo sinon cpu
            imgsz=1280,                      # AR: YOLO imgsz=1280
            conf_panel=0.15,                 # AR: conf(panel)=0.15 (moins strict pour éviter panels manqués)
            conf_inset=0.15,                 # AR: conf(panel_inset)=0.15 (moins strict)  
            conf_balloon=0.45,               # AR: conf(balloon)=0.45 (plus strict pour éviter faux positifs)
            iou=0.25,                        # AR: iou=0.25 (très strict pour éviter fragmentation)
            drop_fullpage_ratio=0.75,        # AR: > 75% = probablement trop grand (plus permissif pour pages mixtes)
            min_area_ratio=0.008,            # AR: < 0.8% = trop petit (plus strict pour haute résolution)
            verbose=True                     # AR: logs explicites
        )
        
        self._detector = PanelDetector(self._cfg)
        log.info("[Panels] ✅ RobustYoloDetector: Modèle AR-compliant chargé!")
        
        # Cache pour éviter résultats vides (AR) 
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # FORCER l'invalidation du cache pour les nouvelles coordonnées précises
        log.info("[Panels] 🧹 Invalidation forcée du cache pour nouvelles coordonnées précises")
        self._cache.clear()
        
    def _get_best_device(self) -> str:
        """Sélectionne le meilleur device selon AR."""
        try:
            import torch
            if torch.backends.mps.is_available():
                log.info("[Panels] 📱 Device: MPS (Apple Silicon)")
                return "mps"
        except Exception:
            pass
        
        log.info("[Panels] 💻 Device: CPU (fallback)")
        return "cpu"
        
    def get_model_info(self):
        """Retourne les informations du modèle pour compatibilité."""
        return {
            "name": "YOLO Robuste AR-Compliant",
            "render_dpi": self.render_dpi,
            "device": self._cfg.device,
            "confidence": self._cfg.conf_panel,  # Compatibilité avec UI existante
            "conf_panel": self._cfg.conf_panel,
            "conf_inset": self._cfg.conf_inset,
            "conf_balloon": self._cfg.conf_balloon,
            "architecture": "AR-compliant-robust"
        }
    
    def detect_panels(self, qimage: QImage, page_size_or_dpi = None) -> List[QRectF]:
        """
        Détecte les panels avec architecture AR-compliant.
        
        AR Requirements:
        - Chaque page non vide renvoie au moins 1 panel/panel_inset
        - Retry automatique si 0 détection
        - Suppression boîtes > 88% si autres existent
        - Suppression boîtes < 0.2%
        - Logs avec préfixe [Panels]
        """
        # Passer None pour img_size car on n'a pas cette info dans detect_panels
        panels, _ = self.detect_panels_and_balloons(qimage, page_size_or_dpi, None)
        return panels
    
    def detect_balloons(self, qimage: QImage, page_size_or_dpi = None) -> List[QRectF]:
        """Détecte uniquement les balloons."""
        # Passer None pour img_size car on n'a pas cette info dans detect_balloons
        _, balloons = self.detect_panels_and_balloons(qimage, page_size_or_dpi, None)
        return balloons
    
    def detect_panels_and_balloons(self, qimage: QImage, page_size_or_dpi = None, img_size = None) -> tuple[List[QRectF], List[QRectF]]:
        """
        Détecte les panels avec architecture AR-compliant.
        
        Args:
            qimage: Image source
            page_size_or_dpi: Taille page PDF (QSizeF) ou DPI (int/float) 
            img_size: Taille de l'image rendue (QSize), optionnel
        
        AR Requirements:
        - Chaque page non vide renvoie au moins 1 panel/panel_inset
        - Retry automatique si 0 détection
        - Suppression boîtes > 88% si autres existent
        - Suppression boîtes < 0.2%
        - Logs avec préfixe [Panels]
        """
        log.info("[Panels] 🔥 RobustYoloDetector.detect_panels() - AR-COMPLIANT EN ACTION!")
        log.info(f"[Panels] 📥 Input QImage: {qimage.width()}x{qimage.height()}, format={qimage.format()}")
        log.info(f"[Panels] 📥 page_size_or_dpi parameter: {page_size_or_dpi}")
        
        try:
            # 1) Conversion QImage -> RGB AR-sécurisée
            log.info("[Panels] 🔄 Converting QImage to RGB array...")
            img_rgb = qimage_to_rgb_array(qimage)
            h, w = img_rgb.shape[:2]
            log.info(f"[Panels] ✅ RGB array created: shape={img_rgb.shape}, dtype={img_rgb.dtype}")
            
            # 2) Estimation zone de contenu (AR: zone > 50% d'encre)
            log.info("[Panels] 🔍 Estimating content size...")
            content_size = self._estimate_content_size(img_rgb)
            log.info(f"[Panels] 📏 Image: {w}x{h}, content: {content_size[0]}x{content_size[1]}")
            
            # 3) Cache key selon AR avec version de signature
            log.info("[Panels] 🔑 Building cache key...")
            cache_key = self._build_cache_key(qimage, content_size)
            # Ajouter version pour invalider cache lors de changements de signature
            cache_key += "_v2_precise_coords"  # Force l'invalidation pour la nouvelle conversion précise
            log.info(f"[Panels] 🔑 Cache key: {cache_key[:100]}...")  # Truncate for readability
            
            if cache_key in self._cache:
                cached_dets = self._cache[cache_key]
                log.info(f"[Panels] 💾 Cache hit! Found {len(cached_dets)} cached detections")
                if len(cached_dets) > 0:  # AR: jamais de cache hit vide
                    log.info(f"[Panels] ensure_panels_for: panels={len(cached_dets)} (cache hit)")
                    panels, balloons = self._dets_to_qrect_separated(cached_dets, page_size_or_dpi, img_size)
                    log.info(f"[Panels] ✅ Returning {len(panels)} panels + {len(balloons)} balloons from cache")
                    return panels, balloons
                else:
                    log.info("[Panels] ⚠️ cached empty avoided (legacy). Recompute…")
            else:
                log.info("[Panels] 💾 Cache miss - will compute detections")
            
            log.info("[Panels] ensure_panels_for: (cache miss)")
            
            # 4) Détection primaire AVEC balloons
            log.info("[Panels] 🎯 Starting primary detection (panels + balloons)...")
            log.info(f"[Panels] 🎯 Detection config: conf_panel={self._cfg.conf_panel}, conf_inset={self._cfg.conf_inset}, conf_balloon={self._cfg.conf_balloon}")
            dets = self._detector.detect_all_classes(img_rgb, content_size=content_size)
            log.info(f"[Panels] 🎯 Primary detection result: {len(dets)} detections")
            
            # 5) AR: Retry si vide
            if not dets:
                retry_conf = max(self._cfg.conf_panel * 0.8, 0.12)
                retry_iou = min(self._cfg.iou + 0.05, 0.70)
                log.info(f"[Panels] ⛑ 0 panels → retry with softer thresholds: conf={retry_conf}, iou={retry_iou}")
                dets = self._detector.detect_all_classes(
                    img_rgb, 
                    content_size=content_size,
                    override_conf=retry_conf,  # AR: conf -20%, min 0.12
                    override_iou=retry_iou,    # AR: iou +0.05, max 0.70
                    tta=False
                )
                log.info(f"[Panels] ⛑ Retry detection result: {len(dets)} detections")
            
            # 6) Tri ordre de lecture (AR: haut→bas, gauche→droite)
            if dets:
                log.info("[Panels] � Merging overlapping panels...")
                dets = self._merge_overlapping_panels(dets, iou_threshold=0.3)
                log.info(f"[Panels] 🔗 After merge: {len(dets)} panels")
                
                log.info("[Panels] �📚 Sorting detections in reading order...")
                dets = self._sort_reading_order(dets)
                log.info(f"[Panels] 📚 Sorted {len(dets)} detections")
                
                # 7) Cache non-vide (AR)
                log.info("[Panels] 💾 Caching non-empty results...")
                self._cache[cache_key] = dets
                log.info(f"[Panels] ensure_panels_for: panels={len(dets)}")
            else:
                log.info("[Panels] 🚫 Not caching empty result")
            
            # 8) Conversion QRectF avec séparation panels/balloons
            log.info(f"[Panels] 🔄 Converting {len(dets)} detections to QRectF...")
            panels, balloons = self._dets_to_qrect_separated(dets, page_size_or_dpi, img_size)
            log.info(f"[Panels] ✅ detect_panels_and_balloons() returning {len(panels)} panels + {len(balloons)} balloons")
            return panels, balloons
            
        except Exception as e:
            log.error(f"[Panels] ❌ Erreur détection AR-compliant: {e}")
            import traceback
            log.error(f"[Panels] ❌ Traceback:\n{traceback.format_exc()}")
            return [], []
    
    def _estimate_content_size(self, img_rgb: np.ndarray) -> Tuple[int, int]:
        """Estime la zone de contenu (AR: zone > 50% d'encre)."""
        h, w = img_rgb.shape[:2]
        
        # Conversion grayscale simple pour estimation
        gray = np.mean(img_rgb, axis=2)
        
        # Seuillage grossier (pixels non-blancs)
        content_mask = gray < 240  # Pixels "non-blancs"
        
        if not content_mask.any():
            return (w, h)  # Fallback si tout blanc
        
        # Bounding box du contenu
        rows = np.any(content_mask, axis=1)
        cols = np.any(content_mask, axis=0)
        
        if not rows.any() or not cols.any():
            return (w, h)
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        content_w = max(1, cmax - cmin + 1)
        content_h = max(1, rmax - rmin + 1)
        
        # AR: zone > 50% d'encre
        return (content_w, content_h)
    
    def _build_cache_key(self, qimage: QImage, content_size: Tuple[int, int]) -> str:
        """Construit clé cache AR-compliant."""
        # AR: clé inclut DPI, modèle, paramètres + VERSION DPI
        key_parts = [
            f"robust_yolo",
            f"dpi_{self.render_dpi}",
            f"dpi_conversion_v3",  # VERSION: fix drop_fullpage_ratio pour pages mixtes
            f"model_multibd_enhanced_v2",
            f"imgsz_{self._cfg.imgsz}",
            f"iou_{self._cfg.iou}",
            f"conf_p_{self._cfg.conf_panel}",
            f"conf_i_{self._cfg.conf_inset}",
            f"conf_b_{self._cfg.conf_balloon}",
            f"content_{content_size[0]}x{content_size[1]}",
            f"img_{qimage.width()}x{qimage.height()}"
        ]
        return "_".join(key_parts)
    
    def _merge_overlapping_panels(self, dets: List[Dict[str, Any]], iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Post-processing pour merger les panels qui se chevauchent.
        Évite la fragmentation excessive du modèle.
        """
        if len(dets) <= 1:
            return dets
        
        log.info(f"[Panels] 🔗 Merging overlapping panels (IoU threshold={iou_threshold})")
        
        # Convertir en format plus facile à manipuler
        boxes = []
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            boxes.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'conf': det['conf'],
                'name': det['name'],
                'original': det
            })
        
        # Trier par confidence (garder les meilleurs)
        boxes.sort(key=lambda x: x['conf'], reverse=True)
        
        merged = []
        used = set()
        
        for i, box1 in enumerate(boxes):
            if i in used:
                continue
                
            # Chercher toutes les boxes qui se chevauchent avec box1
            to_merge = [box1]
            used.add(i)
            
            for j, box2 in enumerate(boxes[i+1:], i+1):
                if j in used:
                    continue
                    
                # Calcul IoU
                inter_x1 = max(box1['x1'], box2['x1'])
                inter_y1 = max(box1['y1'], box2['y1'])
                inter_x2 = min(box1['x2'], box2['x2'])
                inter_y2 = min(box1['y2'], box2['y2'])
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    
                    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
                    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
                    union_area = area1 + area2 - inter_area
                    
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > iou_threshold:
                        to_merge.append(box2)
                        used.add(j)
            
            # Merger toutes les boxes trouvées
            if len(to_merge) == 1:
                # Pas de merge nécessaire
                merged.append(to_merge[0]['original'])
            else:
                # Merger en prenant la bounding box englobante
                min_x1 = min(b['x1'] for b in to_merge)
                min_y1 = min(b['y1'] for b in to_merge)
                max_x2 = max(b['x2'] for b in to_merge)
                max_y2 = max(b['y2'] for b in to_merge)
                
                # Prendre la meilleure confidence
                best_conf = max(b['conf'] for b in to_merge)
                best_name = to_merge[0]['name']  # Garder le premier nom
                
                merged_det = {
                    'bbox': [min_x1, min_y1, max_x2, max_y2],
                    'conf': best_conf,
                    'name': best_name
                }
                merged.append(merged_det)
                
                log.info(f"[Panels] 🔗 Merged {len(to_merge)} panels into one")
        
        log.info(f"[Panels] 🔗 Merge result: {len(dets)} → {len(merged)} panels")
        return merged
    
    def _sort_reading_order(self, dets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tri ordre de lecture AR (haut→bas, gauche→droite)."""
        def _sort_key(d):
            bbox = d["bbox"]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            return (round(cy / 24), cx)  # bucket vertical pour éviter zigzag
        
        return sorted(dets, key=_sort_key)
    
    def _dets_to_qrect(self, dets: List[Dict[str, Any]]) -> List[QRectF]:
        """Conversion détections -> QRectF avec correction DPI."""
        log.info(f"[Panels] 🔄 _dets_to_qrect: Converting {len(dets)} detections to QRectF")
        
        # Conversion DPI: 130 DPI (image) -> 72 DPI (PDF points)
        scale_factor = 72.0 / self.render_dpi
        log.info(f"[Panels] 📏 DPI conversion factor: {self.render_dpi} -> 72 DPI = {scale_factor:.3f}")
        
        rects = []
        for i, det in enumerate(dets):
            log.info(f"[Panels] 🔄 Processing detection {i+1}: {det}")
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Conversion vers coordonnées PDF (72 DPI)
            x1_pdf = x1 * scale_factor
            y1_pdf = y1 * scale_factor
            x2_pdf = x2 * scale_factor  
            y2_pdf = y2 * scale_factor
            
            w_panel = x2_pdf - x1_pdf
            h_panel = y2_pdf - y1_pdf
            
            rect = QRectF(x1_pdf, y1_pdf, w_panel, h_panel)
            rects.append(rect)
            
            log.info(f"[Panels] 🎯 Panel {i+1}: {det['name']} conf={det['conf']:.3f}")
            log.info(f"[Panels] 📏   Image coords: ({x1:.0f},{y1:.0f},{x2-x1:.0f},{y2-y1:.0f})")
            log.info(f"[Panels] 📏   PDF coords: ({x1_pdf:.0f},{y1_pdf:.0f},{w_panel:.0f},{h_panel:.0f})")
        
        log.info(f"[Panels] ✅ _dets_to_qrect: Converted to {len(rects)} QRectF with DPI correction")
        return rects

    def _dets_to_qrect_separated(self, dets: List[Dict[str, Any]], page_size_or_dpi = None, img_size = None) -> tuple[List[QRectF], List[QRectF]]:
        """Conversion détections -> QRectF séparés panels/balloons."""
        print(f"🚨 _dets_to_qrect_separated APPELÉE: {len(dets)} détections")
        log.info(f"[Panels] 🔄 _dets_to_qrect_separated: Converting {len(dets)} detections")
        log.info(f"[Panels] 🔄 PARAMS: page_size_or_dpi={page_size_or_dpi}, img_size={img_size}")
        
        # Conversion avec prise en compte des vraies dimensions
        if page_size_or_dpi is not None and hasattr(page_size_or_dpi, 'width') and img_size is not None:
            # Conversion précise: coordonnées image -> coordonnées PDF
            page_w_pts = page_size_or_dpi.width()
            page_h_pts = page_size_or_dpi.height()
            img_w = img_size.width()
            img_h = img_size.height()
            
            # Facteurs de conversion directs
            scale_x = page_w_pts / img_w
            scale_y = page_h_pts / img_h
            
            print(f"🚨 CONVERSION PRÉCISE: page={page_w_pts}x{page_h_pts}, img={img_w}x{img_h}, scale={scale_x:.4f},{scale_y:.4f}")
            log.info(f"[Panels] 📏 CONVERSION PRÉCISE ACTIVÉE:")
            log.info(f"[Panels] 📏   Page PDF: {page_w_pts:.1f}x{page_h_pts:.1f} pts")  
            log.info(f"[Panels] 📏   Image: {img_w}x{img_h} px")
            log.info(f"[Panels] 📏   Scale: x={scale_x:.4f}, y={scale_y:.4f}")
        else:
            # Conversion DPI par défaut 
            scale_x = scale_y = 72.0 / self.render_dpi
            print(f"🚨 CONVERSION DPI PAR DÉFAUT: scale={scale_x:.3f}")
            log.info(f"[Panels] 📏 CONVERSION DPI PAR DÉFAUT: {self.render_dpi} -> 72 DPI = {scale_x:.3f}")
            log.info(f"[Panels] 📏 RAISON: page_size_or_dpi={page_size_or_dpi}, img_size={img_size}")
        
        panels = []
        balloons = []
        
        # Obtenir les dimensions de la page pour les filtres de position
        page_width = page_size_or_dpi.width() if (page_size_or_dpi and hasattr(page_size_or_dpi, 'width')) else None
        page_height = page_size_or_dpi.height() if (page_size_or_dpi and hasattr(page_size_or_dpi, 'width')) else None
        
        for i, det in enumerate(dets):
            log.info(f"[Panels] 🔄 Processing detection {i+1}: {det}")
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Conversion vers coordonnées PDF avec les facteurs appropriés
            x1_pdf = x1 * scale_x
            y1_pdf = y1 * scale_y
            x2_pdf = x2 * scale_x  
            y2_pdf = y2 * scale_y
            
            w_panel = x2_pdf - x1_pdf
            h_panel = y2_pdf - y1_pdf
            
            rect = QRectF(x1_pdf, y1_pdf, w_panel, h_panel)
            
            # Filtre de position pour balloons (éliminer ceux en dehors de la page)
            if det['name'] == 'balloon' and page_width and page_height:
                # Vérifier si le balloon est majoritairement dans les limites de la page
                center_x = x1_pdf + w_panel / 2
                center_y = y1_pdf + h_panel / 2
                margin = 20  # Marge de tolérance en points PDF
                
                if (center_x < -margin or center_x > page_width + margin or 
                    center_y < -margin or center_y > page_height + margin):
                    log.info(f"[Panels] ❌ Balloon {i+1} éliminé (hors page): center=({center_x:.0f},{center_y:.0f}) page=({page_width:.0f}x{page_height:.0f})")
                    continue
            
            # Séparation panels/balloons selon la classe
            if det['name'] == 'panel':
                panels.append(rect)
                log.info(f"[Panels] 🟢 Panel {len(panels)}: conf={det['conf']:.3f} rect=({x1_pdf:.0f},{y1_pdf:.0f},{w_panel:.0f},{h_panel:.0f})")
            elif det['name'] == 'balloon':
                balloons.append(rect)
                log.info(f"[Panels] 🔴 Balloon {len(balloons)}: conf={det['conf']:.3f} rect=({x1_pdf:.0f},{y1_pdf:.0f},{w_panel:.0f},{h_panel:.0f})")
            else:
                log.warning(f"[Panels] ⚠️ Unknown class: {det['name']}")
        
        log.info(f"[Panels] ✅ Separated: {len(panels)} panels + {len(balloons)} balloons")
        return panels, balloons
