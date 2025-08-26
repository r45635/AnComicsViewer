#!/usr/bin/env python3
"""Debug spécifique du pipeline _predict_raw"""

import sys
sys.path.insert(0, '.')

import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def debug_predict_raw():
    """Debug step-by-step de _predict_raw"""
    
    from src.ancomicsviewer.detectors.multibd_detector import MultiBDPanelDetector
    
    # Image de test simple
    img = np.ones((800, 600, 3), dtype=np.uint8) * 200
    img[100:300, 100:250] = [255, 255, 255]  # Panel simple
    
    detector = MultiBDPanelDetector(device='cpu')
    detector._ensure_model_loaded()
    
    logger.info("=== DEBUG _predict_raw STEP BY STEP ===")
    
    # Test direct modèle
    logger.info("1. Test direct modèle...")
    results = detector.model.predict(
        img, 
        imgsz=1280,
        conf=0.15,  # CONF_BASE
        iou=0.5,
        device=detector.device,
        agnostic_nms=False,
        augment=False,
        max_det=300,
        classes=None,
        verbose=False
    )
    
    if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
        raw_count = int(results[0].boxes.cls.shape[0]) if results[0].boxes.cls is not None else 0
        logger.info(f"Détections brutes: {raw_count}")
        
        if raw_count > 0:
            # Vérifier les types
            logger.info("2. Types des données...")
            logger.info(f"   Type boxes: {type(results[0].boxes.xyxy)}")
            logger.info(f"   Type conf: {type(results[0].boxes.conf)}")
            logger.info(f"   Type cls: {type(results[0].boxes.cls)}")
            
            # Model names
            logger.info("3. Noms du modèle...")
            if hasattr(detector.model, 'names'):
                logger.info(f"   model.names: {detector.model.names}")
                logger.info(f"   Type: {type(detector.model.names)}")
            
            # Conversion
            logger.info("4. Conversion...")
            try:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                labels = results[0].boxes.cls.cpu().numpy().astype(int)
                
                logger.info(f"   Boxes shape: {boxes.shape}")
                logger.info(f"   Scores: {scores}")
                logger.info(f"   Labels: {labels}")
                
                # Mapping noms
                logger.info("5. Mapping noms...")
                id2name = {}
                if hasattr(detector.model, 'names'):
                    names = detector.model.names
                    if isinstance(names, dict):
                        id2name = {int(k): str(v) for k, v in names.items()}
                    elif isinstance(names, (list, tuple)):
                        id2name = {i: str(n) for i, n in enumerate(names)}
                
                logger.info(f"   id2name: {id2name}")
                
                # Noms pour chaque détection
                logger.info("6. Noms des détections...")
                for i, label in enumerate(labels):
                    name = id2name.get(label, f"unknown_{label}")
                    logger.info(f"   Det {i}: class={label} -> '{name}'")
                
                # Filtrage
                logger.info("7. Filtrage...")
                ACCEPT_CLASSES = {"panel", "panel_inset"}
                
                def _norm_name(n: str) -> str:
                    if n is None:
                        return ""
                    return n.strip().lower().replace(" ", "_").replace("-", "_")
                
                names_arr = np.array([_norm_name(id2name.get(i, "")) for i in labels])
                logger.info(f"   Noms normalisés: {names_arr}")
                
                normalized_accept = [_norm_name(c) for c in ACCEPT_CLASSES]
                logger.info(f"   Classes acceptées (normalisées): {normalized_accept}")
                
                keep = np.isin(names_arr, normalized_accept)
                logger.info(f"   Keep mask: {keep}")
                logger.info(f"   Nb gardées: {np.sum(keep)}")
                
            except Exception as e:
                logger.error(f"Erreur conversion: {e}")
                import traceback
                traceback.print_exc()
    
    # Test pipeline complet
    logger.info("8. Test pipeline _predict_raw...")
    result = detector._predict_raw(img)
    logger.info(f"   Résultat final: {len(result)} détections")

if __name__ == "__main__":
    debug_predict_raw()
