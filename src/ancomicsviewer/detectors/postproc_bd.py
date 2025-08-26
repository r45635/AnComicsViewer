# postproc_bd.py
import numpy as np
import cv2
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def clip_to_page(dets: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Coupe les boîtes aux limites [0,w) x [0,h). Supprime celles d'aire <= 0.
    dets: (N,6) [x1,y1,x2,y2,score,cls]
    """
    if len(dets) == 0:
        return dets
    
    # Clip coordinates
    dets[:, 0] = np.clip(dets[:, 0], 0, w-1)  # x1
    dets[:, 1] = np.clip(dets[:, 1], 0, h-1)  # y1
    dets[:, 2] = np.clip(dets[:, 2], 0, w-1)  # x2
    dets[:, 3] = np.clip(dets[:, 3], 0, h-1)  # y2
    
    # Remove boxes with zero or negative area
    areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    valid_mask = areas > 0
    
    return dets[valid_mask]

def adaptive_conf_filter(dets: np.ndarray,
                         min_conf: float = 0.15,
                         base_conf: float = 0.25,
                         target_min: int = 3,
                         target_max: int = 24) -> np.ndarray:
    """
    Filtre par page, classe-aware (panel=0, inset=1).
    - Départ base_conf=0.25.
    - Si < target_min -> seuil = max(min_conf, 20e percentile des scores de la classe).
    - Si > target_max -> seuil = 75e percentile des scores de la classe.
    Retourne les dets >= seuil par classe.
    """
    if len(dets) == 0:
        return dets
    
    filtered_dets = []
    
    for cls_id in [0, 1]:  # panel=0, panel_inset=1
        cls_mask = dets[:, 5] == cls_id
        cls_dets = dets[cls_mask]
        
        if len(cls_dets) == 0:
            continue
            
        scores = cls_dets[:, 4]
        
        # Apply adaptive threshold based on count
        if len(cls_dets) < target_min:
            # Too few detections - lower threshold
            threshold = max(min_conf, float(np.percentile(scores, 20)))
        elif len(cls_dets) > target_max:
            # Too many detections - raise threshold
            threshold = float(np.percentile(scores, 75))
        else:
            # Good count - use base threshold
            threshold = base_conf
        
        # Filter by threshold
        valid_mask = scores >= threshold
        filtered_dets.append(cls_dets[valid_mask])
    
    if filtered_dets:
        return np.vstack(filtered_dets)
    else:
        return np.array([]).reshape(0, 6)

def estimate_content_bbox(img: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Retourne (x1,y1,x2,y2) englobant le 'contenu' de la page:
    - Convertir en Lab ou HSV (prendre L/V).
    - Calculer magnitude de gradient (Sobel).
    - Binaire par seuil d'Otsu sur (gradient > 0).
    - Morphologie (open/close) pour lisser.
    - BBox englobante de la plus grande composante.
    Si échec, retourner toute la page.
    """
    try:
        h, w = img.shape[:2]
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Convert to uint8 for Otsu
        grad_mag_uint8 = (grad_mag / grad_mag.max() * 255).astype(np.uint8)
        
        # Otsu threshold on gradient
        _, binary = cv2.threshold(grad_mag_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find largest connected component
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)
            return (x, y, x + w_box, y + h_box)
        else:
            # Fallback to full page
            return (0, 0, w, h)
            
    except Exception:
        # Fallback to full page on any error
        return (0, 0, img.shape[1], img.shape[0])

def drop_outside_content(dets: np.ndarray,
                         content_bbox: Tuple[int, int, int, int],
                         thr_ratio: float = 0.85) -> np.ndarray:
    """
    Supprime les boîtes dont >= thr_ratio de leur aire est hors de content_bbox.
    """
    if len(dets) == 0:
        return dets
    
    cx1, cy1, cx2, cy2 = content_bbox
    valid_dets = []
    
    for det in dets:
        x1, y1, x2, y2 = det[:4]
        
        # Calculate box area
        box_area = (x2 - x1) * (y2 - y1)
        if box_area <= 0:
            continue
        
        # Calculate intersection with content bbox
        ix1 = max(x1, cx1)
        iy1 = max(y1, cy1)
        ix2 = min(x2, cx2)
        iy2 = min(y2, cy2)
        
        if ix1 < ix2 and iy1 < iy2:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            overlap_ratio = intersection_area / box_area
        else:
            overlap_ratio = 0.0
        
        # Keep if enough overlap with content
        if overlap_ratio >= (1.0 - thr_ratio):
            valid_dets.append(det)
    
    if valid_dets:
        return np.array(valid_dets)
    else:
        return np.array([]).reshape(0, 6)

def size_aspect_priors(dets: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Supprime:
    - Aire < 0.02 * aire_page
    - Aspect ratio < 0.2 ou > 5.0
    Retourne le reste.
    """
    if len(dets) == 0:
        return dets
    
    page_area = h * w
    min_area = 0.02 * page_area
    
    valid_dets = []
    
    for det in dets:
        x1, y1, x2, y2 = det[:4]
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        if area < min_area:
            continue
        
        aspect_ratio = width / max(height, 1e-6)
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            continue
        
        valid_dets.append(det)
    
    if valid_dets:
        return np.array(valid_dets)
    else:
        return np.array([]).reshape(0, 6)

def _calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / max(union, 1e-6)

def class_aware_wbf(dets: np.ndarray, iou: float = 0.65, min_votes: int = 1) -> np.ndarray:
    """
    Weighted Box Fusion séparée par classe (panel / inset).
    - Grouper des boîtes IoU>=iou, même classe.
    - Coordonnées = moyenne pondérée par score.
    - Score = moyenne des scores.
    - Ignorer un groupe si nb_boîtes < min_votes.
    """
    if len(dets) == 0:
        return dets
    
    fused_dets = []
    
    for cls_id in [0, 1]:  # panel=0, panel_inset=1
        cls_mask = dets[:, 5] == cls_id
        cls_dets = dets[cls_mask]
        
        if len(cls_dets) == 0:
            continue
        
        if len(cls_dets) == 1:
            fused_dets.append(cls_dets[0])
            continue
        
        # Group boxes by IoU
        used = np.zeros(len(cls_dets), dtype=bool)
        
        for i in range(len(cls_dets)):
            if used[i]:
                continue
            
            # Find all boxes with IoU >= threshold
            group_indices = [i]
            used[i] = True
            
            for j in range(i + 1, len(cls_dets)):
                if used[j]:
                    continue
                
                iou_val = _calculate_iou(cls_dets[i][:4], cls_dets[j][:4])
                if iou_val >= iou:
                    group_indices.append(j)
                    used[j] = True
            
            # Apply WBF if group has enough votes
            if len(group_indices) >= min_votes:
                group_boxes = cls_dets[group_indices]
                scores = group_boxes[:, 4]
                
                # Weighted average coordinates
                total_score = np.sum(scores)
                weighted_coords = np.sum(group_boxes[:, :4] * scores[:, np.newaxis], axis=0) / total_score
                
                # Average score
                avg_score = np.mean(scores)
                
                # Create fused detection
                fused_det = np.array([*weighted_coords, avg_score, cls_id])
                fused_dets.append(fused_det)
    
    if fused_dets:
        return np.array(fused_dets)
    else:
        return np.array([]).reshape(0, 6)

def nested_inset_rule(dets: np.ndarray) -> np.ndarray:
    """
    Si A est (presque) contenue dans B (IoU(A,B)>0.5 et aire(A)/aire(B) < 0.6):
    - B reste panel (cls=0).
    - A devient panel_inset (cls=1).
    - Si deux panels se contiennent beaucoup (doublons), garder la plus grande OU fusionner via WBF.
    Retourne dets mis à jour.
    """
    if len(dets) == 0:
        return dets
    
    dets_copy = dets.copy()
    
    for i in range(len(dets_copy)):
        for j in range(len(dets_copy)):
            if i == j:
                continue
            
            box_a = dets_copy[i][:4]
            box_b = dets_copy[j][:4]
            
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            
            iou_val = _calculate_iou(box_a, box_b)
            
            # Check if A is contained in B
            if iou_val > 0.5 and area_a / max(area_b, 1e-6) < 0.6:
                # A becomes inset, B stays panel
                dets_copy[i, 5] = 1  # panel_inset
                dets_copy[j, 5] = 0  # panel
    
    return dets_copy

def merge_collinear_strips(dets: np.ndarray, h: int, w: int,
                           axis: str, overlap: float = 0.85, gap: float = 0.02) -> np.ndarray:
    """
    Fusionne des panels adjacents représentant en fait une seule case:
    - axis='x': boîtes côte-à-côte: chevauchement vertical >= overlap et écart horizontal <= gap*largeur_page -> fusion rect englobant.
    - axis='y': idem à la verticale.
    Ne touche pas aux insets (cls=1).
    """
    if len(dets) == 0:
        return dets
    
    # Only process panels (cls=0)
    panel_mask = dets[:, 5] == 0
    panels = dets[panel_mask]
    insets = dets[~panel_mask]
    
    if len(panels) < 2:
        return dets
    
    merged_panels = []
    used = np.zeros(len(panels), dtype=bool)
    
    if axis == 'x':
        # Horizontal merging
        gap_pixels = gap * w
        
        for i in range(len(panels)):
            if used[i]:
                continue
            
            current_group = [i]
            used[i] = True
            
            for j in range(len(panels)):
                if used[j]:
                    continue
                
                # Check vertical overlap
                y1_i, y2_i = panels[i][1], panels[i][3]
                y1_j, y2_j = panels[j][1], panels[j][3]
                
                overlap_y = min(y2_i, y2_j) - max(y1_i, y1_j)
                min_height = min(y2_i - y1_i, y2_j - y1_j)
                
                if overlap_y / max(min_height, 1e-6) >= overlap:
                    # Check horizontal gap
                    x1_i, x2_i = panels[i][0], panels[i][2]
                    x1_j, x2_j = panels[j][0], panels[j][2]
                    
                    gap_x = min(abs(x2_i - x1_j), abs(x2_j - x1_i))
                    
                    if gap_x <= gap_pixels:
                        current_group.append(j)
                        used[j] = True
            
            # Merge group
            if len(current_group) > 1:
                group_boxes = panels[current_group]
                x1 = np.min(group_boxes[:, 0])
                y1 = np.min(group_boxes[:, 1])
                x2 = np.max(group_boxes[:, 2])
                y2 = np.max(group_boxes[:, 3])
                score = np.mean(group_boxes[:, 4])
                
                merged_panels.append([x1, y1, x2, y2, score, 0])
            else:
                merged_panels.append(panels[i])
    
    elif axis == 'y':
        # Vertical merging
        gap_pixels = gap * h
        
        for i in range(len(panels)):
            if used[i]:
                continue
            
            current_group = [i]
            used[i] = True
            
            for j in range(len(panels)):
                if used[j]:
                    continue
                
                # Check horizontal overlap
                x1_i, x2_i = panels[i][0], panels[i][2]
                x1_j, x2_j = panels[j][0], panels[j][2]
                
                overlap_x = min(x2_i, x2_j) - max(x1_i, x1_j)
                min_width = min(x2_i - x1_i, x2_j - x1_j)
                
                if overlap_x / max(min_width, 1e-6) >= overlap:
                    # Check vertical gap
                    y1_i, y2_i = panels[i][1], panels[i][3]
                    y1_j, y2_j = panels[j][1], panels[j][3]
                    
                    gap_y = min(abs(y2_i - y1_j), abs(y2_j - y1_i))
                    
                    if gap_y <= gap_pixels:
                        current_group.append(j)
                        used[j] = True
            
            # Merge group
            if len(current_group) > 1:
                group_boxes = panels[current_group]
                x1 = np.min(group_boxes[:, 0])
                y1 = np.min(group_boxes[:, 1])
                x2 = np.max(group_boxes[:, 2])
                y2 = np.max(group_boxes[:, 3])
                score = np.mean(group_boxes[:, 4])
                
                merged_panels.append([x1, y1, x2, y2, score, 0])
            else:
                merged_panels.append(panels[i])
    
    # Combine merged panels with insets
    result = []
    if merged_panels:
        result.extend(merged_panels)
    if len(insets) > 0:
        result.extend(insets.tolist())
    
    if result:
        return np.array(result)
    else:
        return np.array([]).reshape(0, 6)

def split_large_by_gutters(dets: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Pour chaque boîte (cls=panel) d'aire > 0.35 * aire_page:
      - Cropper l'image de la boîte.
      - Sur la luminance, détecter 'gouttières' via faible gradient / zones claires (morpho).
      - Projeter verticalement et horizontalement, chercher une 'vallée' longue:
        * vertical: vallée de largeur 1-3% de la largeur du crop et hauteur >= 80% -> split vertical.
        * horizontal: vallée de hauteur 1-3% et largeur >= 80% -> split horizontal.
      - Si trouvé, couper la boîte en 2 le long de la vallée (max 2 splits par boîte).
    Retourne la nouvelle liste de boîtes.
    """
    if len(dets) == 0:
        return dets
    
    h, w = img.shape[:2]
    page_area = h * w
    split_threshold = 0.35 * page_area
    
    result_dets = []
    
    for det in dets:
        x1, y1, x2, y2, score, cls = det
        
        # Only split panels (cls=0)
        if cls != 0:
            result_dets.append(det)
            continue
        
        box_area = (x2 - x1) * (y2 - y1)
        
        if box_area <= split_threshold:
            result_dets.append(det)
            continue
        
        # Crop image to box
        x1_int, y1_int = int(x1), int(y1)
        x2_int, y2_int = int(x2), int(y2)
        
        crop = img[y1_int:y2_int, x1_int:x2_int]
        if crop.size == 0:
            result_dets.append(det)
            continue
        
        crop_h, crop_w = crop.shape[:2]
        
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        else:
            gray_crop = crop.copy()
        
        # Try to find vertical split
        split_found = False
        splits = [det]
        
        try:
            # Vertical projection
            v_proj = gray_crop.mean(axis=0)  # type: ignore
            v_smooth = cv2.GaussianBlur(v_proj.reshape(1, -1), (1, 5), 0).flatten()
            
            # Find valleys (low intensity regions)
            valley_threshold = np.percentile(v_smooth, 25)
            valleys = v_smooth < valley_threshold
            
            # Find continuous valley regions
            valley_starts = []
            valley_ends = []
            in_valley = False
            
            for i, is_valley in enumerate(valleys):
                if is_valley and not in_valley:
                    valley_starts.append(i)
                    in_valley = True
                elif not is_valley and in_valley:
                    valley_ends.append(i)
                    in_valley = False
            
            if in_valley:
                valley_ends.append(len(valleys))
            
            # Check for valid vertical splits
            for start, end in zip(valley_starts, valley_ends):
                valley_width = end - start
                valley_rel_width = valley_width / crop_w
                
                if 0.01 <= valley_rel_width <= 0.03:  # 1-3% width
                    split_x = x1 + (start + end) // 2
                    
                    # Split into two boxes
                    left_box = [x1, y1, split_x, y2, score, cls]
                    right_box = [split_x, y1, x2, y2, score, cls]
                    
                    splits = [left_box, right_box]
                    split_found = True
                    break
            
            # If no vertical split, try horizontal
            if not split_found:
                h_proj = gray_crop.mean(axis=1)  # type: ignore
                h_smooth = cv2.GaussianBlur(h_proj.reshape(-1, 1), (5, 1), 0).flatten()
                
                valley_threshold = np.percentile(h_smooth, 25)
                valleys = h_smooth < valley_threshold
                
                valley_starts = []
                valley_ends = []
                in_valley = False
                
                for i, is_valley in enumerate(valleys):
                    if is_valley and not in_valley:
                        valley_starts.append(i)
                        in_valley = True
                    elif not is_valley and in_valley:
                        valley_ends.append(i)
                        in_valley = False
                
                if in_valley:
                    valley_ends.append(len(valleys))
                
                for start, end in zip(valley_starts, valley_ends):
                    valley_height = end - start
                    valley_rel_height = valley_height / crop_h
                    
                    if 0.01 <= valley_rel_height <= 0.03:  # 1-3% height
                        split_y = y1 + (start + end) // 2
                        
                        # Split into two boxes
                        top_box = [x1, y1, x2, split_y, score, cls]
                        bottom_box = [x1, split_y, x2, y2, score, cls]
                        
                        splits = [top_box, bottom_box]
                        break
        
        except Exception:
            # On error, keep original box
            pass
        
        result_dets.extend(splits)
    
    if result_dets:
        return np.array(result_dets)
    else:
        return np.array([]).reshape(0, 6)

def final_class_aware_nms(dets: np.ndarray, iou: float = 0.60) -> np.ndarray:
    """
    NMS appliqué séparément par classe avec IoU donné.
    Garde la boîte au score max par cluster.
    """
    if len(dets) == 0:
        return dets
    
    final_dets = []
    
    for cls_id in [0, 1]:  # panel=0, panel_inset=1
        cls_mask = dets[:, 5] == cls_id
        cls_dets = dets[cls_mask]
        
        if len(cls_dets) == 0:
            continue
        
        # Sort by score (descending)
        sorted_indices = np.argsort(cls_dets[:, 4])[::-1]
        cls_dets_sorted = cls_dets[sorted_indices]
        
        keep = []
        for i in range(len(cls_dets_sorted)):
            should_keep = True
            
            for j in keep:
                iou_val = _calculate_iou(cls_dets_sorted[i][:4], cls_dets_sorted[j][:4])
                if iou_val >= iou:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(i)
        
        final_dets.extend(cls_dets_sorted[keep].tolist())
    
    if final_dets:
        return np.array(final_dets)
    else:
        return np.array([]).reshape(0, 6)
