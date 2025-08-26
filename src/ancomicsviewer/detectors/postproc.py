from typing import List, Tuple
import numpy as np

# ------------------------------------------------------------
# Geometry & IoU
# ------------------------------------------------------------
def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    a_area = max(0.0, (ax2-ax1)) * max(0.0, (ay2-ay1))
    b_area = max(0.0, (bx2-bx1)) * max(0.0, (by2-by1))
    union = a_area + b_area - inter + 1e-12
    return inter / union

# ------------------------------------------------------------
# Weighted Boxes Fusion (tiny dependency-free version)
# ------------------------------------------------------------
def wbf_merge(
    boxes: List[np.ndarray],
    scores: List[np.ndarray],
    labels: List[np.ndarray],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.05,
):
    """
    boxes[t]: (N_t,4) absolute xyxy pixels   | scores[t]: (N_t,) | labels[t]: (N_t,)
    returns merged_boxes, merged_scores, merged_labels (np arrays)
    """
    if not boxes:
        return (np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int))

    all_b = np.concatenate(boxes, axis=0).astype(float)
    all_s = np.concatenate(scores, axis=0).astype(float)
    all_l = np.concatenate(labels, axis=0).astype(int)

    keep = all_s >= skip_box_thr
    all_b, all_s, all_l = all_b[keep], all_s[keep], all_l[keep]

    out_b, out_s, out_l = [], [], []
    for cls in np.unique(all_l):
        idx = np.where(all_l == cls)[0]
        b = all_b[idx]; s = all_s[idx]
        order = np.argsort(-s); b = b[order]; s = s[order]

        clusters, cluster_scores = [], []
        for box, sc in zip(b, s):
            placed = False
            for ci, cb in enumerate(clusters):
                if _iou_xyxy(box, cb) >= iou_thr:
                    w = cluster_scores[ci] + sc + 1e-9
                    clusters[ci] = (cb*cluster_scores[ci] + box*sc) / w
                    cluster_scores[ci] += sc
                    placed = True
                    break
            if not placed:
                clusters.append(box.copy())
                cluster_scores.append(sc)

        out_b.extend(clusters)
        out_s.extend([min(1.0, cs) for cs in cluster_scores])
        out_l.extend([cls]*len(clusters))

    return np.array(out_b), np.array(out_s), np.array(out_l, dtype=int)

# ------------------------------------------------------------
# Noise filtering & inset relabeling
# ------------------------------------------------------------
def filter_noise(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    page_w: int,
    page_h: int,
    min_conf: float = 0.15,
    min_rel_area: float = 0.008,
    max_rel_area: float = 0.95,
    max_aspect: float = 9.0,
):
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores, labels
    W, H = float(page_w), float(page_h)
    areas = (boxes_xyxy[:,2]-boxes_xyxy[:,0])*(boxes_xyxy[:,3]-boxes_xyxy[:,1])
    rel_areas = areas / (W*H + 1e-9)
    widths  = (boxes_xyxy[:,2]-boxes_xyxy[:,0])
    heights = (boxes_xyxy[:,3]-boxes_xyxy[:,1])
    aspect  = np.maximum(widths/heights, heights/widths)

    keep = (
        (scores >= min_conf) &
        (rel_areas >= min_rel_area) &
        (rel_areas <= max_rel_area) &
        (aspect <= max_aspect)
    )
    return boxes_xyxy[keep], scores[keep], labels[keep]

def classify_panels_and_insets(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    page_w: int,
    page_h: int,
    inset_ratio: float = 0.6,
):
    """
    Relabels as panel (0) vs panel_inset (1) using containment + area ratio.
    Balloons (label 2) pass through unchanged.
    """
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, scores, labels

    out_labels = labels.copy()
    areas = (boxes_xyxy[:,2]-boxes_xyxy[:,0])*(boxes_xyxy[:,3]-boxes_xyxy[:,1])

    for i in range(len(boxes_xyxy)):
        xi1, yi1, xi2, yi2 = boxes_xyxy[i]
        for j in range(len(boxes_xyxy)):
            if i == j: 
                continue
            xj1, yj1, xj2, yj2 = boxes_xyxy[j]
            if xi1 >= xj1 and yi1 >= yj1 and xi2 <= xj2 and yi2 <= yj2:
                if areas[i] / (areas[j] + 1e-9) <= inset_ratio:
                    out_labels[i] = 1  # inset
                else:
                    out_labels[i] = 0  # panel
                break
        if out_labels[i] not in (0,1,2):
            out_labels[i] = 0  # default -> panel
    return boxes_xyxy, scores, out_labels
