from typing import List
import numpy as np

def sort_reading_order(boxes_xyxy: np.ndarray, rtl: bool=False) -> np.ndarray:
    """
    Returns indices that sort panels row-by-row in reading order.
    """
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=int)

    y_centers = (boxes_xyxy[:,1] + boxes_xyxy[:,3]) / 2
    heights   = (boxes_xyxy[:,3] - boxes_xyxy[:,1])
    thr = np.median(heights) * 0.35  # tolerant row clustering

    order = np.argsort(y_centers)
    rows = []
    row_ids = np.zeros(len(boxes_xyxy), dtype=int)
    rid = 0; last_y = None
    for idx in order:
        yc = y_centers[idx]
        if last_y is None or abs(yc - last_y) > thr:
            rows.append([])
            rid += 1
            last_y = yc
        rows[-1].append(idx)
        row_ids[idx] = rid

    indices = []
    for r in sorted(set(row_ids)):
        row_idxs = np.where(row_ids == r)[0]
        xs = (boxes_xyxy[row_idxs,0] + boxes_xyxy[row_idxs,2]) / 2
        row_sorted = row_idxs[np.argsort(-xs if rtl else xs)]
        indices.extend(row_sorted.tolist())
    return np.array(indices, dtype=int)
