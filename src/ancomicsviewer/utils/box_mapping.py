# utils/box_mapping.py
def ultra_yolobox_to_display(x1, y1, x2, y2, W, H, S=1280):
    """
    Map Ultralytics YOLO xyxy (on letterboxed SxS image) back to original W×H.
    """
    r = min(S / W, S / H)
    nw, nh = round(W * r), round(H * r)
    pad_x = (S - nw) / 2
    pad_y = (S - nh) / 2
    ox1 = (x1 - pad_x) / r
    oy1 = (y1 - pad_y) / r
    ox2 = (x2 - pad_x) / r
    oy2 = (y2 - pad_y) / r
    ox1 = max(0, min(W, ox1)); ox2 = max(0, min(W, ox2))
    oy1 = max(0, min(H, oy1)); oy2 = max(0, min(H, oy2))
    return ox1, oy1, ox2, oy2
