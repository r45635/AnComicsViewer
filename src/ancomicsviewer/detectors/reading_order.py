# detectors/reading_order.py
from typing import List
from PySide6.QtCore import QRectF, QSizeF

def sort_reading_order(rects: List[QRectF], page_point_size: QSizeF, reading_rtl: bool = False, row_band_frac: float = 0.06) -> List[QRectF]:
    if not rects:
        return rects
    H = float(page_point_size.height()) or 1.0
    band = max(8.0, H * float(row_band_frac))
    rows = []
    for r in rects:
        y = r.top()
        for row in rows:
            if abs(y - row["y"]) <= band:
                row["items"].append(r)
                row["y"] = min(row["y"], y)
                break
        else:
            rows.append({"y": y, "items": [r]})
    rows.sort(key=lambda rr: rr["y"])
    out = []
    if reading_rtl:
        for row in rows:
            row["items"].sort(key=lambda rr: (-rr.left(), rr.top()))
            out.extend(row["items"])
    else:
        for row in rows:
            row["items"].sort(key=lambda rr: (rr.left(), rr.top()))
            out.extend(row["items"])
    return out
