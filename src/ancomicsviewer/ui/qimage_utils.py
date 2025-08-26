import numpy as np
from PySide6.QtGui import QImage

def qimage_to_rgb_np(qimg: QImage) -> np.ndarray:
    """QImage (tous formats) -> np.ndarray RGB uint8 contiguë"""
    if qimg.isNull():
        raise ValueError("QImage is null")
    if qimg.format() != QImage.Format.Format_RGBA8888:
        qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    bpl = qimg.bytesPerLine()
    ptr = qimg.constBits()
    # Fix pour Python : pas de setsize, on utilise bytes directement
    buffer_size = bpl * h
    buffer = bytes(ptr)[:buffer_size]
    arr = np.frombuffer(buffer, dtype=np.uint8).reshape(h, bpl)[:, :w*4]
    rgba = arr.reshape(h, w, 4)
    rgb = rgba[:, :, :3]  # drop alpha
    return np.ascontiguousarray(rgb)  # IMPORTANT
