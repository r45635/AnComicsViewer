# utils/qimage_utils.py
import numpy as np
from PySide6 import QtGui

def qimage_to_numpy(qimg: QtGui.QImage) -> np.ndarray:
    """QImage (Format_ARGB32/RGB32/RGB888) -> numpy HxWx3 RGB uint8."""
    qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    # PySide6 compatible way to get the buffer
    buffer = memoryview(ptr).tobytes()
    arr = np.frombuffer(buffer, np.uint8).reshape(h, qimg.bytesPerLine())[:, :w*3]
    return arr.reshape(h, w, 3)
