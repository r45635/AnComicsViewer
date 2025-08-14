from typing import List
from PySide6.QtCore import QRectF, QSizeF
from PySide6.QtGui import QImage

class BasePanelDetector:
    reading_rtl: bool = False
    
    def detect_panels(self, qimage: QImage, page_point_size: QSizeF) -> List[QRectF]:
        raise NotImplementedError
