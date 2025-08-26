# ui/page_view.py
from PySide6 import QtCore, QtGui, QtWidgets

class PageView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing |
                            QtGui.QPainter.RenderHint.SmoothPixmapTransform |
                            QtGui.QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._page_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._overlay_items = []

    # ---------- page lifecycle ----------
    def clear_page(self):
        self._scene.clear()
        self._page_item = None
        self._overlay_items.clear()

    def show_qimage(self, qimg: QtGui.QImage):
        """Display EXACTLY the image used for inference (same W,H)."""
        self.clear_page()
        pix = QtGui.QPixmap.fromImage(qimg)
        pix.setDevicePixelRatio(1.0)  # Retina safety
        self._page_item = QtWidgets.QGraphicsPixmapItem(pix)
        self._scene.addItem(self._page_item)
        self.setSceneRect(self._page_item.boundingRect())
        self.fitInView(self._page_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    # ---------- overlays ----------
    def draw_detections(self, dets, colors=None, show_fullframe_debug=False):
        """
        dets: [{'x1':..,'y1':..,'x2':..,'y2':..,'cls':'panel','conf':0.95}, ...]
        Coordinates MUST be in the QImage/QPixmap space (pixels).
        """
        if not self._page_item:
            return
        colors = colors or {
            'panel': QtGui.QColor(  0, 200,  80, 80),
            'panel_inset': QtGui.QColor(255, 180,   0, 80),
            'balloon': QtGui.QColor( 30, 120, 255, 80),
            '_edge': QtGui.QColor(  0, 255,   0,255)
        }
        if show_fullframe_debug:
            W = self._page_item.pixmap().width()
            H = self._page_item.pixmap().height()
            dbg = QtWidgets.QGraphicsRectItem(0, 0, W, H, self._page_item)
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 40))
            pen.setCosmetic(True)
            dbg.setPen(pen)

        for d in dets:
            x1, y1, x2, y2 = float(d['x1']), float(d['y1']), float(d['x2']), float(d['y2'])
            cls = d.get('cls', 'panel')
            rect = QtCore.QRectF(x1, y1, x2-x1, y2-y1)
            item = QtWidgets.QGraphicsRectItem(rect, self._page_item)  # <- parenting to pixmap!
            pen = QtGui.QPen(colors['_edge']); pen.setCosmetic(True); pen.setWidth(1)
            item.setPen(pen)
            item.setBrush(QtGui.QBrush(colors.get(cls, QtGui.QColor(0,255,0,60))))
            lbl = QtWidgets.QGraphicsSimpleTextItem(f"{cls} {d.get('conf',0):.2f}", item)
            lbl.setBrush(QtGui.QBrush(QtGui.QColor(255,255,255)))
            lbl.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            lbl.setPos(rect.topLeft() + QtCore.QPointF(2, 2))
            self._overlay_items.append(item)
