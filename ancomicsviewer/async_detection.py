"""Asynchronous panel detection worker.

Provides non-blocking panel detection with progress reporting.
"""

from __future__ import annotations

from typing import Optional, List, Callable
from dataclasses import dataclass
import traceback

from PySide6.QtCore import QObject, Signal, QThread, QMutex, QWaitCondition
from PySide6.QtCore import QRectF, QSizeF
from PySide6.QtGui import QImage

from .config import DetectorConfig
from .detector import PanelDetector


@dataclass
class DetectionTask:
    """A detection task to be processed."""
    page_num: int
    qimage: QImage
    page_point_size: QSizeF
    pdf_path: Optional[str] = None
    dpi: float = 150.0


@dataclass
class DetectionResult:
    """Result of a detection task."""
    page_num: int
    panels: List[QRectF]
    success: bool
    error_message: Optional[str] = None
    elapsed_ms: float = 0.0


class DetectionWorker(QObject):
    """Worker that runs panel detection in a background thread.
    
    Signals:
        started: Emitted when detection starts (page_num)
        progress: Emitted with progress updates (page_num, stage, percent)
        finished: Emitted when detection completes (result)
        error: Emitted on error (page_num, error_message)
    """
    
    started = Signal(int)  # page_num
    progress = Signal(int, str, int)  # page_num, stage, percent
    finished = Signal(object)  # DetectionResult
    error = Signal(int, str)  # page_num, error_message
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__()
        self._config = config or DetectorConfig()
        self._detector = PanelDetector(self._config)
        self._mutex = QMutex()
        self._abort = False
    
    def update_config(self, config: DetectorConfig) -> None:
        """Update detector configuration."""
        self._mutex.lock()
        try:
            self._config = config
            self._detector = PanelDetector(self._config)
        finally:
            self._mutex.unlock()
    
    def abort(self) -> None:
        """Request abortion of current detection."""
        self._mutex.lock()
        self._abort = True
        self._mutex.unlock()
    
    def _is_aborted(self) -> bool:
        self._mutex.lock()
        result = self._abort
        self._mutex.unlock()
        return result
    
    def detect(self, task: DetectionTask) -> None:
        """Run detection on a task (called from thread)."""
        import time
        start_time = time.time()
        
        self._mutex.lock()
        self._abort = False
        self._mutex.unlock()
        
        page_num = task.page_num
        self.started.emit(page_num)
        self.progress.emit(page_num, "Préparation", 10)
        
        try:
            if self._is_aborted():
                self.error.emit(page_num, "Détection annulée")
                return
            
            self.progress.emit(page_num, "Conversion image", 20)
            
            # Run detection
            self.progress.emit(page_num, "Détection des cases", 40)
            
            panels = self._detector.detect_panels(
                task.qimage,
                task.page_point_size,
                page_num=page_num,
                pdf_path=task.pdf_path,
                dpi=task.dpi,
            )
            
            if self._is_aborted():
                self.error.emit(page_num, "Détection annulée")
                return
            
            self.progress.emit(page_num, "Finalisation", 90)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            result = DetectionResult(
                page_num=page_num,
                panels=panels,
                success=True,
                elapsed_ms=elapsed_ms,
            )
            
            self.progress.emit(page_num, "Terminé", 100)
            self.finished.emit(result)
            
        except Exception as e:
            error_msg = f"Erreur: {str(e)}"
            self.error.emit(page_num, error_msg)
            
            result = DetectionResult(
                page_num=page_num,
                panels=[],
                success=False,
                error_message=error_msg,
            )
            self.finished.emit(result)


class AsyncDetectionManager(QObject):
    """Manager for asynchronous panel detection.
    
    Handles:
    - Background thread management
    - Task queuing
    - Result caching
    - Progress reporting
    
    Signals:
        detection_started: Emitted when detection starts (page_num)
        detection_progress: Emitted with progress (page_num, stage, percent)
        detection_finished: Emitted when detection completes (result)
        detection_error: Emitted on error (page_num, error_message)
    """
    
    detection_started = Signal(int)
    detection_progress = Signal(int, str, int)
    detection_finished = Signal(object)
    detection_error = Signal(int, str)
    
    def __init__(self, config: Optional[DetectorConfig] = None, parent=None):
        super().__init__(parent)
        self._config = config or DetectorConfig()
        
        # Worker and thread
        self._thread: Optional[QThread] = None
        self._worker: Optional[DetectionWorker] = None
        
        # Task queue
        self._pending_tasks: List[DetectionTask] = []
        self._current_task: Optional[DetectionTask] = None
        self._mutex = QMutex()
        
        self._setup_worker()
    
    def _setup_worker(self) -> None:
        """Setup background worker thread."""
        self._thread = QThread()
        self._worker = DetectionWorker(self._config)
        self._worker.moveToThread(self._thread)
        
        # Connect signals
        self._worker.started.connect(self.detection_started)
        self._worker.progress.connect(self.detection_progress)
        self._worker.finished.connect(self._on_detection_finished)
        self._worker.error.connect(self.detection_error)
        
        self._thread.start()
    
    def update_config(self, config: DetectorConfig) -> None:
        """Update detector configuration."""
        self._config = config
        if self._worker:
            self._worker.update_config(config)
    
    def request_detection(self, task: DetectionTask) -> None:
        """Request detection for a page.
        
        If already processing, queues the task.
        If same page is already queued, replaces it.
        
        Args:
            task: Detection task to process
        """
        self._mutex.lock()
        try:
            # Remove existing task for same page
            self._pending_tasks = [t for t in self._pending_tasks if t.page_num != task.page_num]
            
            # Add new task
            self._pending_tasks.append(task)
            
            # Start if not busy
            if self._current_task is None:
                self._process_next()
        finally:
            self._mutex.unlock()
    
    def cancel_detection(self, page_num: int) -> None:
        """Cancel detection for a specific page.
        
        Args:
            page_num: Page number to cancel
        """
        self._mutex.lock()
        try:
            # Remove from queue
            self._pending_tasks = [t for t in self._pending_tasks if t.page_num != page_num]
            
            # Abort if currently processing
            if self._current_task and self._current_task.page_num == page_num:
                if self._worker:
                    self._worker.abort()
        finally:
            self._mutex.unlock()
    
    def cancel_all(self) -> None:
        """Cancel all pending and current detections."""
        self._mutex.lock()
        try:
            self._pending_tasks.clear()
            if self._worker:
                self._worker.abort()
        finally:
            self._mutex.unlock()
    
    def _process_next(self) -> None:
        """Process next task in queue (called with mutex held)."""
        if not self._pending_tasks:
            self._current_task = None
            return
        
        self._current_task = self._pending_tasks.pop(0)
        
        # Run detection in worker thread
        if self._worker:
            # Use QMetaObject.invokeMethod for thread-safe call
            from PySide6.QtCore import QMetaObject, Qt, Q_ARG
            QMetaObject.invokeMethod(
                self._worker, 
                "detect", 
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(object, self._current_task)
            )
    
    def _on_detection_finished(self, result: DetectionResult) -> None:
        """Handle detection completion."""
        self._mutex.lock()
        try:
            self._current_task = None
            self._process_next()
        finally:
            self._mutex.unlock()
        
        self.detection_finished.emit(result)
    
    def shutdown(self) -> None:
        """Shutdown worker thread."""
        if self._worker:
            self._worker.abort()
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
    
    @property
    def is_busy(self) -> bool:
        """Check if currently processing a task."""
        self._mutex.lock()
        busy = self._current_task is not None
        self._mutex.unlock()
        return busy
    
    @property
    def pending_count(self) -> int:
        """Number of pending tasks."""
        self._mutex.lock()
        count = len(self._pending_tasks)
        self._mutex.unlock()
        return count
