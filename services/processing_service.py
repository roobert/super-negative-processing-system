"""
SUPER NEGATIVE PROCESSING SYSTEM - Processing Service

Qt-compatible coordinator for background batch processing.
Bridges ProcessPoolExecutor with Qt signals for UI updates.
"""

from PySide6.QtCore import QObject, Signal, QThread
from typing import List, Tuple, Dict, Any, Optional

from services.batch_processor import BatchProcessor, ProcessJob
from services.memory_manager import MemoryManager


class BatchWorker(QObject):
    """
    Worker that runs BatchProcessor in a QThread.

    This allows the batch processing to run without blocking the Qt event loop
    while still providing progress updates via signals.
    """

    # Signals for progress updates (thread-safe via Qt signal mechanism)
    progress = Signal(int, int, str)      # current, total, message
    itemComplete = Signal(int, object)    # index, result dict
    error = Signal(int, str)              # index, error message
    finished = Signal(int, int)           # success_count, total

    def __init__(self, jobs: List[ProcessJob], job_type: str):
        super().__init__()
        self._jobs = jobs
        self._job_type = job_type
        self._processor: Optional[BatchProcessor] = None

    def run(self):
        """Execute batch processing (called when thread starts)."""
        self._processor = BatchProcessor(
            on_progress=self._emit_progress,
            on_item_complete=self._emit_item_complete,
            on_error=self._emit_error,
            on_batch_complete=self._emit_finished,
        )
        self._processor.start(self._jobs, self._job_type)

    def _emit_progress(self, current: int, total: int, message: str):
        """Emit progress signal (safe from worker thread)."""
        self.progress.emit(current, total, message)

    def _emit_item_complete(self, index: int, result: Dict[str, Any]):
        """Emit item complete signal."""
        self.itemComplete.emit(index, result)

    def _emit_error(self, index: int, error_msg: str):
        """Emit error signal."""
        self.error.emit(index, error_msg)

    def _emit_finished(self, success_count: int, total: int):
        """Emit finished signal."""
        self.finished.emit(success_count, total)

    def cancel(self):
        """Cancel batch processing."""
        if self._processor:
            self._processor.cancel()


class ProcessingService(QObject):
    """
    Main service coordinating all background processing.

    Provides a simple interface for batch operations with Qt signal integration.
    Use this from the main window to start batch processing operations.
    """

    # Public signals for UI integration
    progressUpdated = Signal(int, int, str)   # current, total, message
    imageProcessed = Signal(int, object)      # index, result dict
    batchCompleted = Signal(int, int)         # success_count, total
    errorOccurred = Signal(int, str)          # index, error message

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self._worker: Optional[BatchWorker] = None
        self._thread: Optional[QThread] = None
        self._memory_manager = MemoryManager()

    def process_all_images(
        self,
        items: List[Tuple[int, str, str]],  # [(index, path, hash), ...]
        settings: Dict[str, Any],
        job_type: str
    ):
        """
        Start batch processing all images.

        Args:
            items: List of (index, path, image_hash) tuples
            settings: Settings dict passed to each processing function
            job_type: Type of processing ('full_process', 'auto_rotate', 'preset')
        """
        # Cancel any existing operation
        self.cancel()

        # Create jobs from items
        jobs = [
            ProcessJob(index=idx, path=path, image_hash=hash_ or "", settings=settings)
            for idx, path, hash_ in items
        ]

        if not jobs:
            self.batchCompleted.emit(0, 0)
            return

        # Create worker and thread
        self._thread = QThread()
        self._worker = BatchWorker(jobs, job_type)
        self._worker.moveToThread(self._thread)

        # Connect thread lifecycle
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)

        # Connect progress signals (forward to our public signals)
        self._worker.progress.connect(self.progressUpdated.emit)
        self._worker.itemComplete.connect(self.imageProcessed.emit)
        self._worker.error.connect(self.errorOccurred.emit)

        # Start processing
        self._thread.start()

    def auto_process_all(
        self,
        items: List[Tuple[int, str, str]],
        bg_threshold: int = 15,
        sensitivity: int = 50
    ):
        """
        Convenience method: auto-detect crop, rotation, base color for all images.

        Args:
            items: List of (index, path, image_hash) tuples
            bg_threshold: Background threshold setting
            sensitivity: Detection sensitivity setting
        """
        settings = {
            'bg_threshold': bg_threshold,
            'sensitivity': sensitivity,
            'auto_detect': True,
        }
        self.process_all_images(items, settings, 'full_process')

    def auto_rotate_all(self, items: List[Tuple[int, str, str]]):
        """
        Convenience method: detect optimal rotation for all images.

        Args:
            items: List of (index, path, image_hash) tuples
        """
        self.process_all_images(items, {}, 'auto_rotate')

    def apply_preset_to_all(
        self,
        items: List[Tuple[int, str, str]],
        preset: Dict[str, Any],
        preset_key: str
    ):
        """
        Convenience method: apply preset to all images.

        Args:
            items: List of (index, path, image_hash) tuples
            preset: Preset dict with 'adjustments' and 'curves'
            preset_key: Key identifying the preset
        """
        settings = {
            'preset': preset,
            'preset_key': preset_key,
        }
        self.process_all_images(items, settings, 'preset')

    def _on_finished(self, success_count: int, total: int):
        """Handle batch completion."""
        self.batchCompleted.emit(success_count, total)
        self._cleanup()

    def _cleanup(self):
        """Clean up thread and worker."""
        if self._thread:
            self._thread.quit()
            if not self._thread.wait(5000):  # 5 second timeout
                print("[ProcessingService] Thread did not quit cleanly")
                self._thread.terminate()
            self._thread = None
        self._worker = None

    def cancel(self):
        """Cancel all pending work."""
        if self._worker:
            self._worker.cancel()
        self._cleanup()

    def get_worker_count(self) -> int:
        """Get optimal worker count for display in UI."""
        return self._memory_manager.get_optimal_workers()

    def get_available_memory_gb(self) -> float:
        """Get available memory for display in UI."""
        return self._memory_manager.get_available_memory_gb()

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get full resource summary for display."""
        return self._memory_manager.get_resource_summary()

    @property
    def is_running(self) -> bool:
        """Check if a batch operation is currently running."""
        return self._thread is not None and self._thread.isRunning()
