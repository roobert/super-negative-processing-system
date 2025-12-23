"""
SUPER NEGATIVE PROCESSING SYSTEM - Hash Worker

Background file hashing using ThreadPoolExecutor (I/O bound operations).
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from PySide6.QtCore import QObject, Signal, QThread


class HashWorker(QObject):
    """
    Worker for computing file hashes in background.

    Uses ThreadPoolExecutor since file hashing is I/O bound
    (doesn't benefit from ProcessPool, and threads are lighter weight).
    """

    # Signals
    hashComputed = Signal(str, str)    # path, hash
    progress = Signal(int, int)        # current, total
    finished = Signal(dict)            # {path: hash}
    error = Signal(str, str)           # path, error_message

    # Number of threads for I/O operations (disk is the bottleneck, not CPU)
    IO_THREADS = 4

    def __init__(self, paths: List[str]):
        super().__init__()
        self._paths = paths
        self._cancelled = False

    def run(self):
        """Compute hashes using thread pool."""
        results: Dict[str, str] = {}
        total = len(self._paths)

        if total == 0:
            self.finished.emit(results)
            return

        with ThreadPoolExecutor(max_workers=self.IO_THREADS) as executor:
            # Submit all hash jobs
            futures = {
                executor.submit(self._hash_file, path): path
                for path in self._paths
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(futures)):
                if self._cancelled:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

                path = futures[future]
                try:
                    file_hash = future.result(timeout=60)  # 1 minute timeout per file
                    results[path] = file_hash
                    self.hashComputed.emit(path, file_hash)
                except Exception as e:
                    results[path] = ""  # Empty hash on error
                    self.error.emit(path, str(e))

                self.progress.emit(i + 1, total)

        self.finished.emit(results)

    def _hash_file(self, path: str) -> str:
        """
        Compute SHA-256 hash with chunked reading.

        Uses 64KB chunks to balance memory usage and I/O efficiency.
        """
        hasher = hashlib.sha256()
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b''):
                    if self._cancelled:
                        return ""
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            raise IOError(f"Failed to hash {path}: {e}")

    def cancel(self):
        """Cancel hash computation."""
        self._cancelled = True


class HashService(QObject):
    """
    Service for background file hashing with Qt integration.

    Provides a simple interface to compute hashes asynchronously
    without blocking the UI thread.
    """

    # Public signals (forwarded from worker)
    hashComputed = Signal(str, str)    # path, hash
    progress = Signal(int, int)        # current, total
    allCompleted = Signal(dict)        # {path: hash}
    error = Signal(str, str)           # path, error_message

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self._worker: Optional[HashWorker] = None
        self._thread: Optional[QThread] = None

    def compute_hashes_async(self, paths: List[str]):
        """
        Start background hash computation.

        Args:
            paths: List of file paths to hash
        """
        # Cancel any existing operation
        self.cancel()

        if not paths:
            self.allCompleted.emit({})
            return

        # Create worker and thread
        self._thread = QThread()
        self._worker = HashWorker(paths)
        self._worker.moveToThread(self._thread)

        # Connect thread lifecycle
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)

        # Forward signals
        self._worker.hashComputed.connect(self.hashComputed.emit)
        self._worker.progress.connect(self.progress.emit)
        self._worker.error.connect(self.error.emit)

        # Start
        self._thread.start()

    def _on_finished(self, results: Dict[str, str]):
        """Handle completion."""
        self.allCompleted.emit(results)
        self._cleanup()

    def _cleanup(self):
        """Clean up thread and worker."""
        if self._thread:
            self._thread.quit()
            if not self._thread.wait(2000):  # 2 second timeout
                self._thread.terminate()
            self._thread = None
        self._worker = None

    def cancel(self):
        """Cancel hash computation."""
        if self._worker:
            self._worker.cancel()
        self._cleanup()

    @property
    def is_running(self) -> bool:
        """Check if hash computation is running."""
        return self._thread is not None and self._thread.isRunning()
