"""
SUPER NEGATIVE PROCESSING SYSTEM - Batch Processor

Manages ProcessPoolExecutor for parallel image processing operations.
"""

import gc
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

from services.memory_manager import MemoryManager


@dataclass
class ProcessJob:
    """A single image processing job."""
    index: int
    path: str
    image_hash: str
    settings: Dict[str, Any]


class BatchProcessor:
    """
    Manages ProcessPoolExecutor for batch image operations.

    Uses memory-aware batching to avoid OOM on large image sets.
    All processing is done at full resolution for professional accuracy.
    """

    # Map job types to their processing functions
    JOB_FUNCTIONS = {
        'full_process': 'workers.image_processor.process_image_full',
        'auto_rotate': 'workers.image_processor.detect_rotation_parallel',
        'preset': 'workers.image_processor.apply_preset_to_image',
    }

    def __init__(
        self,
        on_progress: Callable[[int, int, str], None],
        on_item_complete: Callable[[int, Dict[str, Any]], None],
        on_error: Callable[[int, str], None],
        on_batch_complete: Callable[[int, int], None],
    ):
        """
        Initialize the batch processor.

        Args:
            on_progress: Callback for progress updates (current, total, message)
            on_item_complete: Callback when an item completes (index, result)
            on_error: Callback when an item fails (index, error_message)
            on_batch_complete: Callback when batch completes (success_count, total)
        """
        self._executor: Optional[ProcessPoolExecutor] = None
        self._cancelled = False
        self._on_progress = on_progress
        self._on_item_complete = on_item_complete
        self._on_error = on_error
        self._on_batch_complete = on_batch_complete
        self._memory_manager = MemoryManager()
        self._completed_count = 0
        self._error_count = 0

    def start(self, jobs: List[ProcessJob], job_type: str):
        """
        Start batch processing with optimal worker count.

        Args:
            jobs: List of ProcessJob instances to process
            job_type: Type of job ('full_process', 'auto_rotate', 'preset')
        """
        if job_type not in self.JOB_FUNCTIONS:
            raise ValueError(f"Unknown job type: {job_type}. Valid types: {list(self.JOB_FUNCTIONS.keys())}")

        # Get the processing function
        func = self._get_job_function(job_type)

        # Calculate optimal configuration
        workers = self._memory_manager.get_optimal_workers()
        batch_size = self._memory_manager.get_batch_size()

        self._cancelled = False
        self._completed_count = 0
        self._error_count = 0
        total = len(jobs)

        # Create executor with optimal worker count
        self._executor = ProcessPoolExecutor(max_workers=workers)

        try:
            # Process in memory-aware batches
            for batch_start in range(0, total, batch_size):
                if self._cancelled:
                    break

                batch_end = min(batch_start + batch_size, total)
                batch = jobs[batch_start:batch_end]

                self._process_batch(batch, func, total)

                # Explicit memory cleanup after each batch
                gc.collect()

        except Exception as e:
            # Log unexpected errors
            print(f"[BatchProcessor] Unexpected error: {e}")
        finally:
            self._shutdown_executor()

        # Report completion
        self._on_batch_complete(self._completed_count, total)

    def _get_job_function(self, job_type: str) -> Callable:
        """Import and return the processing function for a job type."""
        # Import dynamically to avoid circular imports and ensure picklability
        from workers.image_processor import (
            process_image_full,
            detect_rotation_parallel,
            apply_preset_to_image,
        )

        func_map = {
            'full_process': process_image_full,
            'auto_rotate': detect_rotation_parallel,
            'preset': apply_preset_to_image,
        }
        return func_map[job_type]

    def _process_batch(self, batch: List[ProcessJob], func: Callable, total: int):
        """Process a batch of jobs and handle results."""
        # Submit all jobs in batch
        futures: Dict[Future, ProcessJob] = {}
        for job in batch:
            if self._cancelled:
                break
            future = self._executor.submit(func, job.path, job.settings)
            futures[future] = job

        # Collect results as they complete
        for future in as_completed(futures):
            if self._cancelled:
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

            job = futures[future]
            try:
                result = future.result(timeout=300)  # 5 minute timeout per image
                result['image_hash'] = job.image_hash
                result['path'] = job.path
                self._on_item_complete(job.index, result)
                self._completed_count += 1
            except Exception as e:
                self._on_error(job.index, str(e))
                self._error_count += 1

            # Update progress
            processed = self._completed_count + self._error_count
            self._on_progress(processed, total, job.path)

    def _shutdown_executor(self):
        """Shut down the executor gracefully."""
        if self._executor:
            try:
                self._executor.shutdown(wait=not self._cancelled, cancel_futures=self._cancelled)
            except Exception:
                pass
            self._executor = None

    def cancel(self):
        """Cancel all pending work."""
        self._cancelled = True
        self._shutdown_executor()

    @property
    def is_cancelled(self) -> bool:
        """Check if processing was cancelled."""
        return self._cancelled

    @property
    def completed_count(self) -> int:
        """Get the number of successfully completed jobs."""
        return self._completed_count

    @property
    def error_count(self) -> int:
        """Get the number of jobs that failed."""
        return self._error_count

    def get_worker_count(self) -> int:
        """Get the number of workers that will be used."""
        return self._memory_manager.get_optimal_workers()
