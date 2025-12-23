"""
SUPER NEGATIVE PROCESSING SYSTEM - Memory Manager

Monitor system memory and calculate optimal worker/batch configurations.
"""

import multiprocessing
from typing import Tuple

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class MemoryManager:
    """Monitor memory and calculate optimal batch sizes for image processing."""

    # float32 BGR image: 3 channels × 4 bytes per float
    BYTES_PER_PIXEL = 12

    # Reserve this much RAM for system/Qt/other processes
    SAFETY_MARGIN_GB = 2.0

    # Estimated peak memory per worker process (includes OpenCV buffers, etc.)
    MEMORY_PER_WORKER_GB = 0.65

    # Maximum workers regardless of resources (diminishing returns beyond this)
    MAX_WORKERS = 8

    # Minimum workers to maintain some parallelism
    MIN_WORKERS = 2

    def get_cpu_count(self) -> int:
        """Get number of CPU cores available."""
        return multiprocessing.cpu_count()

    def get_available_memory_gb(self) -> float:
        """Get current available memory in GB."""
        if HAS_PSUTIL:
            return psutil.virtual_memory().available / (1024 ** 3)
        else:
            # Fallback: assume 8GB available if psutil not installed
            return 8.0

    def get_total_memory_gb(self) -> float:
        """Get total system memory in GB."""
        if HAS_PSUTIL:
            return psutil.virtual_memory().total / (1024 ** 3)
        else:
            return 16.0

    def get_optimal_workers(self) -> int:
        """
        Calculate optimal number of worker processes.

        Considers both CPU cores and available memory to avoid:
        - Under-utilizing CPUs (too few workers)
        - Running out of memory (too many workers)
        """
        cpu_count = self.get_cpu_count()
        available_gb = self.get_available_memory_gb()

        # Leave one core for main thread/UI
        max_by_cpu = max(self.MIN_WORKERS, cpu_count - 1)

        # Calculate how many workers we can afford memory-wise
        usable_memory = available_gb - self.SAFETY_MARGIN_GB
        max_by_memory = max(1, int(usable_memory / self.MEMORY_PER_WORKER_GB))

        # Take the minimum of CPU and memory constraints, capped by MAX_WORKERS
        optimal = min(max_by_cpu, max_by_memory, self.MAX_WORKERS)

        return max(1, optimal)

    def get_batch_size(self, image_resolution: Tuple[int, int] = (6000, 4000)) -> int:
        """
        Calculate optimal batch size based on image dimensions.

        For very large images (100MP+), we may need smaller batches
        to avoid memory exhaustion even with fewer workers.

        Args:
            image_resolution: (width, height) of images being processed

        Returns:
            Number of images to process per batch
        """
        width, height = image_resolution
        pixels = width * height

        # Memory per image in MB (just the float32 array, not processing overhead)
        image_mb = (pixels * self.BYTES_PER_PIXEL) / (1024 ** 2)

        available_gb = self.get_available_memory_gb()

        # Target using 50% of available memory for the batch
        # This leaves headroom for processing overhead
        target_mb = (available_gb * 0.5) * 1024

        batch_size = max(1, int(target_mb / image_mb))

        # Don't exceed a reasonable batch size (processing in chunks helps with progress feedback)
        return min(batch_size, 20)

    def estimate_image_memory_mb(self, width: int, height: int) -> float:
        """Estimate memory required for a single image in MB."""
        pixels = width * height
        return (pixels * self.BYTES_PER_PIXEL) / (1024 ** 2)

    def can_process_image(self, width: int, height: int) -> bool:
        """Check if we have enough memory to process an image of given size."""
        required_mb = self.estimate_image_memory_mb(width, height)
        # Need at least 3x the image size for processing (input + output + working buffer)
        required_gb = (required_mb * 3) / 1024
        available_gb = self.get_available_memory_gb()
        return available_gb > (required_gb + self.SAFETY_MARGIN_GB)

    def get_resource_summary(self) -> dict:
        """Get a summary of available resources for display."""
        return {
            'cpu_count': self.get_cpu_count(),
            'total_memory_gb': round(self.get_total_memory_gb(), 1),
            'available_memory_gb': round(self.get_available_memory_gb(), 1),
            'optimal_workers': self.get_optimal_workers(),
            'has_psutil': HAS_PSUTIL,
        }
