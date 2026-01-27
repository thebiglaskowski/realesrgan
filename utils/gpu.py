"""GPU detection and management utilities"""

from __future__ import annotations
import torch
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class GPUDetector:
    """Detect and manage available GPUs"""

    @staticmethod
    def get_available_gpus() -> List[Tuple[int, str]]:
        """
        Returns list of (gpu_id, gpu_name) tuples

        Returns:
            List of tuples with GPU info
        """
        if not torch.cuda.is_available():
            logger.info("CUDA not available, CPU only mode")
            return [(-1, "CPU Only")]

        gpus = [(-1, "CPU Only")]
        try:
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpus.append((i, f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)"))
                logger.info(f"Detected GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        except Exception as e:
            logger.error(f"Error detecting GPUs: {e}")

        return gpus

    @staticmethod
    def get_gpu_memory(gpu_id: int) -> Tuple[float, float]:
        """
        Get GPU memory usage

        Args:
            gpu_id: GPU device ID (-1 for CPU)

        Returns:
            Tuple of (used_gb, total_gb)
        """
        if gpu_id < 0 or not torch.cuda.is_available():
            return 0.0, 0.0

        try:
            total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            return allocated, total
        except Exception as e:
            logger.error(f"Error getting GPU memory: {e}")
            return 0.0, 0.0

    @staticmethod
    def clear_gpu_cache(gpu_id: int) -> None:
        """
        Clear GPU cache to free memory

        Args:
            gpu_id: GPU device ID (-1 for CPU)
        """
        if gpu_id < 0 or not torch.cuda.is_available():
            return

        try:
            torch.cuda.empty_cache()
            logger.info(f"Cleared cache for GPU {gpu_id}")
        except Exception as e:
            logger.error(f"Error clearing GPU cache: {e}")

    @staticmethod
    def set_device(gpu_id: int) -> None:
        """
        Set the current CUDA device

        Args:
            gpu_id: GPU device ID (-1 for CPU)
        """
        if gpu_id < 0:
            logger.info("Using CPU mode")
            return

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return

        try:
            torch.cuda.set_device(gpu_id)
            logger.info(f"Set device to GPU {gpu_id}")
        except Exception as e:
            logger.error(f"Error setting device: {e}")

    @staticmethod
    def get_system_info() -> Dict:
        """
        Get system GPU information

        Returns:
            Dict with system info
        """
        from typing import Dict

        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        if torch.cuda.is_available():
            gpus = []
            for i in range(torch.cuda.device_count()):
                used, total = GPUDetector.get_gpu_memory(i)
                gpus.append(
                    {
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_used_gb": f"{used:.1f}",
                        "memory_total_gb": f"{total:.1f}",
                    }
                )
            info["gpus"] = gpus

        return info
