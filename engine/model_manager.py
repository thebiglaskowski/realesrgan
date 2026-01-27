"""Model management with caching, downloading, and checksum verification"""

from __future__ import annotations
from typing import Optional, Dict
import threading
import logging
import hashlib
from pathlib import Path
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading, caching, and GPU allocation"""

    def __init__(self, models_config: Dict, weights_dir: str = "Real-ESRGAN/weights",
                 verify_checksums: bool = False, checksums: Optional[Dict] = None):
        """
        Initialize model manager

        Args:
            models_config: Dictionary of model configurations
            weights_dir: Directory to store model weights
            verify_checksums: Whether to verify model checksums
            checksums: Dictionary of model checksums for verification
        """
        self.models_config = models_config
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.verify_checksums = verify_checksums
        self.checksums = checksums or {}
        self.cached_models: Dict[str, RealESRGANer] = {}
        self.lock = threading.Lock()

    def calculate_sha256(self, file_path: str, chunk_size: int = 8192) -> str:
        """
        Calculate SHA256 checksum of a file

        Args:
            file_path: Path to file
            chunk_size: Chunk size for reading

        Returns:
            str: Hex digest of SHA256 hash
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return ""

    def verify_model_checksum(self, model_name: str, file_path: str) -> bool:
        """
        Verify model checksum if available

        Args:
            model_name: Name of the model
            file_path: Path to model file

        Returns:
            bool: True if checksum matches or verification disabled
        """
        if not self.verify_checksums or model_name not in self.checksums:
            return True

        try:
            expected_checksum = self.checksums[model_name]
            actual_checksum = self.calculate_sha256(file_path)

            if actual_checksum.lower() == expected_checksum.lower():
                logger.info(f"✅ Checksum verified for {model_name}")
                return True
            else:
                logger.error(
                    f"❌ Checksum mismatch for {model_name}: "
                    f"expected {expected_checksum}, got {actual_checksum}"
                )
                return False
        except Exception as e:
            logger.error(f"Error verifying checksum: {e}")
            return False

    def get_model_path(self, model_name: str) -> str:
        """
        Get or download model weights

        Args:
            model_name: Name of the model

        Returns:
            str: Path to model file
        """
        if model_name not in self.models_config:
            raise ValueError(f"Unknown model: {model_name}")

        model_info = self.models_config[model_name]
        model_path = self.weights_dir / f"{model_name}.pth"

        if model_path.exists():
            # Verify checksum if enabled
            if not self.verify_model_checksum(model_name, str(model_path)):
                logger.warning(f"Checksum verification failed, re-downloading {model_name}")
                model_path.unlink()  # Delete corrupted file
            else:
                logger.info(f"Using existing model: {model_path}")
                return str(model_path)

        # Download model
        logger.info(f"Downloading model: {model_name}")
        try:
            model_path = load_file_from_url(
                url=model_info["url"],
                model_dir=str(self.weights_dir),
                progress=True,
                file_name=f"{model_name}.pth",
            )
            logger.info(f"Model downloaded: {model_path}")

            # Verify checksum after download
            if not self.verify_model_checksum(model_name, str(model_path)):
                raise RuntimeError(f"Checksum verification failed for {model_name}")

            return str(model_path)

        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise

    def load_model(
        self,
        model_name: str,
        gpu_id: Optional[int],
        fp32: bool = False,
    ) -> RealESRGANer:
        """
        Load and cache model

        Args:
            model_name: Name of the model
            gpu_id: GPU ID (-1 for CPU)
            fp32: Whether to use FP32 precision

        Returns:
            RealESRGANer instance
        """
        cache_key = f"{model_name}_{gpu_id}_{fp32}"

        with self.lock:
            if cache_key in self.cached_models:
                logger.info(f"Using cached model: {cache_key}")
                return self.cached_models[cache_key]

            logger.info(f"Loading model: {model_name} on GPU {gpu_id}")

            model_info = self.models_config[model_name]
            model_path = self.get_model_path(model_name)

            # Create model architecture
            if model_info["model_class"] == "RRDBNet":
                model = RRDBNet(**model_info["params"])
            else:  # SRVGGNetCompact
                model = SRVGGNetCompact(**model_info["params"])

            # Create upsampler
            upsampler = RealESRGANer(
                scale=model_info["scale"],
                model_path=model_path,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=not fp32,
                gpu_id=gpu_id,
            )

            self.cached_models[cache_key] = upsampler
            logger.info(f"Model loaded and cached: {cache_key}")

            return upsampler

    def unload_model(self, model_name: str, gpu_id: Optional[int], fp32: bool) -> None:
        """
        Unload a cached model to free memory

        Args:
            model_name: Name of the model
            gpu_id: GPU ID
            fp32: Whether FP32 was used
        """
        cache_key = f"{model_name}_{gpu_id}_{fp32}"

        with self.lock:
            if cache_key in self.cached_models:
                del self.cached_models[cache_key]
                logger.info(f"Unloaded model: {cache_key}")

    def clear_cache(self) -> None:
        """Clear all cached models"""
        with self.lock:
            self.cached_models.clear()
            logger.info("Cleared model cache")
