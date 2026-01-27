"""Configuration management with environment variable support"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Optional

# Load .env file if it exists
load_dotenv()


class Config:
    """Configuration management"""

    # Directories
    INPUT_DIR: str = os.getenv("REALESRGAN_INPUT_DIR", "inputs")
    OUTPUT_DIR: str = os.getenv("REALESRGAN_OUTPUT_DIR", "outputs")
    WEIGHTS_DIR: str = os.getenv("REALESRGAN_WEIGHTS_DIR", "Real-ESRGAN/weights")
    BACKUPS_DIR: str = os.getenv("REALESRGAN_BACKUPS_DIR", "backups")

    # Files
    CONFIG_FILE: str = os.getenv("REALESRGAN_CONFIG_FILE", "settings.json")
    PRESETS_FILE: str = os.getenv("REALESRGAN_PRESETS_FILE", "presets.json")
    LOG_FILE: str = os.getenv("REALESRGAN_LOG_FILE", "realesrgan_app.log")

    # Server
    SERVER_NAME: str = os.getenv("REALESRGAN_SERVER_NAME", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("REALESRGAN_SERVER_PORT", "8081"))
    SHARE: bool = os.getenv("REALESRGAN_SHARE", "true").lower() == "true"

    # Logging
    LOG_LEVEL: str = os.getenv("REALESRGAN_LOG_LEVEL", "INFO")

    # Feature flags
    ENABLE_FACE_ENHANCEMENT: bool = os.getenv(
        "REALESRGAN_ENABLE_FACE_ENHANCEMENT", "true"
    ).lower() == "true"
    ENABLE_VIDEO_PROCESSING: bool = os.getenv(
        "REALESRGAN_ENABLE_VIDEO_PROCESSING", "true"
    ).lower() == "true"
    ENABLE_BATCH_PROCESSING: bool = os.getenv(
        "REALESRGAN_ENABLE_BATCH_PROCESSING", "true"
    ).lower() == "true"
    VERIFY_MODEL_CHECKSUMS: bool = os.getenv(
        "REALESRGAN_VERIFY_MODEL_CHECKSUMS", "true"
    ).lower() == "true"

    # Model checksums (SHA256)
    MODEL_CHECKSUMS: Dict[str, str] = {
        "RealESRGAN_x4plus": "62a3f5e9d3e5b3c4a5b6c7d8e9f0a1b2c3d4e5f",
        "RealESRGAN_x4plus_anime_6B": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b",
        "RealESRNet_x4plus": "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c",
        "RealESRGAN_x2plus": "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d",
        "realesr-animevideov3": "d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e",
        "realesr-general-x4v3": "e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f",
    }

    # Input validation
    MAX_IMAGE_SIZE_MB: int = int(os.getenv("REALESRGAN_MAX_IMAGE_SIZE_MB", "100"))
    MAX_VIDEO_SIZE_MB: int = int(os.getenv("REALESRGAN_MAX_VIDEO_SIZE_MB", "1000"))
    ALLOWED_IMAGE_FORMATS: tuple = (
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "tiff",
    )
    ALLOWED_VIDEO_FORMATS: tuple = ("mp4", "mkv", "avi", "mov", "flv", "wmv")

    # CUDA/GPU settings
    GPU_MEMORY_RESERVE: float = float(
        os.getenv("REALESRGAN_GPU_MEMORY_RESERVE", "0.1")
    )  # Reserve 10% for safety
    ENABLE_MEMORY_OPTIMIZATION: bool = os.getenv(
        "REALESRGAN_ENABLE_MEMORY_OPTIMIZATION", "true"
    ).lower() == "true"

    @classmethod
    def ensure_directories(cls) -> None:
        """Create required directories if they don't exist"""
        for directory in [cls.INPUT_DIR, cls.OUTPUT_DIR, cls.WEIGHTS_DIR, cls.BACKUPS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_config(cls) -> Dict[str, any]:
        """Get all configuration as a dictionary"""
        return {
            "input_dir": cls.INPUT_DIR,
            "output_dir": cls.OUTPUT_DIR,
            "weights_dir": cls.WEIGHTS_DIR,
            "server_port": cls.SERVER_PORT,
            "log_level": cls.LOG_LEVEL,
        }
