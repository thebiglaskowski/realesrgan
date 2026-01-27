"""Input validation utilities"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def validate_image_file(file_path: str | None, config) -> Tuple[bool, str]:
    """
    Validate image file for processing

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not file_path:
        return False, "No file provided"

    # Check file exists
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > config.MAX_IMAGE_SIZE_MB:
        return (
            False,
            f"File too large: {file_size_mb:.1f}MB "
            f"(max: {config.MAX_IMAGE_SIZE_MB}MB)",
        )

    # Check file extension
    ext = Path(file_path).suffix.lstrip(".").lower()
    if ext not in config.ALLOWED_IMAGE_FORMATS:
        return (
            False,
            f"Unsupported format: {ext} "
            f"(allowed: {', '.join(config.ALLOWED_IMAGE_FORMATS)})",
        )

    # Try to open as image
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def validate_pil_image(image) -> Tuple[bool, str]:
    """
    Validate PIL Image object

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if image is None:
        return False, "No image provided"

    if not isinstance(image, Image.Image):
        return False, "Invalid image object type"

    # Check image mode
    if image.mode not in ("RGB", "RGBA", "L"):
        try:
            image = image.convert("RGB")
        except Exception as e:
            return False, f"Cannot convert image mode: {str(e)}"

    return True, ""


def validate_video_file(file_path: str | None, config) -> Tuple[bool, str]:
    """
    Validate video file for processing

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not file_path:
        return False, "No file provided"

    # Check file exists
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > config.MAX_VIDEO_SIZE_MB:
        return (
            False,
            f"File too large: {file_size_mb:.1f}MB "
            f"(max: {config.MAX_VIDEO_SIZE_MB}MB)",
        )

    # Check file extension
    ext = Path(file_path).suffix.lstrip(".").lower()
    if ext not in config.ALLOWED_VIDEO_FORMATS:
        return (
            False,
            f"Unsupported format: {ext} "
            f"(allowed: {', '.join(config.ALLOWED_VIDEO_FORMATS)})",
        )

    return True, ""


def validate_model_name(model_name: str, available_models: dict) -> Tuple[bool, str]:
    """
    Validate model name against available models

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not model_name:
        return False, "No model specified"

    if model_name not in available_models:
        return (
            False,
            f"Unknown model: {model_name} "
            f"(available: {', '.join(available_models.keys())})",
        )

    return True, ""


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal

    Args:
        filename: Original filename

    Returns:
        str: Sanitized filename with only basename
    """
    return os.path.basename(filename)


def validate_scale(scale: int) -> Tuple[bool, str]:
    """
    Validate upscale factor

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not isinstance(scale, int):
        return False, "Scale must be an integer"

    if scale not in (2, 3, 4):
        return False, f"Scale must be 2, 3, or 4 (got {scale})"

    return True, ""


def validate_tile_size(tile: int) -> Tuple[bool, str]:
    """
    Validate tile size

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not isinstance(tile, int):
        return False, "Tile size must be an integer"

    if tile < 0:
        return False, "Tile size cannot be negative"

    if tile > 0 and tile < 32:
        return False, "Tile size must be 0 or >= 32"

    return True, ""
