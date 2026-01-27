"""Image quality metrics calculation"""

from __future__ import annotations
import os
import logging
from typing import Dict
from PIL import Image
import io

logger = logging.getLogger(__name__)


def calculate_image_metrics(original: Image.Image, enhanced: Image.Image) -> Dict:
    """
    Calculate quality metrics for comparison

    Args:
        original: PIL Image object (original)
        enhanced: PIL Image object (enhanced)

    Returns:
        Dict with metrics
    """
    try:
        metrics = {
            "original_size": f"{original.size[0]}x{original.size[1]}",
            "enhanced_size": f"{enhanced.size[0]}x{enhanced.size[1]}",
            "scale_factor": f"{enhanced.size[0] / original.size[0]:.1f}x",
        }

        # Calculate file sizes using in-memory approach (no temp files)
        original_buffer = io.BytesIO()
        enhanced_buffer = io.BytesIO()

        original.save(original_buffer, "PNG")
        enhanced.save(enhanced_buffer, "PNG")

        original_filesize = original_buffer.getbuffer().nbytes / 1024
        enhanced_filesize = enhanced_buffer.getbuffer().nbytes / 1024

        metrics["original_filesize"] = f"{original_filesize:.1f} KB"
        metrics["enhanced_filesize"] = f"{enhanced_filesize:.1f} KB"

        # Calculate size change percentage
        if original_filesize > 0:
            size_change = ((enhanced_filesize - original_filesize) / original_filesize) * 100
            metrics["size_change"] = f"{size_change:+.1f}%"

        return metrics

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {
            "original_size": f"{original.size[0]}x{original.size[1]}",
            "enhanced_size": f"{enhanced.size[0]}x{enhanced.size[1]}",
            "error": str(e),
        }


def get_image_info(image: Image.Image) -> Dict:
    """
    Get basic image information

    Args:
        image: PIL Image object

    Returns:
        Dict with image info
    """
    return {
        "size": f"{image.size[0]}x{image.size[1]}",
        "mode": image.mode,
        "format": image.format or "Unknown",
        "pixels": image.size[0] * image.size[1],
    }
