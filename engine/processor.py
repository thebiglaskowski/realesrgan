"""Image and video processing with improved memory management"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict
import os
from datetime import datetime
import logging
import cv2
import numpy as np
from PIL import Image
import torch

from utils.metrics import calculate_image_metrics
from utils.validation import validate_pil_image, validate_image_file, validate_video_file
from utils.gpu import GPUDetector

logger = logging.getLogger(__name__)


def enhance_image_direct(
    input_image: Image.Image,
    model_name: str,
    scale: int,
    tile: int,
    fp32: bool,
    face_enhance: bool,
    auto_tile: bool,
    denoise_strength: float,
    gpu_id: int,
    model_manager,
    config,
) -> Tuple[Optional[str], Optional[str], str, Optional[Dict]]:
    """
    Enhance image using direct Python API with improved memory management

    Args:
        input_image: PIL Image to enhance
        model_name: Name of the model to use
        scale: Upscale factor (2, 3, or 4)
        tile: Tile size (0 for auto)
        fp32: Use FP32 precision
        face_enhance: Enable face enhancement
        auto_tile: Auto-calculate tile size
        denoise_strength: Denoise strength (0-1)
        gpu_id: GPU device ID (-1 for CPU)
        model_manager: ModelManager instance
        config: Configuration object

    Returns:
        Tuple of (output_path, input_path, status_message, metrics_dict)
    """
    try:
        # Validate input
        is_valid, error_msg = validate_pil_image(input_image)
        if not is_valid:
            return None, None, f"❌ {error_msg}", None

        # Auto-calculate tile if enabled
        if auto_tile:
            from engine.intelligent_enhancer import IntelligentEnhancer
            tile = IntelligentEnhancer.calculate_optimal_tile(input_image)

        # Create output directories
        os.makedirs(config.INPUT_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        # Create timestamped paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = os.path.join(config.INPUT_DIR, f"input_{timestamp}.png")
        output_path = os.path.join(config.OUTPUT_DIR, f"enhanced_{timestamp}.png")

        # Save input image
        input_image.save(input_path)

        # Convert PIL to OpenCV
        img = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

        # Load main model
        upsampler = model_manager.load_model(model_name, gpu_id, fp32)
        upsampler.tile = tile
        upsampler.tile_pad = 10
        upsampler.pre_pad = 0

        # Perform enhancement
        try:
            # Face enhancement if enabled
            if face_enhance and config.ENABLE_FACE_ENHANCEMENT:
                try:
                    from gfpgan import GFPGANer

                    # Create face enhancer as a separate instance
                    face_enhancer = GFPGANer(
                        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                        upscale=scale,
                        arch="clean",
                        channel_multiplier=2,
                        bg_upsampler=upsampler,
                    )

                    # Enhanced with face enhancement
                    _, _, output = face_enhancer.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True
                    )

                    # Clean up face enhancer to free CUDA memory
                    del face_enhancer
                    GPUDetector.clear_gpu_cache(gpu_id)

                except Exception as e:
                    logger.warning(f"Face enhancement failed: {e}, using regular enhancement")
                    output, _ = upsampler.enhance(img, outscale=scale)
            else:
                # Regular enhancement without face enhancement
                output, _ = upsampler.enhance(img, outscale=scale)

            # Save output
            cv2.imwrite(output_path, output)

            # Calculate metrics
            enhanced_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            metrics = calculate_image_metrics(input_image, enhanced_image)

            status = (
                f"✅ Enhanced successfully!\n"
                f"📊 Tile: {tile if tile > 0 else 'Auto'} | "
                f"GPU: {gpu_id if gpu_id >= 0 else 'CPU'}"
            )
            logger.info(f"Image enhanced: {input_path} -> {output_path}")

            return output_path, input_path, status, metrics

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                error_msg = (
                    "❌ GPU out of memory! Try:\n"
                    "- Enable Auto Tile\n"
                    "- Reduce tile size\n"
                    "- Disable Face Enhancement\n"
                    "- Use FP16 mode\n"
                    "- Switch to CPU"
                )
            else:
                error_msg = f"❌ Runtime error: {str(e)}"
            logger.error(error_msg)
            return None, None, error_msg, None

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logger.error(error_msg)
        return None, None, error_msg, None

    finally:
        # Cleanup to free memory
        if gpu_id >= 0:
            GPUDetector.clear_gpu_cache(gpu_id)


def enhance_video_direct(
    video_path: str,
    model_name: str,
    scale: int,
    tile: int,
    fp32: bool,
    face_enhance: bool,
    denoise_strength: float,
    gpu_id: int,
    model_manager,
    config,
) -> Tuple[Optional[str], str]:
    """
    Enhance video using direct API (frame-by-frame)

    Args:
        video_path: Path to input video
        model_name: Name of the model
        scale: Upscale factor
        tile: Tile size
        fp32: Use FP32 precision
        face_enhance: Enable face enhancement
        denoise_strength: Denoise strength
        gpu_id: GPU device ID
        model_manager: ModelManager instance
        config: Configuration object

    Returns:
        Tuple of (output_path, status_message)
    """
    try:
        # Validate video file
        is_valid, error_msg = validate_video_file(video_path, config)
        if not is_valid:
            return None, f"❌ {error_msg}"

        # For now, use subprocess method (TODO: implement full direct processing)
        # This is a placeholder that can be extended
        logger.warning("Direct video processing not yet implemented, use subprocess method")
        return None, "❌ Direct video processing not yet implemented"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def process_batch(
    image_paths: List[str],
    model_name: str,
    scale: int,
    tile: int,
    fp32: bool,
    face_enhance: bool,
    auto_tile: bool,
    denoise_strength: float,
    gpu_id: int,
    model_manager,
    config,
) -> Tuple[List[str], str]:
    """
    Process multiple images in batch

    Args:
        image_paths: List of image file paths
        model_name: Name of the model
        scale: Upscale factor
        tile: Tile size
        fp32: Use FP32 precision
        face_enhance: Enable face enhancement
        auto_tile: Auto-calculate tile size
        denoise_strength: Denoise strength
        gpu_id: GPU device ID
        model_manager: ModelManager instance
        config: Configuration object

    Returns:
        Tuple of (results_list, status_message)
    """
    if not image_paths or len(image_paths) == 0:
        return [], "❌ No images provided"

    results = []
    total = len(image_paths)
    successful = 0
    failed = 0

    for idx, img_path in enumerate(image_paths, 1):
        try:
            logger.info(f"Processing batch image {idx}/{total}: {img_path}")

            # Validate and open image
            is_valid, error_msg = validate_image_file(img_path, config)
            if not is_valid:
                logger.error(f"Image validation failed: {error_msg}")
                failed += 1
                continue

            image = Image.open(img_path)

            # Process image
            enhanced_path, input_path, status, metrics = enhance_image_direct(
                image,
                model_name,
                scale,
                tile,
                fp32,
                face_enhance,
                auto_tile,
                denoise_strength,
                gpu_id,
                model_manager,
                config,
            )

            if enhanced_path:
                results.append(enhanced_path)
                successful += 1
            else:
                failed += 1
                logger.error(f"Failed to process image {idx}: {status}")

        except Exception as e:
            failed += 1
            logger.error(f"Error processing image {idx}: {e}")

    status_msg = (
        f"✅ Batch complete: {successful} successful, {failed} failed out of {total} total"
    )
    logger.info(status_msg)

    return results, status_msg
