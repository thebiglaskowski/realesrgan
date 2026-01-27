"""Intelligent image enhancement settings analyzer"""

from __future__ import annotations
from typing import Dict
import cv2
import numpy as np
from PIL import Image, ImageStat
import logging

logger = logging.getLogger(__name__)

# Thresholds for content detection
SATURATION_ANIME_THRESHOLD = 100
EDGE_DENSITY_ANIME_THRESHOLD = 0.02
COLOR_STDDEV_ANIME_THRESHOLD = 50
EDGE_DENSITY_ANIME_THRESHOLD_2 = 0.03
SATURATION_PHOTO_THRESHOLD = 80
COLOR_STDDEV_PHOTO_THRESHOLD = 60

# Noise level thresholds
NOISE_HIGH_THRESHOLD = 0.6
NOISE_MEDIUM_THRESHOLD = 0.4

# Resolution thresholds
HIGH_RES_THRESHOLD = 8_000_000
LARGE_RES_THRESHOLD = 16_000_000
SMALL_RES_THRESHOLD = 2_000_000


class IntelligentEnhancer:
    """AI-powered intelligent settings selector"""

    @staticmethod
    def detect_content_type(image: Image.Image) -> str:
        """
        Detect if image is photo, anime, or illustration.
        Uses color distribution, edge patterns, and saturation analysis.

        Args:
            image: PIL Image object

        Returns:
            str: "anime" or "photo"
        """
        try:
            # Convert to numpy for analysis
            img_array = np.array(image)

            # Analyze color saturation
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1].mean()

            # Analyze edge patterns
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = edges.sum() / edges.size

            # Analyze color palette diversity
            stat = ImageStat.Stat(image)
            color_stddev = np.mean(stat.stddev)

            # Decision logic
            if saturation > SATURATION_ANIME_THRESHOLD and edge_density > EDGE_DENSITY_ANIME_THRESHOLD:
                # High saturation + strong edges = anime/illustration
                return "anime"
            elif color_stddev < COLOR_STDDEV_ANIME_THRESHOLD and edge_density > EDGE_DENSITY_ANIME_THRESHOLD_2:
                # Low color variance + strong edges = anime
                return "anime"
            elif saturation < SATURATION_PHOTO_THRESHOLD and color_stddev > COLOR_STDDEV_PHOTO_THRESHOLD:
                # Natural saturation + high variance = photo
                return "photo"
            else:
                # Default to photo
                return "photo"

        except Exception as e:
            logger.error(f"Content detection error: {e}")
            return "photo"  # Safe default

    @staticmethod
    def detect_faces(image: Image.Image) -> int:
        """
        Detect number of faces in image using OpenCV Haar Cascade.

        Args:
            image: PIL Image object

        Returns:
            int: Number of faces detected
        """
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Load face cascade
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            return len(faces)

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return 0

    @staticmethod
    def analyze_noise_level(image: Image.Image) -> float:
        """
        Estimate noise level in image.

        Args:
            image: PIL Image object

        Returns:
            float: Noise level 0.0 (clean) to 1.0 (very noisy)
        """
        try:
            img_array = np.array(image.convert("L"))  # Convert to grayscale

            # Use Laplacian variance as noise indicator
            laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()

            # Normalize to 0-1 range (empirical thresholds)
            noise_level = min(laplacian_var / 1000, 1.0)

            return noise_level

        except Exception as e:
            logger.error(f"Noise analysis error: {e}")
            return 0.3  # Medium default

    @staticmethod
    def calculate_optimal_tile(image: Image.Image) -> int:
        """
        Calculate optimal tile size based on resolution

        Args:
            image: PIL Image object

        Returns:
            int: Recommended tile size (0 = no tiling)
        """
        width, height = image.size
        pixels = width * height

        if pixels > LARGE_RES_THRESHOLD:  # 4K+
            return 400
        elif pixels > HIGH_RES_THRESHOLD:  # 1440p+
            return 512
        else:
            return 0

    @classmethod
    def get_intelligent_settings(cls, image: Image.Image) -> Dict:
        """
        Analyze image and return optimal settings.
        This is the main AI function!

        Args:
            image: PIL Image object

        Returns:
            Dict with settings, reasoning, and analysis
        """
        logger.info("🤖 Running intelligent analysis...")

        # Detect content type
        content_type = cls.detect_content_type(image)
        logger.info(f"   Content type: {content_type}")

        # Detect faces
        num_faces = cls.detect_faces(image)
        logger.info(f"   Faces detected: {num_faces}")

        # Analyze noise
        noise_level = cls.analyze_noise_level(image)
        logger.info(f"   Noise level: {noise_level:.2f}")

        # Calculate optimal tile
        optimal_tile = cls.calculate_optimal_tile(image)
        logger.info(f"   Optimal tile: {optimal_tile}")

        # Get image dimensions
        width, height = image.size
        pixels = width * height

        # Decision tree for optimal settings
        settings = {}
        reasoning = []

        # Model selection
        if content_type == "anime":
            settings["model"] = "RealESRGAN_x4plus_anime_6B"
            reasoning.append("🎨 Anime/illustration detected → Using anime-optimized model")
        else:
            settings["model"] = "RealESRGAN_x4plus"
            reasoning.append("📷 Photo detected → Using general-purpose model")

        # Face enhancement
        if num_faces > 0:
            settings["face_enhance"] = True
            reasoning.append(f"👤 {num_faces} face(s) detected → Enabling face enhancement")
        else:
            settings["face_enhance"] = False
            reasoning.append("No faces detected → Face enhancement off")

        # Scale selection
        if pixels > HIGH_RES_THRESHOLD:  # Already high-res
            settings["scale"] = 2
            reasoning.append("📐 High resolution image → Using 2x scale (avoid over-processing)")
        else:
            settings["scale"] = 4
            reasoning.append("📐 Standard resolution → Using 4x scale for maximum quality")

        # Precision mode
        if num_faces > 0 or pixels < SMALL_RES_THRESHOLD:  # Portraits or small images
            settings["fp32"] = True
            reasoning.append("⚡ Enabling FP32 for maximum quality")
        else:
            settings["fp32"] = False
            reasoning.append("⚡ Using FP16 for balanced speed/quality")

        # Tile size
        settings["tile"] = optimal_tile
        settings["auto_tile"] = True
        if optimal_tile > 0:
            reasoning.append(f"🔲 Large image → Using tile size {optimal_tile}")
        else:
            reasoning.append("🔲 Standard image → No tiling needed")

        # Denoise strength (for compatible models)
        if noise_level > NOISE_HIGH_THRESHOLD:
            settings["denoise_strength"] = 0.8
            reasoning.append(f"🎛️ High noise detected → Denoise strength 0.8")
        elif noise_level > NOISE_MEDIUM_THRESHOLD:
            settings["denoise_strength"] = 0.6
            reasoning.append(f"🎛️ Medium noise detected → Denoise strength 0.6")
        else:
            settings["denoise_strength"] = 0.3
            reasoning.append(f"🎛️ Clean image → Denoise strength 0.3")

        logger.info("✅ Intelligent analysis complete!")

        return {
            "settings": settings,
            "reasoning": reasoning,
            "analysis": {
                "content_type": content_type,
                "faces": num_faces,
                "noise_level": f"{noise_level:.2f}",
                "resolution": f"{width}x{height}",
                "optimal_tile": optimal_tile,
            },
        }
