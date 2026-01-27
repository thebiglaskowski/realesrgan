"""Unit tests for intelligent enhancer"""

from __future__ import annotations
import unittest
from PIL import Image
import numpy as np

from engine.intelligent_enhancer import IntelligentEnhancer


class TestIntelligentEnhancer(unittest.TestCase):
    """Test IntelligentEnhancer class"""

    def setUp(self):
        """Create test images"""
        # Photo-like image (natural colors, varied saturation)
        self.photo_image = Image.new("RGB", (100, 100))
        pixels = self.photo_image.load()
        for i in range(100):
            for j in range(100):
                pixels[i, j] = (
                    50 + i % 100,
                    100 + j % 100,
                    150 + (i + j) % 100,
                )
        self.photo_image.paste(self.photo_image)

        # Anime-like image (high saturation, clear edges)
        self.anime_image = Image.new("RGB", (100, 100))
        pixels = self.anime_image.load()
        for i in range(100):
            for j in range(100):
                if i < 50:
                    pixels[i, j] = (255, 0, 0)  # Bright red
                else:
                    pixels[i, j] = (0, 255, 0)  # Bright green
        self.anime_image.paste(self.anime_image)

    def test_detect_content_type_returns_string(self):
        """Test that detect_content_type returns a string"""
        result = IntelligentEnhancer.detect_content_type(self.photo_image)
        self.assertIsInstance(result, str)
        self.assertIn(result, ["anime", "photo"])

    def test_detect_faces_returns_int(self):
        """Test that detect_faces returns an integer"""
        result = IntelligentEnhancer.detect_faces(self.photo_image)
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    def test_analyze_noise_level_returns_float(self):
        """Test that analyze_noise_level returns a float in range [0, 1]"""
        result = IntelligentEnhancer.analyze_noise_level(self.photo_image)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_calculate_optimal_tile_returns_int(self):
        """Test that calculate_optimal_tile returns an integer"""
        result = IntelligentEnhancer.calculate_optimal_tile(self.photo_image)
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    def test_calculate_optimal_tile_values(self):
        """Test tile size calculation for different resolutions"""
        # Small image
        small_img = Image.new("RGB", (256, 256))
        tile = IntelligentEnhancer.calculate_optimal_tile(small_img)
        self.assertEqual(tile, 0)

        # Medium image
        medium_img = Image.new("RGB", (1440, 1440))
        tile = IntelligentEnhancer.calculate_optimal_tile(medium_img)
        self.assertGreater(tile, 0)

        # Large image
        large_img = Image.new("RGB", (2160, 2160))
        tile = IntelligentEnhancer.calculate_optimal_tile(large_img)
        self.assertGreater(tile, 0)

    def test_get_intelligent_settings_returns_dict(self):
        """Test that get_intelligent_settings returns correct structure"""
        result = IntelligentEnhancer.get_intelligent_settings(self.photo_image)

        self.assertIsInstance(result, dict)
        self.assertIn("settings", result)
        self.assertIn("reasoning", result)
        self.assertIn("analysis", result)

        # Check settings structure
        settings = result["settings"]
        self.assertIn("model", settings)
        self.assertIn("scale", settings)
        self.assertIn("fp32", settings)
        self.assertIn("face_enhance", settings)
        self.assertIn("tile", settings)
        self.assertIn("auto_tile", settings)

        # Check analysis structure
        analysis = result["analysis"]
        self.assertIn("content_type", analysis)
        self.assertIn("faces", analysis)
        self.assertIn("noise_level", analysis)
        self.assertIn("resolution", analysis)

        # Check reasoning is a list
        self.assertIsInstance(result["reasoning"], list)
        self.assertGreater(len(result["reasoning"]), 0)

    def test_intelligent_settings_values(self):
        """Test that intelligent settings have valid values"""
        result = IntelligentEnhancer.get_intelligent_settings(self.photo_image)
        settings = result["settings"]

        # Model should be one of the valid models
        valid_models = [
            "RealESRGAN_x4plus",
            "RealESRGAN_x4plus_anime_6B",
        ]
        self.assertIn(settings["model"], valid_models)

        # Scale should be 2, 3, or 4
        self.assertIn(settings["scale"], [2, 3, 4])

        # fp32 and face_enhance should be booleans
        self.assertIsInstance(settings["fp32"], bool)
        self.assertIsInstance(settings["face_enhance"], bool)

        # Denoise strength should be float between 0 and 1
        self.assertIsInstance(settings["denoise_strength"], float)
        self.assertGreaterEqual(settings["denoise_strength"], 0.0)
        self.assertLessEqual(settings["denoise_strength"], 1.0)


if __name__ == "__main__":
    unittest.main()
