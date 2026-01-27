"""Unit tests for validation utilities"""

from __future__ import annotations
import unittest
from pathlib import Path
import tempfile
from PIL import Image
import os

# Mock config for testing
class MockConfig:
    MAX_IMAGE_SIZE_MB = 100
    MAX_VIDEO_SIZE_MB = 1000
    ALLOWED_IMAGE_FORMATS = ("jpg", "jpeg", "png", "webp", "bmp", "tiff")
    ALLOWED_VIDEO_FORMATS = ("mp4", "mkv", "avi", "mov", "flv", "wmv")


class TestImageValidation(unittest.TestCase):
    """Test image validation functions"""

    def setUp(self):
        """Create temporary test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = MockConfig()

        # Create a valid test image
        img = Image.new("RGB", (100, 100), color="red")
        self.valid_image_path = os.path.join(self.temp_dir, "test.png")
        img.save(self.valid_image_path)

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_validate_valid_image(self):
        """Test validation of valid image"""
        from utils.validation import validate_image_file

        is_valid, error_msg = validate_image_file(self.valid_image_path, self.config)
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")

    def test_validate_nonexistent_file(self):
        """Test validation of nonexistent file"""
        from utils.validation import validate_image_file

        is_valid, error_msg = validate_image_file("/nonexistent/file.png", self.config)
        self.assertFalse(is_valid)
        self.assertIn("not found", error_msg)

    def test_validate_none_file(self):
        """Test validation of None file"""
        from utils.validation import validate_image_file

        is_valid, error_msg = validate_image_file(None, self.config)
        self.assertFalse(is_valid)
        self.assertIn("No file provided", error_msg)

    def test_validate_invalid_extension(self):
        """Test validation of invalid extension"""
        from utils.validation import validate_image_file

        invalid_path = os.path.join(self.temp_dir, "test.exe")
        Path(invalid_path).touch()

        is_valid, error_msg = validate_image_file(invalid_path, self.config)
        self.assertFalse(is_valid)
        self.assertIn("Unsupported format", error_msg)


class TestPILImageValidation(unittest.TestCase):
    """Test PIL Image validation"""

    def test_validate_valid_pil_image(self):
        """Test validation of valid PIL image"""
        from utils.validation import validate_pil_image

        img = Image.new("RGB", (100, 100), color="blue")
        is_valid, error_msg = validate_pil_image(img)
        self.assertTrue(is_valid)

    def test_validate_none_image(self):
        """Test validation of None image"""
        from utils.validation import validate_pil_image

        is_valid, error_msg = validate_pil_image(None)
        self.assertFalse(is_valid)
        self.assertIn("No image provided", error_msg)

    def test_validate_invalid_object(self):
        """Test validation of invalid object type"""
        from utils.validation import validate_pil_image

        is_valid, error_msg = validate_pil_image("not an image")
        self.assertFalse(is_valid)
        self.assertIn("Invalid image object type", error_msg)


class TestScaleValidation(unittest.TestCase):
    """Test scale validation"""

    def test_validate_valid_scales(self):
        """Test validation of valid scales"""
        from utils.validation import validate_scale

        for scale in [2, 3, 4]:
            is_valid, error_msg = validate_scale(scale)
            self.assertTrue(is_valid, f"Scale {scale} should be valid")

    def test_validate_invalid_scales(self):
        """Test validation of invalid scales"""
        from utils.validation import validate_scale

        for scale in [1, 5, 8]:
            is_valid, error_msg = validate_scale(scale)
            self.assertFalse(is_valid, f"Scale {scale} should be invalid")


class TestTileSizeValidation(unittest.TestCase):
    """Test tile size validation"""

    def test_validate_valid_tile_sizes(self):
        """Test validation of valid tile sizes"""
        from utils.validation import validate_tile_size

        for tile in [0, 32, 64, 128, 256, 512]:
            is_valid, error_msg = validate_tile_size(tile)
            self.assertTrue(is_valid, f"Tile size {tile} should be valid")

    def test_validate_invalid_tile_sizes(self):
        """Test validation of invalid tile sizes"""
        from utils.validation import validate_tile_size

        is_valid, error_msg = validate_tile_size(-1)
        self.assertFalse(is_valid)

        is_valid, error_msg = validate_tile_size(16)
        self.assertFalse(is_valid)


class TestFilenamesSanitization(unittest.TestCase):
    """Test filename sanitization"""

    def test_sanitize_simple_filename(self):
        """Test sanitization of simple filename"""
        from utils.validation import sanitize_filename

        result = sanitize_filename("test.png")
        self.assertEqual(result, "test.png")

    def test_sanitize_path_traversal(self):
        """Test sanitization prevents path traversal"""
        from utils.validation import sanitize_filename

        result = sanitize_filename("../../../etc/passwd")
        self.assertEqual(result, "passwd")

        result = sanitize_filename("/absolute/path/test.png")
        self.assertEqual(result, "test.png")


if __name__ == "__main__":
    unittest.main()
