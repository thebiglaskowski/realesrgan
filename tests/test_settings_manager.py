"""Unit tests for settings manager"""

from __future__ import annotations
import unittest
import tempfile
import os
import json

from engine.settings_manager import SettingsManager


class TestSettingsManager(unittest.TestCase):
    """Test SettingsManager class"""

    def setUp(self):
        """Create temporary test files"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "settings.json")
        self.presets_file = os.path.join(self.temp_dir, "presets.json")

        self.default_presets = {
            "High Quality": {"model": "RealESRGAN_x4plus", "scale": 4, "fp32": True},
            "Fast": {"model": "RealESRGAN_x2plus", "scale": 2, "fp32": False},
        }

        self.manager = SettingsManager(
            self.config_file, self.presets_file, self.default_presets
        )

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_load_settings_defaults(self):
        """Test loading settings when no file exists"""
        settings = self.manager.load_settings()

        self.assertIsInstance(settings, dict)
        self.assertIn("model", settings)
        self.assertIn("scale", settings)

    def test_save_and_load_settings(self):
        """Test saving and loading settings"""
        test_settings = {
            "model": "RealESRGAN_x4plus_anime_6B",
            "scale": 4,
            "fp32": True,
            "face_enhance": False,
            "auto_tile": True,
            "gpu_id": 0,
        }

        # Clear cache before saving
        self.manager.clear_cache()

        # Save settings
        result = self.manager.save_settings(test_settings)
        self.assertIn("✅", result)
        self.assertTrue(os.path.exists(self.config_file))

        # Load settings
        self.manager.clear_cache()
        loaded_settings = self.manager.load_settings()

        self.assertEqual(loaded_settings["model"], test_settings["model"])
        self.assertEqual(loaded_settings["scale"], test_settings["scale"])
        self.assertEqual(loaded_settings["fp32"], test_settings["fp32"])

    def test_load_presets_defaults(self):
        """Test loading presets when no file exists"""
        presets = self.manager.load_presets()

        self.assertIsInstance(presets, dict)
        self.assertEqual(len(presets), len(self.default_presets))
        self.assertIn("High Quality", presets)
        self.assertIn("Fast", presets)

    def test_save_and_load_preset(self):
        """Test saving and loading a preset"""
        preset_settings = {
            "model": "RealESRGAN_x2plus",
            "scale": 2,
            "fp32": False,
            "face_enhance": False,
            "auto_tile": True,
        }

        # Save preset
        result = self.manager.save_preset("Custom", preset_settings)
        self.assertIn("✅", result)
        self.assertTrue(os.path.exists(self.presets_file))

        # Load preset
        self.manager.clear_cache()
        preset = self.manager.get_preset("Custom")

        self.assertIsNotNone(preset)
        self.assertEqual(preset["model"], preset_settings["model"])
        self.assertEqual(preset["scale"], preset_settings["scale"])

    def test_get_nonexistent_preset(self):
        """Test getting a nonexistent preset"""
        preset = self.manager.get_preset("NonExistent")
        self.assertIsNone(preset)

    def test_delete_preset(self):
        """Test deleting a preset"""
        preset_settings = {
            "model": "RealESRGAN_x4plus",
            "scale": 4,
            "fp32": True,
        }

        # Save and then delete
        self.manager.save_preset("ToDelete", preset_settings)
        self.manager.clear_cache()

        result = self.manager.delete_preset("ToDelete")
        self.assertIn("✅", result)

        # Verify it's deleted
        self.manager.clear_cache()
        preset = self.manager.get_preset("ToDelete")
        self.assertIsNone(preset)

    def test_delete_nonexistent_preset(self):
        """Test deleting a nonexistent preset"""
        result = self.manager.delete_preset("NonExistent")
        self.assertIn("❌", result)

    def test_settings_caching(self):
        """Test that settings are properly cached"""
        settings1 = self.manager.load_settings()
        settings2 = self.manager.load_settings()

        # Should return the same object (cached)
        self.assertIs(settings1, settings2)

    def test_presets_caching(self):
        """Test that presets are properly cached"""
        presets1 = self.manager.load_presets()
        presets2 = self.manager.load_presets()

        # Should return the same object (cached)
        self.assertIs(presets1, presets2)

    def test_clear_cache(self):
        """Test clearing the cache"""
        self.manager.load_settings()
        self.manager.load_presets()

        self.manager.clear_cache()

        # Cache should be empty
        self.assertIsNone(self.manager._settings_cache)
        self.assertIsNone(self.manager._presets_cache)


if __name__ == "__main__":
    unittest.main()
