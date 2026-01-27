"""Settings and presets management"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SettingsManager:
    """Manage application settings and presets"""

    def __init__(self, config_file: str, presets_file: str, default_presets: Dict):
        """
        Initialize settings manager

        Args:
            config_file: Path to settings JSON file
            presets_file: Path to presets JSON file
            default_presets: Default presets dictionary
        """
        self.config_file = config_file
        self.presets_file = presets_file
        self.default_presets = default_presets
        self._settings_cache: Optional[Dict] = None
        self._presets_cache: Optional[Dict] = None

    def load_settings(self) -> Dict:
        """
        Load saved settings from JSON

        Returns:
            Dict with settings
        """
        if self._settings_cache is not None:
            return self._settings_cache

        if Path(self.config_file).exists():
            try:
                with open(self.config_file, "r") as f:
                    self._settings_cache = json.load(f)
                    logger.info(f"Loaded settings from {self.config_file}")
                    return self._settings_cache
            except Exception as e:
                logger.error(f"Error loading settings: {e}")

        # Return defaults if file doesn't exist or error occurred
        default_settings = {
            "model": "RealESRGAN_x4plus",
            "scale": 4,
            "fp32": False,
            "face_enhance": False,
            "auto_tile": True,
            "gpu_id": 0,
        }
        self._settings_cache = default_settings
        return default_settings

    def save_settings(self, settings: Dict) -> str:
        """
        Save settings to JSON

        Args:
            settings: Settings dictionary

        Returns:
            str: Status message
        """
        try:
            with open(self.config_file, "w") as f:
                json.dump(settings, f, indent=2)
            self._settings_cache = settings
            logger.info(f"Saved settings to {self.config_file}")
            return "✅ Settings saved!"
        except Exception as e:
            error_msg = f"❌ Error saving settings: {e}"
            logger.error(error_msg)
            return error_msg

    def load_presets(self) -> Dict:
        """
        Load presets from JSON

        Returns:
            Dict with presets
        """
        if self._presets_cache is not None:
            return self._presets_cache

        if Path(self.presets_file).exists():
            try:
                with open(self.presets_file, "r") as f:
                    self._presets_cache = json.load(f)
                    logger.info(f"Loaded presets from {self.presets_file}")
                    return self._presets_cache
            except Exception as e:
                logger.error(f"Error loading presets: {e}")

        # Return defaults if file doesn't exist or error occurred
        self._presets_cache = self.default_presets.copy()
        return self._presets_cache

    def save_preset(self, name: str, settings: Dict) -> str:
        """
        Save a new preset

        Args:
            name: Preset name
            settings: Settings dictionary

        Returns:
            str: Status message
        """
        try:
            presets = self.load_presets()
            presets[name] = settings

            with open(self.presets_file, "w") as f:
                json.dump(presets, f, indent=2)

            self._presets_cache = presets
            logger.info(f"Saved preset '{name}'")
            return f"✅ Preset '{name}' saved!"
        except Exception as e:
            error_msg = f"❌ Error saving preset: {e}"
            logger.error(error_msg)
            return error_msg

    def get_preset(self, name: str) -> Optional[Dict]:
        """
        Get a preset by name

        Args:
            name: Preset name

        Returns:
            Dict with preset settings or None
        """
        presets = self.load_presets()
        return presets.get(name)

    def delete_preset(self, name: str) -> str:
        """
        Delete a preset

        Args:
            name: Preset name

        Returns:
            str: Status message
        """
        try:
            presets = self.load_presets()
            if name in presets:
                del presets[name]

                with open(self.presets_file, "w") as f:
                    json.dump(presets, f, indent=2)

                self._presets_cache = presets
                logger.info(f"Deleted preset '{name}'")
                return f"✅ Preset '{name}' deleted!"
            else:
                return f"❌ Preset '{name}' not found"
        except Exception as e:
            error_msg = f"❌ Error deleting preset: {e}"
            logger.error(error_msg)
            return error_msg

    def clear_cache(self) -> None:
        """Clear settings and presets cache"""
        self._settings_cache = None
        self._presets_cache = None
        logger.info("Cleared settings cache")
