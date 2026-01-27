"""
Real-ESRGAN Ultimate - Main Entry Point
AI-Powered Intelligent Auto-Enhancement with Full Video Support
"""

from __future__ import annotations
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils.logger import setup_logging, get_logger
from engine import IntelligentEnhancer, ModelManager, SettingsManager

# Initialize configuration
Config.ensure_directories()
logger = setup_logging(Config.LOG_FILE, Config.LOG_LEVEL)

logger.info("=" * 60)
logger.info("🤖 Real-ESRGAN Ultimate - Starting Application")
logger.info("=" * 60)

# Log configuration
logger.info(f"Input Directory: {Config.INPUT_DIR}")
logger.info(f"Output Directory: {Config.OUTPUT_DIR}")
logger.info(f"Log Level: {Config.LOG_LEVEL}")
logger.info(f"Server: {Config.SERVER_NAME}:{Config.SERVER_PORT}")


def create_ui():
    """Create the Gradio web interface"""
    import gradio as gr
    import torch
    from utils.gpu import GPUDetector
    from ui.components import create_interface

    logger.info("Creating Gradio interface...")

    # Initialize core components
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"GPU Count: {torch.cuda.device_count()}")

    # Create interface
    demo = create_interface(logger, Config)
    return demo


def main():
    """Main application entry point"""
    try:
        logger.info("Initializing Real-ESRGAN Ultimate...")

        # Create and launch interface
        demo = create_ui()

        logger.info("Launching Gradio server...")
        demo.queue()
        demo.launch(
            share=Config.SHARE,
            server_name=Config.SERVER_NAME,
            server_port=Config.SERVER_PORT,
            show_error=True,
        )

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
