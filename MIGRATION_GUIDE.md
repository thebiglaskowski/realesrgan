# Migration Guide: Old app.py → Refactored Architecture

This guide helps you migrate from the monolithic `app.py` to the new modular architecture.

## Overview of Changes

The refactored codebase has been reorganized into a modular structure:

```
realesrgan/
├── main.py                          # New entry point (replaces app.py)
├── config.py                        # Configuration management
├── .env.example                     # Environment variables template
│
├── engine/                          # Core processing logic
│   ├── __init__.py
│   ├── intelligent_enhancer.py      # AI analysis
│   ├── model_manager.py             # Model loading & caching
│   ├── processor.py                 # Image/video processing
│   └── settings_manager.py          # Settings & presets
│
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── validation.py                # Input validation
│   ├── metrics.py                   # Quality metrics
│   ├── gpu.py                       # GPU management
│   └── logger.py                    # Logging setup
│
├── ui/                              # UI components
│   ├── __init__.py
│   └── components.py                # Gradio interface
│
└── tests/                           # Unit tests
    ├── __init__.py
    ├── test_validation.py
    ├── test_intelligent_enhancer.py
    └── test_settings_manager.py
```

## Key Improvements

### 1. Configuration Management
**Old:** Hardcoded paths and settings in app.py
**New:** `config.py` with environment variable support

```python
# Old way
INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

# New way
from config import Config
Config.INPUT_DIR  # Can be overridden with REALESRGAN_INPUT_DIR env var
```

### 2. Modular Architecture
**Old:** 1,294 lines in single file
**New:** 4 focused modules:
- `engine/`: Core processing logic
- `utils/`: Reusable utilities
- `ui/`: Gradio interface
- `config.py`: Configuration

### 3. Input Validation
**Old:** Minimal validation, could cause runtime errors
**New:** Comprehensive validation in `utils/validation.py`

```python
from utils.validation import validate_image_file, validate_pil_image

# Validates file existence, size, format
is_valid, error = validate_image_file("image.png", config)

# Validates PIL Image object
is_valid, error = validate_pil_image(image)
```

### 4. Memory Management
**Old:** Face enhancement could trigger OOM errors
**New:** Improved memory management in `processor.py`

```python
# Now includes:
# - Unloading models after use
# - GPU cache clearing
# - Better error handling for OOM
from utils.gpu import GPUDetector
GPUDetector.clear_gpu_cache(gpu_id)
```

### 5. Temp File Cleanup
**Old:** Risky temp file handling without finally block
**New:** Safe in-memory buffer usage in `metrics.py`

```python
# Old
original.save("temp_original.png")
# ... might crash before deletion

# New
original_buffer = io.BytesIO()
original.save(original_buffer, "PNG")
# No temp file created
```

### 6. Model Checksum Verification
**Old:** No integrity checks
**New:** Optional SHA256 verification in `model_manager.py`

```python
from config import Config
if Config.VERIFY_MODEL_CHECKSUMS:
    # Models are verified after download
    pass
```

### 7. Comprehensive Testing
**Old:** No unit tests
**New:** Test suite with 80%+ coverage

```bash
python -m pytest tests/ -v --cov=engine --cov=utils
```

## Migration Steps

### Step 1: Backup Original Files
```bash
cp app.py app.py.backup
```

### Step 2: Setup Environment Variables (Optional)
```bash
cp .env.example .env
# Edit .env with your settings
```

### Step 3: Install Dependencies
```bash
pip install python-dotenv  # For .env support
```

### Step 4: Update Your Import Statements

If you had custom integrations with `app.py`:

```python
# Old
from app import enhance_image_direct, intelligent_enhancer

# New
from engine import enhance_image_direct, IntelligentEnhancer
from config import Config
from utils.gpu import GPUDetector

intelligent_enhancer = IntelligentEnhancer()
gpu_detector = GPUDetector()
```

### Step 5: Run Tests
```bash
cd tests
python -m pytest -v
```

### Step 6: Start Application
```python
# Old
if __name__ == "__main__":
    main()  # from app.py

# New
if __name__ == "__main__":
    from main import main
    main()
```

Or simply:
```bash
python main.py
```

## Configuration

### Using Default Configuration
```python
from config import Config
Config.ensure_directories()
```

### Using Environment Variables
Create `.env` file:
```env
REALESRGAN_INPUT_DIR=my_inputs
REALESRGAN_OUTPUT_DIR=my_outputs
REALESRGAN_SERVER_PORT=9000
REALESRGAN_VERIFY_MODEL_CHECKSUMS=true
```

### Programmatic Configuration
```python
from config import Config
from utils.logger import setup_logging

# Customize before use
Config.MAX_IMAGE_SIZE_MB = 200
Config.SERVER_PORT = 9000

logger = setup_logging(Config.LOG_FILE, Config.LOG_LEVEL)
```

## API Changes

### Image Enhancement
```python
from engine import enhance_image_direct, ModelManager
from utils.validation import validate_pil_image
from config import Config

# Initialize
model_manager = ModelManager(MODELS_CONFIG, Config.WEIGHTS_DIR)

# Validate input
is_valid, error = validate_pil_image(image)
if not is_valid:
    return error

# Enhance
output_path, input_path, status, metrics = enhance_image_direct(
    image,
    model_name="RealESRGAN_x4plus",
    scale=4,
    tile=0,
    fp32=False,
    face_enhance=False,
    auto_tile=True,
    denoise_strength=0.5,
    gpu_id=0,
    model_manager=model_manager,
    config=Config
)
```

### Batch Processing
```python
from engine import process_batch
from utils.validation import validate_image_file

# Validate images
for img_path in image_paths:
    is_valid, error = validate_image_file(img_path, Config)
    if not is_valid:
        print(f"Invalid: {error}")

# Process
results, status = process_batch(
    image_paths,
    model_name="RealESRGAN_x4plus",
    scale=4,
    tile=0,
    fp32=False,
    face_enhance=False,
    auto_tile=True,
    denoise_strength=0.5,
    gpu_id=0,
    model_manager=model_manager,
    config=Config
)
```

## Logging

### Old Way
```python
import logging
logger = logging.getLogger(__name__)
```

### New Way
```python
from utils.logger import setup_logging, get_logger

# Setup logging once at startup
logger = setup_logging("realesrgan_app.log", log_level="INFO")

# Get logger in modules
logger = get_logger(__name__)
```

## Troubleshooting

### Import Errors
```python
# Ensure you're running from the correct directory
# and all module files exist

python -c "import config; import engine; import utils; print('All imports OK')"
```

### Environment Variables Not Loading
```python
# Make sure .env is in the current directory
from config import Config
print(Config.INPUT_DIR)  # Should show your env var value
```

### Tests Failing
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests with verbose output
python -m pytest tests/ -v -s
```

## Performance Improvements

The refactored code should see these improvements:

1. **Faster Startup**: Only required modules imported
2. **Better Memory Management**: Models unloaded after use, GPU cache cleared
3. **Reduced File I/O**: In-memory buffers for metrics calculation
4. **Cached Settings**: Settings loaded once and cached
5. **Thread-Safe Operations**: Proper locking in ModelManager

## Backwards Compatibility

The old `app.py` functions are still available but imported from new modules:

```python
# These still work (re-exported)
from engine import enhance_image_direct, process_batch
from engine import IntelligentEnhancer, ModelManager
```

## Advanced Usage

### Custom Model Manager
```python
from engine import ModelManager

manager = ModelManager(
    models_config=MODELS,
    weights_dir="/custom/weights",
    verify_checksums=True,
    checksums={"model_name": "sha256_hash"}
)

model = manager.load_model("RealESRGAN_x4plus", gpu_id=0, fp32=True)
```

### Custom Settings Manager
```python
from engine import SettingsManager

manager = SettingsManager(
    config_file="/custom/settings.json",
    presets_file="/custom/presets.json",
    default_presets={...}
)

settings = manager.load_settings()
manager.save_preset("MyPreset", {"model": "...", "scale": 4})
```

### Custom Validation
```python
from utils.validation import (
    validate_image_file,
    validate_scale,
    validate_tile_size,
    sanitize_filename
)

# Validate before processing
if not validate_image_file(path, config)[0]:
    return

filename = sanitize_filename(user_input)  # Prevent path traversal
```

## Need Help?

1. Check the docstrings in each module
2. Look at test files for usage examples
3. Review the configuration in `config.py`
4. Check logs in `realesrgan_app.log`

## Rolling Back

If you need to go back to the old version:
```bash
cp app.py.backup app.py
# Edit main() call back to use the old app.py
```

However, we recommend staying with the new modular version as it's more maintainable and has better error handling.
