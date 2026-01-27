# Real-ESRGAN Ultimate - Architecture Documentation

## Project Structure Overview

```
realesrgan/
├── main.py                      # Application entry point
├── config.py                    # Configuration management with env vars
├── .env.example                 # Template for environment variables
│
├── engine/                      # Core processing engine
│   ├── __init__.py
│   ├── intelligent_enhancer.py  # AI-powered settings analyzer
│   ├── model_manager.py         # Model loading, caching, checksum verification
│   ├── processor.py             # Image/video processing with improved memory management
│   └── settings_manager.py      # Settings and presets persistence
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── validation.py            # Input validation (images, videos, parameters)
│   ├── metrics.py               # Image quality metrics calculation
│   ├── gpu.py                   # GPU detection and memory management
│   └── logger.py                # Logging setup with rotation
│
├── ui/                          # User interface
│   ├── __init__.py
│   └── components.py            # Gradio interface components
│
└── tests/                       # Unit tests
    ├── __init__.py
    ├── test_validation.py
    ├── test_intelligent_enhancer.py
    └── test_settings_manager.py
```

## Module Responsibilities

### main.py
**Entry point for the application**

- Initializes configuration
- Sets up logging
- Creates Gradio interface
- Launches web server

```python
python main.py
```

### config.py
**Configuration management with environment variable support**

Features:
- Directory paths (input, output, weights, backups)
- File paths (config, presets, logs)
- Server settings (host, port, sharing)
- Feature flags (face enhancement, video processing, batch processing)
- Validation limits (max image/video size)
- GPU optimization settings

Usage:
```python
from config import Config
Config.ensure_directories()
print(Config.INPUT_DIR)  # Use in code or via REALESRGAN_INPUT_DIR env var
```

### engine/intelligent_enhancer.py
**AI-powered image analysis and settings recommendation**

Key Methods:
- `detect_content_type(image)` → "anime" | "photo"
- `detect_faces(image)` → int (face count)
- `analyze_noise_level(image)` → float (0.0-1.0)
- `calculate_optimal_tile(image)` → int (tile size)
- `get_intelligent_settings(image)` → Dict (full recommendation)

Returns:
```python
{
    "settings": {
        "model": "RealESRGAN_x4plus",
        "scale": 4,
        "fp32": True,
        "face_enhance": False,
        "tile": 0,
        "auto_tile": True,
        "denoise_strength": 0.5
    },
    "reasoning": ["Reason 1", "Reason 2", ...],
    "analysis": {
        "content_type": "photo",
        "faces": 1,
        "noise_level": "0.35",
        "resolution": "1920x1080",
        "optimal_tile": 0
    }
}
```

### engine/model_manager.py
**Model loading, caching, downloading, and verification**

Key Features:
- Thread-safe model caching
- Automatic weight downloading from GitHub
- Optional SHA256 checksum verification
- Memory-efficient model unloading

Usage:
```python
from engine import ModelManager
from config import Config

manager = ModelManager(
    models_config=MODELS,
    weights_dir=Config.WEIGHTS_DIR,
    verify_checksums=Config.VERIFY_MODEL_CHECKSUMS
)

upsampler = manager.load_model("RealESRGAN_x4plus", gpu_id=0, fp32=False)
manager.unload_model("RealESRGAN_x4plus", gpu_id=0, fp32=False)
```

### engine/processor.py
**Image and video processing with improved memory management**

Key Functions:
- `enhance_image_direct()` - Process single image with CUDA memory optimization
- `enhance_video_direct()` - Process video (placeholder for future)
- `process_batch()` - Batch process multiple images

Improvements:
- GPU memory clearing after face enhancement
- Graceful OOM error handling
- Input validation before processing
- In-memory metrics calculation (no temp files)

### engine/settings_manager.py
**Settings and presets persistence**

Key Methods:
- `load_settings()` / `save_settings()` - User settings
- `load_presets()` / `save_preset()` / `get_preset()` / `delete_preset()`
- Built-in caching for performance
- JSON-based persistence

### utils/validation.py
**Comprehensive input validation**

Validation Functions:
- `validate_image_file(path, config)` - File existence, size, format, integrity
- `validate_pil_image(image)` - PIL Image object validation
- `validate_video_file(path, config)` - Video file validation
- `validate_model_name(name, models)` - Model availability check
- `validate_scale(scale)` - Scale factor validation (2, 3, 4)
- `validate_tile_size(tile)` - Tile size validation
- `sanitize_filename(filename)` - Path traversal prevention

Returns:
```python
is_valid, error_message = validate_image_file("image.png", config)
# is_valid: bool
# error_message: str (empty if valid)
```

### utils/metrics.py
**Image quality metrics calculation**

Functions:
- `calculate_image_metrics(original, enhanced)` - Compare before/after
- `get_image_info(image)` - Get basic image information

Returns:
```python
{
    "original_size": "1920x1080",
    "enhanced_size": "3840x2160",
    "scale_factor": "2.0x",
    "original_filesize": "456.2 KB",
    "enhanced_filesize": "1234.5 KB",
    "size_change": "+170.6%"
}
```

### utils/gpu.py
**GPU detection and memory management**

Static Methods:
- `get_available_gpus()` → List[(gpu_id, gpu_name)]
- `get_gpu_memory(gpu_id)` → (used_gb, total_gb)
- `clear_gpu_cache(gpu_id)` - Free GPU memory
- `set_device(gpu_id)` - Set active GPU
- `get_system_info()` - Full system GPU info

### utils/logger.py
**Logging configuration with file rotation**

Functions:
- `setup_logging(log_file, log_level)` - Initialize logging system
- `get_logger(name)` - Get logger for a module

Features:
- Console and file output
- Automatic log rotation (10MB default)
- Keeps 5 backup files
- Configurable log level

### ui/components.py
**Gradio web interface components**

Provides:
- 5-tab interface (Image Enhancement, Video, Batch, Presets, System Info)
- AI analysis section with recommendations
- Manual settings controls
- Before/after comparison slider
- Quality metrics display
- Batch processing gallery
- System information display

## Data Flow

### Image Enhancement Flow
```
User uploads image
    ↓
Gradio receives PIL Image
    ↓
validate_pil_image() - Validate input
    ↓
IntelligentEnhancer.get_intelligent_settings()
    ├─ detect_content_type()
    ├─ detect_faces()
    ├─ analyze_noise_level()
    └─ calculate_optimal_tile()
    ↓
Display AI recommendations
    ↓
User clicks "Enhance Image"
    ↓
enhance_image_direct()
    ├─ Validate PIL Image
    ├─ Auto-calculate tile if needed
    ├─ Load model via ModelManager (cached)
    ├─ Perform upscaling
    ├─ Face enhancement (if enabled)
    ├─ Clear GPU cache
    └─ Calculate metrics
    ↓
Display comparison slider + metrics
```

### Batch Processing Flow
```
User uploads multiple images
    ↓
For each image:
    ├─ validate_image_file() - Validate
    ├─ Open image with PIL
    ├─ Call enhance_image_direct()
    └─ Collect results
    ↓
Display status: X successful, Y failed
    ↓
Show gallery of results
```

## Error Handling

### Validation Errors
- Caught early before processing
- Specific error messages to user
- Logged for debugging

### CUDA OOM Errors
- Caught in `enhance_image_direct()`
- Suggests: auto-tile, reduce tile, FP16, disable face enhancement
- Logged with traceback
- Graceful fallback to next image in batch

### Model Download Errors
- Logged with URL and error details
- Suggests manual download if network fails
- Checksum verification ensures integrity

### Settings File Errors
- Falls back to defaults if JSON invalid
- Creates backup before overwriting
- Logs all errors

## Configuration Options

### Environment Variables
```bash
# Directories
REALESRGAN_INPUT_DIR=inputs
REALESRGAN_OUTPUT_DIR=outputs
REALESRGAN_WEIGHTS_DIR=Real-ESRGAN/weights

# Server
REALESRGAN_SERVER_PORT=8081
REALESRGAN_SERVER_NAME=0.0.0.0
REALESRGAN_SHARE=true

# Features
REALESRGAN_ENABLE_FACE_ENHANCEMENT=true
REALESRGAN_VERIFY_MODEL_CHECKSUMS=true

# GPU
REALESRGAN_GPU_MEMORY_RESERVE=0.1
REALESRGAN_ENABLE_MEMORY_OPTIMIZATION=true

# Validation
REALESRGAN_MAX_IMAGE_SIZE_MB=100
REALESRGAN_MAX_VIDEO_SIZE_MB=1000
```

## Performance Characteristics

| Operation | Time | Memory |
|-----------|------|--------|
| Model Loading (first) | 5-10s | 2-3GB |
| Model Loading (cached) | <100ms | 0 |
| Image Analysis | 100-200ms | 500MB |
| Image Enhancement (4MP) | 5-10s | 2-3GB |
| Face Enhancement (add) | 2-5s | 1-2GB |
| Batch Processing (10 images) | 50-100s | 2-3GB (shared) |

## Thread Safety

- **ModelManager**: Thread-safe via `threading.Lock()`
- **SettingsManager**: Not thread-safe by default (single-threaded Gradio)
- **GPUDetector**: Thread-safe static methods
- **Logging**: Thread-safe via Python logging module

## Dependencies

### Core
- PyTorch >= 1.7 (with CUDA)
- BasicSR >= 1.4.2
- OpenCV >= 4.5
- Pillow

### Optional
- GFPGAN >= 1.3.8 (face enhancement)
- FFmpeg (video processing)

### Development
- pytest (testing)
- pytest-cov (coverage)
- python-dotenv (env vars)

## Future Improvements

1. **Direct Video Processing**: Frame-by-frame processing in Python (currently subprocess)
2. **Streaming Inference**: Process very large images in chunks
3. **Multi-GPU Support**: Distribute batch processing across GPUs
4. **Web Socket Updates**: Real-time progress for long operations
5. **Database Backend**: Store processing history and metadata
6. **REST API**: Standalone API separate from Gradio UI
7. **Docker Support**: Containerized deployment
8. **Model Fine-tuning**: User-friendly model training interface

## Testing

Run all tests:
```bash
python -m pytest tests/ -v --cov=engine --cov=utils
```

Test coverage target: 80%+

Test categories:
- Unit tests: Individual function behavior
- Integration tests: Module interactions
- Performance tests: Speed benchmarks

## Debugging

Enable debug logging:
```python
from config import Config
from utils.logger import setup_logging

Config.LOG_LEVEL = "DEBUG"
logger = setup_logging(Config.LOG_FILE, "DEBUG")
```

Check logs:
```bash
tail -f realesrgan_app.log
```

## Best Practices

1. **Always validate input** before processing
2. **Use ModelManager** for consistent model loading
3. **Clear GPU cache** after memory-intensive operations
4. **Use SettingsManager** for persistent settings
5. **Check logs** for debugging issues
6. **Use Config** for all configuration
7. **Import from __init__.py** for clean API

## Example Usage

```python
from main import Config, logger
from engine import IntelligentEnhancer, ModelManager, enhance_image_direct
from utils.validation import validate_pil_image
from utils.gpu import GPUDetector
from PIL import Image

# Setup
Config.ensure_directories()
gpus = GPUDetector.get_available_gpus()

# Load models
model_manager = ModelManager(MODELS, Config.WEIGHTS_DIR)

# Analyze image
enhancer = IntelligentEnhancer()
image = Image.open("photo.jpg")
settings = enhancer.get_intelligent_settings(image)
print(settings["reasoning"])

# Enhance
result_path, input_path, status, metrics = enhance_image_direct(
    image,
    model_name=settings["settings"]["model"],
    scale=settings["settings"]["scale"],
    **settings["settings"],
    gpu_id=0,
    model_manager=model_manager,
    config=Config
)

print(f"Status: {status}")
print(f"Metrics: {metrics}")
```
