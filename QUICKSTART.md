# 🚀 Quick Start Guide - Real-ESRGAN Ultimate

Get up and running in 5 minutes!

## ⚡ TL;DR - Start Now

```bash
cd "g:\My Drive\scripts\realesrgan"
conda activate realesrgan
python app_ultimate.py
```

Then open your browser to `http://localhost:7860`

---

## 📦 First Time Setup (15 minutes)

### Step 1: Activate Environment (1 min)

```bash
cd "g:\My Drive\scripts\realesrgan"
conda activate realesrgan
```

### Step 2: Verify Installation (2 min)

```bash
# Check Python version
python --version  # Should be 3.10.x

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA Available: True
GPU Count: 1 (or more)
```

### Step 3: Launch Ultimate Edition (1 min)

```bash
python app_ultimate.py
```

You should see:
```
Starting Real-ESRGAN Ultimate...
CUDA Available: True
GPU Count: 1
Running on local URL:  http://0.0.0.0:7860
```

### Step 4: Open in Browser

Navigate to: `http://localhost:7860`

---

## 🎯 Your First Enhancement (2 minutes)

### Single Image

1. **Click** "🖼️ Single Image" tab
2. **Upload** an image (drag & drop or click)
3. **Select** model:
   - Photo? → `RealESRGAN_x4plus`
   - Anime? → `RealESRGAN_x4plus_anime_6B`
4. **Click** "🚀 Enhance Image"
5. **View** before/after comparison!

### Batch Processing

1. **Click** "📦 Batch Processing" tab
2. **Upload** multiple images
3. **Configure** settings once (applies to all)
4. **Click** "🚀 Process Batch"
5. **Download** all results from gallery

---

## 💡 Pro Tips for First Use

### For Photos
```
Model: RealESRGAN_x4plus
Scale: 4x
Auto Tile: ✅ Enabled
FP32: ❌ Disabled (FP16 is fine)
Face Enhancement: ✅ Enabled (for portraits)
```

### For Anime/Illustrations
```
Model: RealESRGAN_x4plus_anime_6B
Scale: 4x
Auto Tile: ✅ Enabled
FP32: ❌ Disabled
Face Enhancement: ❌ Disabled
```

### For Quick Tests
```
Model: RealESRGAN_x2plus
Scale: 2x
Auto Tile: ✅ Enabled
FP32: ❌ Disabled
```

---

## 🎨 Using Presets (30 seconds)

1. **Go to** "🎨 Presets" tab
2. **Select** a preset:
   - **High Quality** - Best quality, slower
   - **Fast Processing** - Quick results
   - **Anime Optimized** - For anime content
   - **Portrait Enhancement** - Face enhancement enabled
3. **Click** "📥 Load Preset"
4. **Go back** to Single Image tab
5. Settings are now applied!

### Create Your Own Preset

1. Configure settings in Single Image tab
2. Go to Presets tab
3. Enter a name (e.g., "My 4K Settings")
4. Click "💾 Save as New Preset"
5. It's saved forever!

---

## 🔧 Common First-Time Issues

### "CUDA out of memory"
**Solution:** Enable Auto Tile (it's enabled by default)

### "Model not found"
**Solution:** Models auto-download on first use. Wait for download to complete.

### Processing is slow
**Solution:**
- Make sure you selected a GPU (not CPU)
- First run is slower (loading model), subsequent runs are faster
- Try FP16 mode instead of FP32

### Can't see before/after comparison
**Solution:** The comparison appears after enhancement completes. Make sure processing succeeded (check status box).

---

## 📊 Understanding the Interface

### Tabs Overview

| Tab | Purpose | Use When |
|-----|---------|----------|
| 🖼️ Single Image | Process one image | Testing settings, single files |
| 📦 Batch Processing | Process multiple images | Bulk enhancement |
| 🎨 Presets | Save/load settings | Quick access to configurations |
| ℹ️ System Info | View GPU, models, logs | Troubleshooting, checking system |

### Settings Explained Simply

- **Model**: Which AI to use (different models = different quality)
- **Scale**: How much bigger? 2x = double, 4x = quadruple
- **Auto Tile**: Let the app handle memory (recommended: ON)
- **GPU Selection**: Which graphics card to use
- **FP32**: Higher quality but slower (recommended: OFF)
- **Face Enhancement**: Special AI for faces (for portraits: ON)

---

## 🎮 Keyboard Shortcuts

- **Tab**: Navigate between controls
- **Enter**: Trigger enhancement (when button is focused)
- **Esc**: Cancel/close dialogs
- **Ctrl+C**: Stop the server (in terminal)

---

## 📁 Where Are My Files?

### Input Files
```
g:\My Drive\scripts\realesrgan\inputs\
```
Your uploaded images are saved here with timestamps.

### Output Files
```
g:\My Drive\scripts\realesrgan\outputs\
```
Enhanced images are saved here with `enhanced_` prefix.

### Settings & Logs
```
g:\My Drive\scripts\realesrgan\settings.json    # Your settings
g:\My Drive\scripts\realesrgan\presets.json     # Your presets
g:\My Drive\scripts\realesrgan\realesrgan_app.log  # App logs
```

---

## 🚨 Emergency Troubleshooting

### App Won't Start
```bash
# Check environment
conda activate realesrgan
conda list | grep gradio  # Should show gradio 5.23.3

# Reinstall gradio
pip install --force-reinstall gradio==5.23.3

# Try again
python app_ultimate.py
```

### Import Errors
```bash
cd Real-ESRGAN
pip install -r requirements.txt
python setup.py develop
cd ..
```

### CUDA Errors
```bash
# Test CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Still Having Issues?
1. Check logs: System Info tab → Refresh Logs
2. Check console output where you ran `python app_ultimate.py`
3. Try the improved version: `python app_improved.py`
4. Fall back to original: `python app.py`

---

## 🎯 Next Steps

Once you're comfortable:

1. **Explore Models**: Try all 6 models to see which works best for your content
2. **Create Presets**: Save your favorite settings for quick access
3. **Batch Process**: Process multiple images at once
4. **Check System Info**: Monitor GPU usage and logs
5. **Read Full Guide**: Check README_ULTIMATE.md for advanced features

---

## 📚 Quick Reference

### Best Models by Content Type

| Content Type | Recommended Model | Alternative |
|-------------|-------------------|-------------|
| Photos | RealESRGAN_x4plus | RealESRNet_x4plus |
| Anime | RealESRGAN_x4plus_anime_6B | realesr-animevideov3 |
| Illustrations | RealESRGAN_x4plus_anime_6B | RealESRGAN_x4plus |
| Quick 2x | RealESRGAN_x2plus | - |
| Noisy Images | realesr-general-x4v3 | - |

### Processing Time Estimates (4K image, RTX 3060)

| Configuration | Time |
|--------------|------|
| 2x scale, FP16 | ~5-10 seconds |
| 4x scale, FP16 | ~10-20 seconds |
| 4x scale, FP32 | ~20-40 seconds |
| 4x scale, FP32 + Face | ~30-60 seconds |

*Times vary based on GPU, image size, and complexity*

---

## 💬 Tips from the Community

### Maximize Quality
1. Use original/uncompressed sources
2. Enable FP32 for final output
3. Use face enhancement for portraits
4. Process at 4x scale
5. Save as PNG (not JPEG)

### Maximize Speed
1. Use FP16 mode
2. Process at 2x scale
3. Use fastest model (animevideov3)
4. Disable face enhancement
5. Use batch processing for multiple files

### Manage Memory
1. Keep auto-tile enabled
2. Close other applications
3. Use FP16 mode
4. Process smaller batches
5. Monitor GPU in System Info

---

**You're all set! Start enhancing! 🚀**

For detailed documentation, see [README_ULTIMATE.md](README_ULTIMATE.md)
