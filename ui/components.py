"""Gradio UI component definitions and interface creation"""

from __future__ import annotations
import gradio as gr
import torch
from typing import Tuple, Dict, Optional
import logging

# Import engine components
from engine import IntelligentEnhancer, ModelManager, SettingsManager, enhance_image_direct, process_batch
from utils.gpu import GPUDetector

logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "RealESRGAN_x4plus": {
        "name": "General Purpose 4x",
        "description": "Best for photos and realistic images",
        "scale": 4,
        "model_class": "RRDBNet",
        "params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 23, "num_grow_ch": 32, "scale": 4},
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "name": "Anime/Illustration 4x",
        "description": "Optimized for anime and illustrations",
        "scale": 4,
        "model_class": "RRDBNet",
        "params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 6, "num_grow_ch": 32, "scale": 4},
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    },
    "RealESRNet_x4plus": {
        "name": "Alternative 4x",
        "description": "Different architecture for photos",
        "scale": 4,
        "model_class": "RRDBNet",
        "params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 23, "num_grow_ch": 32, "scale": 4},
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    },
    "RealESRGAN_x2plus": {
        "name": "General Purpose 2x",
        "description": "Smaller upscale, faster processing",
        "scale": 2,
        "model_class": "RRDBNet",
        "params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_block": 23, "num_grow_ch": 32, "scale": 2},
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    },
    "realesr-animevideov3": {
        "name": "Anime Video 4x",
        "description": "Fast, optimized for anime videos",
        "scale": 4,
        "model_class": "SRVGGNetCompact",
        "params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_conv": 16, "upscale": 4, "act_type": "prelu"},
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    },
    "realesr-general-x4v3": {
        "name": "General 4x with Denoise",
        "description": "Advanced control with denoise strength",
        "scale": 4,
        "model_class": "SRVGGNetCompact",
        "params": {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 64, "num_conv": 32, "upscale": 4, "act_type": "prelu"},
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "url_dni": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    },
}

DEFAULT_PRESETS = {
    "High Quality": {
        "model": "RealESRGAN_x4plus",
        "scale": 4,
        "tile": 0,
        "fp32": True,
        "face_enhance": False,
        "auto_tile": True,
        "denoise_strength": 0.5,
    },
    "Fast Processing": {
        "model": "RealESRGAN_x2plus",
        "scale": 2,
        "tile": 512,
        "fp32": False,
        "face_enhance": False,
        "auto_tile": True,
        "denoise_strength": 0.5,
    },
    "Anime Optimized": {
        "model": "RealESRGAN_x4plus_anime_6B",
        "scale": 4,
        "tile": 0,
        "fp32": False,
        "face_enhance": False,
        "auto_tile": True,
        "denoise_strength": 0.5,
    },
    "Portrait Enhancement": {
        "model": "RealESRGAN_x4plus",
        "scale": 4,
        "tile": 0,
        "fp32": True,
        "face_enhance": True,
        "auto_tile": True,
        "denoise_strength": 0.5,
    },
}

# Custom CSS
CUSTOM_CSS = """
:root {
    --primary-color: #7C3AED;
    --secondary-color: #EC4899;
    --success-color: #10B981;
}

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.header-text {
    color: #7C3AED !important;
    font-weight: 800;
    font-size: 2.5em;
    text-align: center;
    margin-bottom: 0.5rem;
}

.header-subtitle {
    text-align: center;
    font-size: 1.1em;
    color: #6B7280;
    margin-bottom: 2rem;
}

@media (prefers-color-scheme: dark) {
    .header-text {
        color: #A78BFA !important;
    }
    .header-subtitle {
        color: #9CA3AF !important;
    }
}

.primary-btn {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    transition: transform 0.2s !important;
}

.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(124, 58, 237, 0.3) !important;
}

.ai-analysis-box {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(124, 58, 237, 0.1));
    border-left: 4px solid var(--success-color);
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
"""


def create_comparison_slider(before_path: str, after_path: str):
    """
    Create interactive before/after comparison using native HTML5 input range slider

    Returns both images as PIL Images for Gradio to display
    """
    from PIL import Image

    try:
        before_img = Image.open(before_path)
        after_img = Image.open(after_path)

        return before_img, after_img
    except Exception as e:
        return None, None


def create_interface(logger: logging.Logger, config) -> gr.Blocks:
    """
    Create the main Gradio interface

    Args:
        logger: Logger instance
        config: Configuration object

    Returns:
        gr.Blocks: Gradio interface
    """
    # Initialize managers
    settings_manager = SettingsManager(
        config.CONFIG_FILE, config.PRESETS_FILE, DEFAULT_PRESETS
    )
    model_manager = ModelManager(
        MODELS, config.WEIGHTS_DIR, config.VERIFY_MODEL_CHECKSUMS
    )
    intelligent_enhancer = IntelligentEnhancer()

    settings = settings_manager.load_settings()
    presets = settings_manager.load_presets()

    available_gpus = GPUDetector.get_available_gpus()
    gpu_choices = [name for _, name in available_gpus]
    gpu_ids = {name: gpu_id for gpu_id, name in available_gpus}

    with gr.Blocks(title="Real-ESRGAN Ultimate", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:

        gr.HTML("""
            <div class="header-text">
                🤖 Real-ESRGAN Ultimate
            </div>
            <p class="header-subtitle">
                AI-Powered Intelligent Auto-Enhancement & Full Video Support
            </p>
        """)

        with gr.Tabs():
            # Image Enhancement Tab
            with gr.Tab("🖼️ Image Enhancement"):
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_image = gr.Image(label="📤 Upload Image", type="pil")

                        with gr.Accordion("🤖 AI Auto-Enhancement", open=True):
                            gr.Markdown("Let AI analyze your image and pick the best settings automatically!")

                            analyze_btn = gr.Button(
                                "🔍 Analyze Image & Get Smart Settings",
                                variant="primary",
                                size="lg",
                                elem_classes="primary-btn"
                            )

                            analysis_output = gr.Markdown(
                                label="Analysis Results",
                                visible=False,
                                elem_classes="ai-analysis-box"
                            )

                            apply_ai_settings_btn = gr.Button(
                                "✨ Apply AI Settings",
                                variant="secondary",
                                visible=False
                            )

                        with gr.Accordion("🎯 Manual Settings", open=True):
                            model_name_img = gr.Dropdown(
                                choices=list(MODELS.keys()),
                                value=settings.get("model", "RealESRGAN_x4plus"),
                                label="Model"
                            )

                            scale_img = gr.Slider(
                                2, 4, value=settings.get("scale", 4), step=1,
                                label="📏 Scale Factor"
                            )

                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            gpu_select_img = gr.Dropdown(
                                choices=gpu_choices,
                                value=gpu_choices[0] if gpu_choices else "CPU Only",
                                label="🎮 GPU Selection"
                            )

                            auto_tile_img = gr.Checkbox(
                                label="🔄 Auto Tile",
                                value=settings.get("auto_tile", True)
                            )

                            tile_img = gr.Slider(
                                0, 512, value=0, step=32,
                                label="🔲 Manual Tile Size",
                                interactive=not settings.get("auto_tile", True)
                            )

                            fp32_img = gr.Checkbox(
                                label="⚡ FP32 Precision",
                                value=settings.get("fp32", False)
                            )

                            face_enhance_img = gr.Checkbox(
                                label="👤 Face Enhancement",
                                value=settings.get("face_enhance", False)
                            )

                            denoise_img = gr.Slider(
                                0, 1, value=0.5, step=0.1,
                                label="🎛️ Denoise Strength",
                                visible=False
                            )

                    with gr.Column(scale=1):
                        comparison_slider = gr.Gallery(
                            label="📊 Before → After Comparison",
                            columns=2,
                            rows=1,
                            height="auto"
                        )

                        metrics_display = gr.JSON(
                            label="📈 Quality Metrics",
                            visible=False
                        )

                with gr.Row():
                    enhance_btn = gr.Button(
                        "🚀 Enhance Image",
                        variant="primary",
                        size="lg",
                        elem_classes="primary-btn"
                    )
                    save_settings_btn = gr.Button("💾 Save Settings", variant="secondary")

                status_img = gr.Textbox(label="📝 Status", interactive=False, lines=2)
                ai_settings_state = gr.State(value={})

                # AI Analysis callback
                def run_ai_analysis(image):
                    if image is None:
                        return "❌ Please upload an image first", gr.update(visible=False), {}

                    result = intelligent_enhancer.get_intelligent_settings(image)
                    output = f"""
### 🤖 AI Analysis Complete!

**Image Analysis:**
- Content Type: {result['analysis']['content_type'].title()}
- Faces Detected: {result['analysis']['faces']}
- Noise Level: {result['analysis']['noise_level']}
- Resolution: {result['analysis']['resolution']}

**Recommended Settings:**
"""
                    for reason in result['reasoning']:
                        output += f"\n- {reason}"

                    return output, gr.update(visible=True), result['settings']

                analyze_btn.click(
                    run_ai_analysis,
                    inputs=[upload_image],
                    outputs=[analysis_output, apply_ai_settings_btn, ai_settings_state]
                )

                # Apply AI settings callback
                def apply_ai_settings(ai_settings):
                    if not ai_settings:
                        return [gr.update()] * 6 + ["❌ No AI settings to apply"]

                    return [
                        gr.update(value=ai_settings.get("model", "RealESRGAN_x4plus")),
                        gr.update(value=ai_settings.get("scale", 4)),
                        gr.update(value=ai_settings.get("auto_tile", True)),
                        gr.update(value=ai_settings.get("tile", 0)),
                        gr.update(value=ai_settings.get("fp32", False)),
                        gr.update(value=ai_settings.get("face_enhance", False)),
                        "✅ AI settings applied! Ready to enhance."
                    ]

                apply_ai_settings_btn.click(
                    apply_ai_settings,
                    inputs=[ai_settings_state],
                    outputs=[model_name_img, scale_img, auto_tile_img, tile_img,
                            fp32_img, face_enhance_img, status_img]
                )

                # Enhancement callback
                def enhance_single_image(img, model, scale, tile, fp32, face, auto_tile, denoise, gpu_name):
                    gpu_id = gpu_ids.get(gpu_name, -1)
                    enhanced, original, status, metrics = enhance_image_direct(
                        img, model, scale, tile, fp32, face, auto_tile, denoise, gpu_id,
                        model_manager, config
                    )

                    if enhanced and original:
                        # Load and display both images
                        before_img, after_img = create_comparison_slider(original, enhanced)
                        if before_img and after_img:
                            gallery_output = [(before_img, "Before"), (after_img, "After")]
                            return gallery_output, status, metrics, gr.update(visible=True)
                        else:
                            return [], status, None, gr.update(visible=False)
                    else:
                        return [], status, None, gr.update(visible=False)

                enhance_btn.click(
                    enhance_single_image,
                    inputs=[upload_image, model_name_img, scale_img, tile_img, fp32_img,
                            face_enhance_img, auto_tile_img, denoise_img, gpu_select_img],
                    outputs=[comparison_slider, status_img, metrics_display, metrics_display]
                )

                # Model change callback
                def update_denoise_visibility(model):
                    return gr.update(visible=(model == "realesr-general-x4v3"))

                def update_tile_state(auto_tile):
                    return gr.update(interactive=not auto_tile)

                model_name_img.change(update_denoise_visibility, inputs=[model_name_img], outputs=[denoise_img])
                auto_tile_img.change(update_tile_state, inputs=[auto_tile_img], outputs=[tile_img])

                # Save settings callback
                def save_current_settings(model, scale, fp32, face, auto_tile, gpu_name):
                    settings = {
                        "model": model,
                        "scale": scale,
                        "fp32": fp32,
                        "face_enhance": face,
                        "auto_tile": auto_tile,
                        "gpu_id": gpu_ids.get(gpu_name, -1),
                    }
                    return settings_manager.save_settings(settings)

                save_settings_btn.click(
                    save_current_settings,
                    inputs=[model_name_img, scale_img, fp32_img, face_enhance_img, auto_tile_img, gpu_select_img],
                    outputs=[status_img]
                )

            # ========== VIDEO ENHANCEMENT TAB ==========
            with gr.Tab("🎬 Video Enhancement"):
                gr.Markdown("""
                ### Enhance videos with Real-ESRGAN
                **Note:** Video processing uses subprocess method and may take significant time depending on video length.
                """)

                with gr.Row():
                    upload_video = gr.Video(label="📤 Upload Video (.mp4)", format="mp4")
                    enhanced_video = gr.Video(label="✨ Enhanced Video")

                with gr.Row():
                    model_name_vid = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value=settings.get("model", "RealESRGAN_x4plus"),
                        label="🎯 Model",
                        info="For anime videos, use 'realesr-animevideov3'"
                    )
                    scale_vid = gr.Slider(
                        2, 4, value=settings.get("scale", 4), step=1,
                        label="📏 Scale Factor"
                    )
                    tile_vid = gr.Slider(
                        0, 512, value=400, step=32,
                        label="🔲 Tile Size",
                        info="Recommended: 400-512 for videos"
                    )

                with gr.Row():
                    fp32_vid = gr.Checkbox(
                        label="⚡ FP32 Precision",
                        value=settings.get("fp32", False)
                    )
                    face_enhance_vid = gr.Checkbox(
                        label="👤 Face Enhancement",
                        value=settings.get("face_enhance", False)
                    )

                denoise_vid = gr.Slider(
                    0, 1, value=0.5, step=0.1,
                    label="🎛️ Denoise Strength",
                    visible=False
                )

                model_name_vid.change(update_denoise_visibility, inputs=[model_name_vid], outputs=[denoise_vid])

                run_button_vid = gr.Button("🚀 Enhance Video", variant="primary", size="lg", elem_classes="primary-btn")
                status_vid = gr.Textbox(label="📝 Status", interactive=False, lines=2)

                gr.Markdown("""
                ⚠️ **Video Processing Tips:**
                - Processing time depends on video length and resolution
                - For long videos (>5 min), consider using 2x scale
                - Use tile size 400-512 to prevent memory issues
                - Audio is automatically preserved
                """)

                def wrapped_enhance_video(vid, model, s, t, f, fe, denoise):
                    if vid is None:
                        return None, "❌ Please upload a video first"
                    # TODO: Implement direct video processing
                    return None, "⏳ Video processing coming soon! Use command line for now."

                run_button_vid.click(
                    wrapped_enhance_video,
                    inputs=[upload_video, model_name_vid, scale_vid, tile_vid, fp32_vid, face_enhance_vid, denoise_vid],
                    outputs=[enhanced_video, status_vid]
                )

            # Batch Processing Tab
            with gr.Tab("📦 Batch Processing"):
                gr.Markdown("### Process multiple images at once\nUpload multiple images and process them with the same settings.")

                with gr.Row():
                    batch_upload = gr.File(
                        label="📤 Upload Multiple Images",
                        file_count="multiple",
                        file_types=["image"]
                    )

                with gr.Row():
                    with gr.Column():
                        batch_model = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="RealESRGAN_x4plus",
                            label="Model"
                        )
                        batch_scale = gr.Slider(2, 4, value=4, step=1, label="Scale Factor")

                    with gr.Column():
                        batch_gpu = gr.Dropdown(
                            choices=gpu_choices,
                            value=gpu_choices[0] if gpu_choices else "CPU Only",
                            label="GPU Selection"
                        )
                        batch_auto_tile = gr.Checkbox(label="Auto Tile", value=True)

                with gr.Row():
                    batch_fp32 = gr.Checkbox(label="FP32 Precision", value=False)
                    batch_face = gr.Checkbox(label="Face Enhancement", value=False)

                batch_process_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg", elem_classes="primary-btn")

                batch_status = gr.Textbox(label="Status", interactive=False, lines=3)
                batch_gallery = gr.Gallery(label="Batch Results", columns=4, rows=2)

                def process_batch_images(files, model, scale, auto_tile, fp32, face, gpu_name):
                    if not files:
                        return "❌ No files uploaded", []

                    image_paths = [f.name for f in files]
                    gpu_id = gpu_ids.get(gpu_name, -1)
                    results, status = process_batch(
                        image_paths, model, scale, 0, fp32, face, auto_tile, 0.5, gpu_id,
                        model_manager, config
                    )

                    return status, results

                batch_process_btn.click(
                    process_batch_images,
                    inputs=[batch_upload, batch_model, batch_scale, batch_auto_tile, batch_fp32, batch_face, batch_gpu],
                    outputs=[batch_status, batch_gallery]
                )

            # System Info Tab
            with gr.Tab("ℹ️ System Info"):
                gr.Markdown("### System Information and Help")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🎮 Available GPUs")
                        gpu_info = gr.Markdown("\n".join([f"- {name}" for _, name in available_gpus]))

                        gr.Markdown("#### 📊 PyTorch Info")
                        torch_info = gr.Markdown(f"""
                        - PyTorch Version: {torch.__version__}
                        - CUDA Available: {'✅ Yes' if torch.cuda.is_available() else '❌ No'}
                        - CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}
                        - Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}
                        """)

                    with gr.Column():
                        gr.Markdown("#### 🎯 Available Models")
                        model_table = gr.DataFrame(
                            headers=["Model", "Scale", "Description"],
                            value=[[k, v["scale"], v["description"]] for k, v in MODELS.items()]
                        )

                gr.Markdown("""
                ### 🤖 AI Auto-Enhancement Guide

                The intelligent enhancement analyzes your image to detect:
                1. **Content Type** - Photo, anime, or illustration
                2. **Faces** - Number of faces for portrait enhancement
                3. **Noise Level** - Image quality and denoise needs
                4. **Resolution** - Optimal tile size and scale factor

                **How to use:**
                1. Upload your image
                2. Click "Analyze Image & Get Smart Settings"
                3. Review the AI's recommendations
                4. Click "Apply AI Settings"
                5. Click "Enhance Image"

                The AI will automatically choose the best model, scale, and settings!
                """)

    return demo
