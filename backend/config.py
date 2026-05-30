"""Shared configuration: ComfyUI endpoints, on-disk paths, model filenames, and the
tunable constants used across the backend. Pure data — no imports from sibling modules."""
from __future__ import annotations

from pathlib import Path

# ---------- ComfyUI endpoints ----------
COMFY_HTTP = "http://127.0.0.1:8188"
COMFY_WS = "ws://127.0.0.1:8188/ws"
COMFY_ROOT = Path("/home/yunus/Documents/comfyui")
COMFY_OUTPUT = COMFY_ROOT / "output"
COMFY_INPUT = COMFY_ROOT / "input"

# ---------- Studio paths ----------
STUDIO_ROOT = Path(__file__).resolve().parent
THUMB_DIR = STUDIO_ROOT / "thumbs"
HISTORY_FILE = STUDIO_ROOT / "history.json"
# Persistent character library: saved character canons + their reference images,
# so users can re-use the same protagonist across multiple storybooks.
CHARACTER_LIBRARY_FILE = STUDIO_ROOT / "characters.json"
CHARACTER_LIBRARY_DIR = STUDIO_ROOT / "character_refs"

# ---------- Wan video models ----------
T5_MODEL = "umt5-xxl-enc-bf16.safetensors"
VAE_MODEL = "Wan2_1_VAE_bf16.safetensors"
CLIP_VISION_MODEL = "open-clip-xlm-roberta-large-vit-huge-14_visual_fp32.safetensors"
# Wan 2.2 14B (MoE — two experts) drive storybook page animation
WAN22_I2V_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
WAN22_I2V_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"

# ---------- Image generation models ----------
FLUX_MODEL = "flux1-schnell-fp8.safetensors"
FLUX_T5 = "t5xxl_fp8_e4m3fn.safetensors"
FLUX_CLIP_L = "clip_l.safetensors"
FLUX_VAE = "ae.safetensors"
FLUX_KONTEXT_MODEL = "flux1-dev-kontext_fp8_scaled.safetensors"
SDXL_MODEL = "sd_xl_base_1.0.safetensors"
SDXL_LIGHTNING_LORA = "sdxl_lightning_4step_lora.safetensors"
UPSCALER_MODEL = "4x-UltraSharp.pth"

DEFAULT_NEGATIVE = "blurry, low quality, distorted, bad anatomy, watermark, text, jpeg artifacts, deformed, ugly"

# ---------- Tunables ----------
# Denoise for the img2img Kontext end-keyframe edit. Lower = background locked harder but
# pose moves less; higher = pose moves more but the room drifts. 0.6 keeps the room while
# letting the character change pose. Tune if actions look too subtle (raise) or the
# background still drifts (lower).
KONTEXT_EDIT_DENOISE = 0.6

# Crossfade between stitched clips: ~3 frames @ 16fps — short enough to read as a smooth
# seam, not a visible dissolve. Paired with last-frame chaining (the boundary frames are
# near-identical), it mainly absorbs the small motion-velocity discontinuity at each cut.
XFADE_DUR = 0.18

STYLE_PREFIXES = {
    "pixar":      "Pixar 3D animation style, soft warm lighting, high detail, cinematic",
    "watercolor": "watercolor children's book illustration, soft pastels, hand-painted texture",
    "anime":      "anime children's book illustration, soft pastel colors, cel-shaded, Studio Ghibli inspired",
    "cartoon":    "friendly cartoon illustration, bold outlines, bright cheerful colors",
}
STORY_ASPECT_DIMS = {
    # ~30% more pixels per axis than the old 832×480 for noticeably sharper output.
    # Wan time scales roughly with pixel count.
    "landscape": (1024, 576),
    "square":    (768, 768),
    "portrait":  (576, 1024),
}

# Pages used in smoke-test mode (StorybookParams.smoke). 3 gives: an opening ambient page
# (pure I2V), a same-location beat (img2img background-lock), and one more — so a single
# run exercises both keyframe paths and ≥2 stitched seams.
SMOKE_PAGES = 3

# Persistent WebSocket client id the backend uses to monitor every gen's ComfyUI events.
MONITOR_CLIENT_ID = "wan-studio-monitor"
