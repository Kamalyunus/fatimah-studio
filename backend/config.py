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

# ---------- VACE (reference-conditioned animation) ----------
# VACE replaces the I2V conditioning path: start/end keyframes go in as masked control
# frames and the character sheet goes in as a reference image, so Wan maintains character
# identity DURING animation instead of only at the keyframes. The VACE modules attach to
# the Wan 2.2 *T2V* base experts (not I2V) via the model loader's vace_model input.
# Disabled: with only first/last frames pinned, the T2V base under-constrains the
# middle of each clip — extra people and background shifts appear mid-scene, then
# snap back at the end keyframe. I2V (whole-clip anchoring to the start frame) is
# more coherent for storybooks until VACE gets denser control frames.
USE_VACE = False                    # storybook Phase B routes to the VACE workflow when True
WAN22_T2V_HIGH = "Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
WAN22_T2V_LOW = "Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors"
WAN22_VACE_HIGH = "Wan2_2_Fun_VACE_module_A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
WAN22_VACE_LOW = "Wan2_2_Fun_VACE_module_A14B_LOW_fp8_e4m3fn_scaled_KJ.safetensors"
VACE_STRENGTH = 1.0                 # vace_context scale; 1.0 = full conditioning

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
# Storybook Wan clips: the cast is fixed per scene, so push hard against Wan's habit of
# walking extra people through the shot mid-clip.
STORYBOOK_NEGATIVE = (
    DEFAULT_NEGATIVE
    + ", extra people, extra characters, crowd, bystanders, strangers, "
    "new person entering the frame, people walking by in the background, duplicate character"
)

# ---------- Tunables ----------
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

# Pages used in smoke-test mode (StorybookParams.smoke). 3 gives a couple of
# same-location beats plus at least one location cut, exercising ≥2 stitched seams.
SMOKE_PAGES = 3

# Persistent WebSocket client id the backend uses to monitor every gen's ComfyUI events.
MONITOR_CLIENT_ID = "wan-studio-monitor"
