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

# ---------- Narration (Kokoro-82M, CPU) ----------
# Narration is rendered before any video: each page's spoken length sets that page's
# clip length, so the picture is cut to the voice instead of every page being a
# uniform 5 seconds.
TTS_VOICE = "af_heart"      # warm female narrator; bf_emma is the other cached voice
TTS_LANG = "a"              # 'a' = American English (matches af_* voices)
TTS_SPEED = 0.92            # slightly under 1.0 — storybook pacing for young children
TTS_SAMPLE_RATE = 24000     # Kokoro's native rate
NARRATION_DIR = STUDIO_ROOT / "narration"    # per-page WAVs, keyed by gen id

# Silence padded around each page's narration inside its clip, in seconds. Gives the
# picture a beat to settle before the voice starts and after it ends, which is what
# makes a page turn feel unhurried rather than clipped.
NARRATION_LEAD_IN = 0.35
NARRATION_TAIL = 0.55

# Clip length is derived from narration length, then clamped. Wan 2.2 is trained around
# 81 frames @16fps (~5s); 49 is the shortest that still reads as a shot rather than a
# blip. Pages whose narration runs longer than the cap hold on the last frame at stitch.
# 97 frames (6.06s) rather than Wan's canonical 81: a natural picture-book line runs
# 12-18 words ~= 4.5-6.5s of speech, and covering that from 81 frames would need a
# 1.4x slow-down on every page. 97 keeps the retime gentle at ~20% more render time.
VID_FPS = 16
MIN_VID_FRAMES = 49
MAX_VID_FRAMES = 97

# ---------- Finishing (interpolation, grade, grain, mix) ----------
# Wan renders 16fps; RIFE doubles it to 32 and the delivery mux conforms to 24fps.
# This is the cheapest single change that stops the output reading as an "AI clip",
# and it needs no re-rendering. Interpolation runs per clip BEFORE stitching so it
# never smears across a cut.
INTERP_ENABLED = True
INTERP_CKPT = "rife47.pth"
INTERP_MULTIPLIER = 2
OUTPUT_FPS = 24

# One LUT-free grade pass: each clip is nudged toward the run's median exposure and
# colour so separately-generated shots feel like one piece of film, then a whisper of
# grain goes over the finished cut to break up the plasticky diffusion sheen.
GRADE_ENABLED = True
GRAIN_STRENGTH = 6          # ffmpeg noise filter strength; 0 disables

# Loudness target for the delivered mix (EBU R128, the usual streaming/podcast target).
LOUDNORM_I = -16.0
LOUDNORM_TP = -1.5
LOUDNORM_LRA = 11.0

# Optional music bed: drop a WAV/MP3 here and it is mixed under the narration and
# ducked automatically. Left empty by default — generative scoring (ACE-Step) needs a
# separate model download.
MUSIC_BED = ""
MUSIC_GAIN = 0.18           # linear gain applied to the bed before ducking

# ---------- Take selection ----------
# Every page is drafted several times and the best draft's seed is used for the real
# render. Drafts keep the FULL frame count — motion is what we're choosing between — but
# drop to half resolution and low steps, which costs about a tenth of a final render, so
# best-of-N adds roughly 40% to a run instead of multiplying it.
#
# Caveat worth remembering: a draft shares its seed with the final render but not its
# resolution or step count, so it predicts composition and gross motion well and fine
# detail poorly. We are picking a motion arc, not grading finished frames.
TAKES_ENABLED = True
DRAFT_TAKES = 3             # drafts per page; 1 (or TAKES_ENABLED=False) restores old behaviour
DRAFT_STEPS = 8
DRAFT_SCALE = 0.5           # of the final resolution; keep to multiples of 16 after scaling
DRAFT_SEED_STRIDE = 1013    # prime-ish gap so takes don't land on adjacent, similar seeds

# Persistent WebSocket client id the backend uses to monitor every gen's ComfyUI events.
MONITOR_CLIENT_ID = "wan-studio-monitor"
