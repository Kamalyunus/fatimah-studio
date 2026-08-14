"""Pydantic request/response models for the REST API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateParams(BaseModel):
    """Wan 2.2 I2V parameters. Built internally by the storybook orchestrator and
    fed to build_wan22_i2v_workflow — no longer user-facing."""
    prompt: str
    negative: str = ""
    width: int = 832
    height: int = 480
    frames: int = 49
    steps: int = 20
    cfg: float = 6.0
    shift: float = 5.0
    seed: int = 42
    fps: int = 16
    scheduler: str = "unipc"
    noise_aug: float = 0.0
    image: str = ""           # start frame filename inside ComfyUI's input/
    multi_gpu: bool = True

    # Quality / dual-GPU knobs
    attention_mode: str = "sdpa"        # sdpa | flash_attn_2 | sageattn | radial_sage_attention
    block_swap_count: int = 0           # 0 = disabled. >0 = N transformer blocks offloaded
    block_swap_device: str = "cuda:1"   # cpu | cuda:0 | cuda:1 — where swapped blocks live
    vae_tiling: bool = False             # tiled VAE decode for very high resolutions
    keep_t5_loaded: bool = False        # if true, T5 stays on its device after encoding

    # Wan FLF2V (first-last frame): provide explicit ending frame for guaranteed pose continuity
    end_image: str = ""

    # VACE: character reference sheet (filename inside ComfyUI's input/) injected as a
    # ref image so identity is enforced during animation, not just at the keyframes.
    vace_ref_image: str = ""

    # Wan quality knobs (enable all by default for storybook quality)
    use_slg: bool = True       # Skip Layer Guidance — quality
    use_feta: bool = True      # Enhance-A-Video — motion smoothness
    use_teacache: bool = True  # ~1.5-2x speedup with minor quality cost


class ImageGenerateParams(BaseModel):
    user_name: str = ""
    user_emoji: str = ""

    image_mode: str = Field("create", pattern="^(create|modify)$")
    prompt: str
    negative: str = ""
    width: int = 1024
    height: int = 1024
    seed: int = 0  # 0 = random
    model: str = Field("flux", pattern="^(flux|sdxl)$")
    image: str = ""  # filename inside ComfyUI's input/ — used for modify
    strength: float = 0.6  # for modify: 0=unchanged, 1=fully regenerated. 0.3=subtle, 0.6=moderate, 0.85=bold
    # When true, chain a 2x UltraSharp upscale onto the workflow before saving.
    # Set by /api/image_generate so every user Create/Modify is auto-enhanced;
    # left False for the storybook page generator (those are already pre-sized for Wan).
    auto_upscale: bool = False


class UpscaleParams(BaseModel):
    user_name: str = ""
    user_emoji: str = ""

    image: str
    factor: int = 4  # 2 or 4


class StorybookParams(BaseModel):
    user_name: str = ""
    user_emoji: str = ""

    story: str
    n_pages: int = 12  # 2..15
    style: str = "pixar"  # pixar | watercolor | anime | cartoon
    aspect: str = "landscape"  # landscape | square | portrait
    # Fast smoke-test mode: caps the story to SMOKE_PAGES, runs Wan at low steps/frames,
    # and AUTO-APPROVES the keyframe gate so a full plan→images→video→stitch run completes
    # unattended in a few minutes. Use it to eyeball the background-lock (img2img keyframes),
    # frame-chaining, and crossfade seams before committing to a full-length, full-quality run.
    smoke: bool = False
    # Optional: id of a saved character to re-use across stories. When set, the
    # orchestrator skips generating page 1's start image and uses the saved
    # reference + canon instead, so the protagonist looks the same as previous runs.
    character_id: str = ""


class SavedCharacter(BaseModel):
    """A character that can be re-used across multiple storybooks. Stored under
    `characters.json` with the canonical reference image alongside in `character_refs/`."""
    id: str
    name: str
    canon: dict   # {name, species, colors, features, clothing, accessories}
    character: str   # the prose two-sentence canon (fallback when canon dict is empty)
    ref_filename: str   # PNG in CHARACTER_LIBRARY_DIR
    created_at: float
    source_gen_id: str = ""   # which storybook gen produced this character


class SaveCharacterParams(BaseModel):
    name: str   # short user-facing name for the character (e.g. "Mochi")
    gen_id: str   # storybook gen_id whose page-1 start image becomes the canonical ref


class ImprovePromptParams(BaseModel):
    prompt: str
    style: str = ""   # optional chip label, e.g. "cinematic" — lets the LLM apply a visual style


class HistoryEntry(BaseModel):
    id: str
    prompt_id: str
    filename: str
    mode: str
    prompt: str
    params: dict
    created_at: float
    duration_s: float | None = None


class UseAsInputParams(BaseModel):
    filename: str   # filename in COMFY_OUTPUT to copy into COMFY_INPUT for re-use


class RegenerateKeyframeParams(BaseModel):
    scene_index: int
    frame: str = Field("end", pattern="^(start|end)$")


class RegenerateSceneParams(BaseModel):
    gen_id: str
    scene_index: int
