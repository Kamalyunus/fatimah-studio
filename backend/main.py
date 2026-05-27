"""Fatimah Studio backend — FastAPI relay over ComfyUI.

Exposes a clean REST + WebSocket API for the React frontend:
  POST /api/storybook          plan + illustrate + animate + narrate a storybook
  POST /api/image_generate     Flux/SDXL text-to-image (or image-to-image)
  POST /api/image_upscale      4x-UltraSharp upscale
  POST /api/llm/improve        rewrite a short prompt into a richer one
  POST /api/interrupt          cancel the currently running gen
  POST /api/upload             upload an image (returns {filename})
  GET  /api/state              current active gen + last error
  WS   /api/ws/{client_id}     relay of ComfyUI progress
  GET  /api/history            list past generations
  DELETE /api/history/{id}     delete a generation
  GET  /api/video|image|thumb/{filename}  stream artifacts

Workflow JSON is built server-side so the frontend only sends user-facing params.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import websockets
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import llm
import tts

# ---------- Config ----------

COMFY_HTTP = "http://127.0.0.1:8188"
COMFY_WS = "ws://127.0.0.1:8188/ws"
COMFY_ROOT = Path("/home/yunus/Documents/comfyui")
COMFY_OUTPUT = COMFY_ROOT / "output"
COMFY_INPUT = COMFY_ROOT / "input"

STUDIO_ROOT = Path(__file__).resolve().parent
THUMB_DIR = STUDIO_ROOT / "thumbs"
HISTORY_FILE = STUDIO_ROOT / "history.json"

T5_MODEL = "umt5-xxl-enc-bf16.safetensors"
VAE_MODEL = "Wan2_1_VAE_bf16.safetensors"
CLIP_VISION_MODEL = "open-clip-xlm-roberta-large-vit-huge-14_visual_fp32.safetensors"

# Wan 2.2 14B (MoE — two experts) drive storybook page animation
WAN22_I2V_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
WAN22_I2V_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"

# Image generation models
FLUX_MODEL = "flux1-schnell-fp8.safetensors"
FLUX_T5 = "t5xxl_fp8_e4m3fn.safetensors"
FLUX_CLIP_L = "clip_l.safetensors"
FLUX_VAE = "ae.safetensors"
FLUX_KONTEXT_MODEL = "flux1-dev-kontext_fp8_scaled.safetensors"
SDXL_MODEL = "sd_xl_base_1.0.safetensors"
SDXL_LIGHTNING_LORA = "sdxl_lightning_4step_lora.safetensors"
UPSCALER_MODEL = "4x-UltraSharp.pth"

DEFAULT_NEGATIVE = "blurry, low quality, distorted, bad anatomy, watermark, text, jpeg artifacts, deformed, ugly"


# ---------- Tiny on-disk JSON store ----------

def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _save_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# ---------- Workflow builders ----------

def _block_swap_node(p: "GenerateParams") -> dict | None:
    if p.block_swap_count <= 0:
        return None
    if p.multi_gpu:
        return {
            "class_type": "WanVideoBlockSwapMultiGPU",
            "inputs": {
                "blocks_to_swap": p.block_swap_count,
                "offload_img_emb": False,
                "offload_txt_emb": False,
                "use_non_blocking": True,
                "prefetch_blocks": 1,
                "vace_blocks_to_swap": 0,
                "block_swap_debug": False,
                "swap_device": p.block_swap_device,
            },
        }
    return {
        "class_type": "WanVideoBlockSwap",
        "inputs": {
            "blocks_to_swap": p.block_swap_count,
            "offload_img_emb": False,
            "offload_txt_emb": False,
            "use_non_blocking": True,
            "prefetch_blocks": 1,
        },
    }


def _t5_node(p: "GenerateParams") -> dict:
    if p.multi_gpu:
        return {
            "class_type": "LoadWanVideoT5TextEncoderMultiGPU",
            "inputs": {
                "model_name": T5_MODEL,
                "precision": "bf16",
                "quantization": "disabled",
                "device": "cuda:1",
            },
        }
    return {
        "class_type": "LoadWanVideoT5TextEncoder",
        "inputs": {
            "model_name": T5_MODEL,
            "precision": "bf16",
            "quantization": "disabled",
            "load_device": "offload_device",
        },
    }


def _text_encode_node(p: "GenerateParams") -> dict:
    force_offload = not p.keep_t5_loaded
    if p.multi_gpu:
        return {
            "class_type": "WanVideoTextEncodeMultiGPU",
            "inputs": {
                "positive_prompt": p.prompt,
                "negative_prompt": p.negative or DEFAULT_NEGATIVE,
                "t5": ["t5", 0],
                "force_offload": force_offload,
                "load_device": "cuda:1",
            },
        }
    return {
        "class_type": "WanVideoTextEncode",
        "inputs": {
            "positive_prompt": p.prompt,
            "negative_prompt": p.negative or DEFAULT_NEGATIVE,
            "t5": ["t5", 0],
            "force_offload": force_offload,
            "device": "gpu",
        },
    }


def _interpolate_node(p: "GenerateParams") -> dict:
    """RIFE frame interpolation. Doubles (or more) the frame count, run between decode and save."""
    return {
        "class_type": "RIFE VFI",
        "inputs": {
            "ckpt_name": "rife49.pth",
            "frames": ["decode", 0],
            "clear_cache_after_n_frames": 10,
            "multiplier": max(2, int(p.interpolate_multiplier)),
            "fast_mode": True,
            "ensemble": False,
            "scale_factor": 1.0,
            "dtype": "float16",
            "torch_compile": False,
            "batch_size": 4,
        },
    }


def _save_node(prefix: str, fps: int, source_ref: list | None = None) -> dict:
    return {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": source_ref or ["decode", 0],
            "frame_rate": fps,
            "loop_count": 0,
            "filename_prefix": prefix,
            "format": "video/h264-mp4",
            "pingpong": False,
            "save_output": True,
        },
    }


def _model_loader_node_named(model_filename: str, p: "GenerateParams", with_block_swap: bool, node_key: str = "model_loader") -> dict:
    """Wan 2.2 model loader (named so the two-expert MoE pattern can have separate nodes)."""
    inputs = {
        "model": model_filename,
        "base_precision": "bf16",
        "quantization": "fp8_e4m3fn",
        "load_device": "offload_device",   # cold-load; sampler will hot it
        "attention_mode": p.attention_mode,
    }
    if p.multi_gpu:
        inputs["compute_device"] = "cuda:0"
    if with_block_swap:
        inputs["block_swap_args"] = ["block_swap", 0]
    return {
        "class_type": "WanVideoModelLoaderMultiGPU" if p.multi_gpu else "WanVideoModelLoader",
        "inputs": inputs,
    }


def build_wan22_i2v_workflow(p: "GenerateParams") -> dict:
    """Wan 2.2 14B I2V with MoE: high-noise expert handles steps 0..mid, low-noise mid..end.
    The two experts share VRAM via offload — only one is on cuda:0 at a time."""
    mg = p.multi_gpu
    total_steps = max(2, int(p.steps))
    boundary = total_steps // 2
    block_swap = _block_swap_node(p)

    wf: dict = {}
    if block_swap:
        wf["block_swap"] = block_swap

    # Two Wan 2.2 experts (cold-loaded; sampler activates and offloads each in turn)
    wf["model_high"] = _model_loader_node_named(WAN22_I2V_HIGH, p, with_block_swap=bool(block_swap), node_key="model_high")
    wf["model_low"]  = _model_loader_node_named(WAN22_I2V_LOW,  p, with_block_swap=bool(block_swap), node_key="model_low")

    # T5 + VAE same as Wan 2.1 (compatible)
    wf["t5"] = _t5_node(p)
    wf["vae"] = {"class_type": "WanVideoVAELoader", "inputs": {"model_name": VAE_MODEL, "precision": "bf16"}}

    # CLIP-vision (for I2V image conditioning) — same as 2.1
    if p.multi_gpu:
        wf["clip_vision"] = {
            "class_type": "LoadWanVideoClipTextEncoderMultiGPU",
            "inputs": {"model_name": CLIP_VISION_MODEL, "precision": "fp16", "device": "cuda:1"},
        }
    else:
        wf["clip_vision"] = {
            "class_type": "LoadWanVideoClipTextEncoder",
            "inputs": {"model_name": CLIP_VISION_MODEL, "precision": "fp16", "load_device": "offload_device"},
        }

    wf["load_image"] = {"class_type": "LoadImage", "inputs": {"image": p.image}}
    wf["text_encode"] = _text_encode_node(p)

    wf["clip_vision_encode"] = {
        "class_type": "WanVideoClipVisionEncodeMultiGPU" if mg else "WanVideoClipVisionEncode",
        "inputs": {
            "clip_vision": ["clip_vision", 0],
            "image_1": ["load_image", 0],
            "strength_1": 1.0, "strength_2": 1.0,
            "crop": "center", "combine_embeds": "average",
            "force_offload": True,
            **({"load_device": "cuda:1"} if mg else {}),
        },
    }
    # FLF2V: if an end_image is given, load it and pass to i2v_encode for first+last-frame guidance
    i2v_inputs = {
        "width": p.width, "height": p.height, "num_frames": p.frames,
        "noise_aug_strength": p.noise_aug,
        "start_latent_strength": 1.0, "end_latent_strength": 1.0,
        "force_offload": True,
        "vae": ["vae", 0],
        "clip_embeds": ["clip_vision_encode", 0],
        "start_image": ["load_image", 0],
    }
    if p.end_image:
        wf["load_end_image"] = {"class_type": "LoadImage", "inputs": {"image": p.end_image}}
        i2v_inputs["end_image"] = ["load_end_image", 0]
    wf["i2v_encode"] = {
        "class_type": "WanVideoImageToVideoEncode",
        "inputs": i2v_inputs,
    }

    # Quality knobs — wired into both samplers
    quality_args: dict = {}
    if p.use_slg:
        wf["slg"] = {
            "class_type": "WanVideoSLG",
            "inputs": {"blocks": "9", "start_percent": 0.1, "end_percent": 0.5},
        }
        quality_args["slg_args"] = ["slg", 0]
    if p.use_feta:
        wf["feta"] = {
            "class_type": "WanVideoEnhanceAVideo",
            "inputs": {"weight": 2.0, "start_percent": 0.0, "end_percent": 1.0},
        }
        quality_args["feta_args"] = ["feta", 0]
    if p.use_teacache:
        wf["teacache"] = {
            "class_type": "WanVideoTeaCache",
            "inputs": {
                "rel_l1_thresh": 0.20,  # Wan recommended
                "start_step": 1,
                "end_step": max(2, total_steps - 1),
                "cache_device": "main_device",
                "use_coefficients": True,
                "mode": "e",
            },
        }
        quality_args["cache_args"] = ["teacache", 0]

    # High-noise expert: steps 0..boundary
    sampler_inputs_common = {
        "image_embeds": ["i2v_encode", 0],
        "text_embeds": ["text_encode", 0],
        "steps": total_steps,
        "cfg": p.cfg,
        "shift": p.shift,
        "seed": p.seed,
        "force_offload": True,
        "scheduler": p.scheduler,
        "riflex_freq_index": 0,
        **quality_args,
    }
    if mg:
        sampler_inputs_common["compute_device"] = "cuda:0"

    wf["sampler_high"] = {
        "class_type": "WanVideoSamplerMultiGPU" if mg else "WanVideoSampler",
        "inputs": {
            **sampler_inputs_common,
            "model": ["model_high", 0],
            "start_step": 0,
            "end_step": boundary,
        },
    }
    wf["sampler_low"] = {
        "class_type": "WanVideoSamplerMultiGPU" if mg else "WanVideoSampler",
        "inputs": {
            **sampler_inputs_common,
            "model": ["model_low", 0],
            "samples": ["sampler_high", 0],
            "start_step": boundary,
            "end_step": total_steps,
            "add_noise_to_samples": False,
        },
    }

    wf["decode"] = {
        "class_type": "WanVideoDecode",
        "inputs": {
            "vae": ["vae", 0],
            "samples": ["sampler_low", 0],
            "enable_vae_tiling": p.vae_tiling,
            "tile_x": 272, "tile_y": 272,
            "tile_stride_x": 144, "tile_stride_y": 128,
        },
    }
    if p.interpolate:
        wf["interpolate"] = _interpolate_node(p)
        wf["save"] = _save_node("wan_studio_i2v_v22",
                                 p.fps * max(2, int(p.interpolate_multiplier)),
                                 ["interpolate", 0])
    else:
        wf["save"] = _save_node("wan_studio_i2v_v22", p.fps)
    return wf


# ---------- Image workflow builders ----------

def build_flux_kontext_workflow(prompt: str, width: int, height: int, seed: int, reference_image: str, steps: int = 20) -> dict:
    """Flux Kontext: in-context generation conditioned on a reference image.
    Use a reference image of the character; prompt describes the new scene.
    The character's appearance is preserved by the reference latents themselves."""
    return {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": FLUX_KONTEXT_MODEL, "weight_dtype": "default"},
        },
        "clip": {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": FLUX_CLIP_L, "clip_name2": FLUX_T5, "type": "flux"},
        },
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "load_ref": {"class_type": "LoadImage", "inputs": {"image": reference_image}},
        "scale_ref": {
            "class_type": "FluxKontextImageScale",
            "inputs": {"image": ["load_ref", 0]},
        },
        "encode_ref": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["scale_ref", 0], "vae": ["vae", 0]},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["clip", 0]},
        },
        "ref_latent": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["positive", 0],
                "latent": ["encode_ref", 0],
            },
        },
        "guidance": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["ref_latent", 0], "guidance": 2.5},
        },
        "negative": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["positive", 0]},
        },
        "empty_latent": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["unet", 0],
                "positive": ["guidance", 0],
                "negative": ["negative", 0],
                "latent_image": ["empty_latent", 0],
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "seed": seed,
            },
        },
        "decode": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["decode", 0], "filename_prefix": "wan_studio_image_flux"},
        },
    }


def build_flux_image_workflow(p: "ImageGenerateParams") -> dict:
    """Flux.1 [schnell] text-to-image (or image-to-image when p.image is set)."""
    is_i2i = p.image_mode == "modify" and p.image
    seed = p.seed if p.seed else int(time.time() * 1000) % (2**31)

    wf = {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": FLUX_MODEL,
                "weight_dtype": "fp8_e4m3fn",
            },
        },
        "clip": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": FLUX_CLIP_L,
                "clip_name2": FLUX_T5,
                "type": "flux",
            },
        },
        "vae": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": FLUX_VAE},
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": p.prompt, "clip": ["clip", 0]},
        },
        "guidance": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["positive", 0], "guidance": 3.5},
        },
        # Schnell ignores negative, but KSampler needs one
        "negative": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["positive", 0]},
        },
    }

    if is_i2i:
        wf["load_image"] = {"class_type": "LoadImage", "inputs": {"image": p.image}}
        wf["encode"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["load_image", 0], "vae": ["vae", 0]},
        }
        latent_ref = ["encode", 0]
        denoise = float(p.strength)
    else:
        wf["empty_latent"] = {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": p.width, "height": p.height, "batch_size": 1},
        }
        latent_ref = ["empty_latent", 0]
        denoise = 1.0

    wf["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["unet", 0],
            "positive": ["guidance", 0],
            "negative": ["negative", 0],
            "latent_image": latent_ref,
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": denoise,
            "seed": seed,
        },
    }
    wf["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]},
    }
    wf["save"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["decode", 0],
            "filename_prefix": "wan_studio_image_flux",
        },
    }
    return wf


def build_sdxl_image_workflow(p: "ImageGenerateParams") -> dict:
    """SDXL base + Lightning 4-step LoRA. Slightly faster than Flux, lower quality on faces/text."""
    is_i2i = p.image_mode == "modify" and p.image
    seed = p.seed if p.seed else int(time.time() * 1000) % (2**31)

    wf = {
        "checkpoint": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": SDXL_MODEL},
        },
        "lora": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["checkpoint", 0],
                "clip": ["checkpoint", 1],
                "lora_name": SDXL_LIGHTNING_LORA,
                "strength_model": 1.0,
                "strength_clip": 1.0,
            },
        },
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": p.prompt, "clip": ["lora", 1]},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": p.negative or DEFAULT_NEGATIVE, "clip": ["lora", 1]},
        },
    }

    if is_i2i:
        wf["load_image"] = {"class_type": "LoadImage", "inputs": {"image": p.image}}
        wf["encode"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["load_image", 0], "vae": ["checkpoint", 2]},
        }
        latent_ref = ["encode", 0]
        denoise = float(p.strength)
    else:
        wf["empty_latent"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": p.width, "height": p.height, "batch_size": 1},
        }
        latent_ref = ["empty_latent", 0]
        denoise = 1.0

    wf["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["lora", 0],
            "positive": ["positive", 0],
            "negative": ["negative", 0],
            "latent_image": latent_ref,
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "sgm_uniform",
            "denoise": denoise,
            "seed": seed,
        },
    }
    wf["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler", 0], "vae": ["checkpoint", 2]},
    }
    wf["save"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["decode", 0],
            "filename_prefix": "wan_studio_image_sdxl",
        },
    }
    return wf


def build_upscale_workflow(p: "UpscaleParams") -> dict:
    """Upscale an image with Real-ESRGAN-style 4x model. factor=2 scales the 4x output down to 2x."""
    wf = {
        "load_image": {"class_type": "LoadImage", "inputs": {"image": p.image}},
        "upscale_model": {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": UPSCALER_MODEL},
        },
        "upscale": {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["upscale_model", 0],
                "image": ["load_image", 0],
            },
        },
    }
    final_ref = ["upscale", 0]
    if p.factor == 2:
        wf["scale_down"] = {
            "class_type": "ImageScaleBy",
            "inputs": {
                "image": ["upscale", 0],
                "upscale_method": "lanczos",
                "scale_by": 0.5,
            },
        }
        final_ref = ["scale_down", 0]

    wf["save"] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": final_ref,
            "filename_prefix": "wan_studio_image_upscale",
        },
    }
    return wf


# ---------- Models ----------

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

    # Frame interpolation (post-process via RIFE)
    interpolate: bool = False           # if true, run RIFE between decode and save
    interpolate_multiplier: int = 2     # 2 = double fps, 3 = triple, etc.

    # Wan FLF2V (first-last frame): provide explicit ending frame for guaranteed pose continuity
    end_image: str = ""

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


class UpscaleParams(BaseModel):
    user_name: str = ""
    user_emoji: str = ""

    image: str
    factor: int = 4  # 2 or 4


class StorybookParams(BaseModel):
    user_name: str = ""
    user_emoji: str = ""

    story: str
    n_pages: int = 6  # 3..9
    style: str = "pixar"  # pixar | watercolor | anime | cartoon
    aspect: str = "landscape"  # landscape | square | portrait


class ImprovePromptParams(BaseModel):
    prompt: str


class HistoryEntry(BaseModel):
    id: str
    prompt_id: str
    filename: str
    mode: str
    prompt: str
    params: dict
    created_at: float
    duration_s: float | None = None


# ---------- App-wide active-gen state ----------

MONITOR_CLIENT_ID = "wan-studio-monitor"

class GenState:
    def __init__(self, prompt_id: str, gen_id: str, params: dict, started_at: float, kind: str = "video"):
        self.prompt_id = prompt_id
        self.gen_id = gen_id
        self.params = params
        self.started_at = started_at
        self.kind = kind  # "video" | "image" | "upscale" | "storybook"
        self.node: Optional[str] = None
        self.step: int = 0
        self.total_steps: int = 0
        self.preview_images: list[str] = []  # storybook: per-page Flux outputs
        self.scene_descriptions: list[str] = []  # storybook: per-page LLM scene descriptions
        self.character: str = ""  # storybook: LLM-generated character description

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "gen_id": self.gen_id,
            "params": self.params,
            "started_at": self.started_at,
            "kind": self.kind,
            "node": self.node,
            "step": self.step,
            "total_steps": self.total_steps,
            "preview_images": list(self.preview_images),
            "scene_descriptions": list(self.scene_descriptions),
            "character": self.character,
            "elapsed_s": time.time() - self.started_at,
        }


_state_lock: Optional[asyncio.Lock] = None
_active_gen: Optional[GenState] = None
_last_error: Optional[str] = None
_monitor_task: Optional[asyncio.Task] = None


async def _save_completion_to_history(gen: GenState):
    """Fetch result from ComfyUI history and append to our history.json."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{COMFY_HTTP}/history/{gen.prompt_id}")
        h = r.json().get(gen.prompt_id, {})
        outputs = h.get("outputs", {})
        filename = None
        for _, out in outputs.items():
            for key in ("videos", "gifs", "images"):
                for item in (out.get(key) or []):
                    if item.get("filename"):
                        filename = item["filename"]
                        break
                if filename:
                    break
            if filename:
                break
        if not filename:
            return
        items = _load_json(HISTORY_FILE, [])
        items = [it for it in items if it.get("prompt_id") != gen.prompt_id]
        # Mode label for display
        mode = (
            gen.params.get("image_mode")  # create / modify / upscale
            or gen.params.get("mode")     # "storybook"
            or gen.kind
        )
        items.insert(0, {
            "id": gen.gen_id,
            "prompt_id": gen.prompt_id,
            "filename": filename,
            "kind": gen.kind,
            "mode": mode,
            "prompt": gen.params.get("prompt", ""),
            "params": gen.params,
            "created_by_name": gen.params.get("user_name", ""),
            "created_by_emoji": gen.params.get("user_emoji", ""),
            "created_at": gen.started_at,
            "duration_s": time.time() - gen.started_at,
        })
        items = items[:200]
        _save_json(HISTORY_FILE, items)
    except Exception as e:
        print(f"[monitor] history save failed: {e}")


async def _handle_monitor_event(evt: dict):
    global _active_gen, _last_error
    if _active_gen is None:
        return
    ty = evt.get("type")
    data = evt.get("data") or {}
    pid = data.get("prompt_id")
    if pid and pid != _active_gen.prompt_id:
        return

    if ty == "executing":
        node = data.get("node")
        if node is None:
            # pipeline completed for this prompt
            done = _active_gen
            _active_gen = None
            await _save_completion_to_history(done)
        else:
            _active_gen.node = node
    elif ty == "progress":
        v = int(data.get("value", 0) or 0)
        m = int(data.get("max", 1) or 1)
        node = data.get("node") or _active_gen.node
        _active_gen.node = node
        if node and node.lower() == "sampler" and m <= 100:
            _active_gen.step = v
            _active_gen.total_steps = m
    elif ty == "execution_error":
        _last_error = f"{data.get('node_type','?')}: {data.get('exception_message','?')}"
        _active_gen = None


async def _monitor_loop():
    """Persistent WebSocket to ComfyUI. All gens submit with MONITOR_CLIENT_ID,
    so this single socket sees every event for those gens."""
    backoff = 1.0
    while True:
        try:
            url = f"{COMFY_WS}?clientId={MONITOR_CLIENT_ID}"
            async with websockets.connect(url, max_size=2**24, ping_interval=20) as ws:
                backoff = 1.0
                async for msg in ws:
                    if isinstance(msg, (bytes, bytearray)):
                        continue
                    try:
                        evt = json.loads(msg)
                    except Exception:
                        continue
                    await _handle_monitor_event(evt)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[monitor] WS dropped: {e} — reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def _queue_poll_loop():
    """Fallback to the WS monitor: every 5s check ComfyUI's queue + history.
    Catches gens submitted before backend start, gens whose WS events we missed,
    and confirms completion even if the WS dropped."""
    global _active_gen, _last_error
    while True:
        try:
            await asyncio.sleep(5)
            if _active_gen is None:
                continue
            # Storybook is orchestrated entirely by _run_storybook; its prompt_id is a
            # synthetic "storybook-..." string and isn't tracked by ComfyUI's queue.
            # The orchestrator clears _active_gen itself when done.
            if _active_gen.kind == "storybook":
                continue
            pid = _active_gen.prompt_id
            async with httpx.AsyncClient(timeout=5) as c:
                qr = await c.get(f"{COMFY_HTTP}/queue")
            q = qr.json()
            running_pids = {item[1] for item in (q.get("queue_running") or [])}
            pending_pids = {item[1] for item in (q.get("queue_pending") or [])}
            if pid in running_pids or pid in pending_pids:
                continue  # still in flight, nothing to do
            # No longer in queue → check history
            async with httpx.AsyncClient(timeout=5) as c:
                hr = await c.get(f"{COMFY_HTTP}/history/{pid}")
            h = hr.json().get(pid)
            if not h:
                # Vanished without history entry — treat as cancelled
                _active_gen = None
                continue
            status = (h.get("status") or {}).get("status_str", "")
            if status == "error":
                msgs = (h.get("status") or {}).get("messages") or []
                err_msg = "Generation failed"
                for m in msgs:
                    if isinstance(m, list) and m and m[0] == "execution_error":
                        err_msg = (m[1] or {}).get("exception_message") or err_msg
                        break
                _last_error = err_msg
                _active_gen = None
            else:
                # success — save to history, clear active
                done = _active_gen
                _active_gen = None
                await _save_completion_to_history(done)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[poll] error: {e}")


async def _recover_active_gen():
    """At startup, see if ComfyUI is already running a job we should track."""
    global _active_gen
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{COMFY_HTTP}/queue")
        running = r.json().get("queue_running") or []
        if not running:
            return
        # Each item: [number, prompt_id, workflow_dict, extra_data?, ...]
        first = running[0]
        prompt_id = first[1]
        workflow = first[2] if len(first) > 2 else {}
        # Heuristically infer mode / prompt from the workflow
        mode = "i2v" if "i2v_encode" in workflow else "t2v"
        prompt_text = (
            workflow.get("text_encode", {}).get("inputs", {}).get("positive_prompt", "")
        )
        _active_gen = GenState(
            prompt_id=prompt_id,
            gen_id=f"recovered-{uuid.uuid4().hex[:6]}",
            params={"mode": mode, "prompt": prompt_text, "_recovered": True},
            started_at=time.time(),  # we lost the real start; close enough
        )
        print(f"[monitor] recovered active gen prompt_id={prompt_id}")
    except Exception as e:
        print(f"[monitor] state recovery failed: {e}")


_poll_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state_lock, _monitor_task, _poll_task
    THUMB_DIR.mkdir(exist_ok=True)
    _state_lock = asyncio.Lock()
    await _recover_active_gen()
    _monitor_task = asyncio.create_task(_monitor_loop())
    _poll_task = asyncio.create_task(_queue_poll_loop())
    yield
    for t in (_monitor_task, _poll_task):
        if t:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Helpers ----------

def _now() -> float:
    return time.time()


def _generate_thumb(video_path: Path, thumb_path: Path) -> bool:
    if thumb_path.exists():
        return True
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vframes", "1", "-q:v", "5",
             "-vf", "scale=320:-1", str(thumb_path)],
            check=True, capture_output=True, timeout=20,
        )
        return True
    except Exception:
        return False


# ---------- Routes ----------

@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            r = await c.get(f"{COMFY_HTTP}/system_stats")
        return {"ok": True, "comfy": r.json().get("system", {}).get("comfyui_version")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    COMFY_INPUT.mkdir(exist_ok=True)
    ext = Path(file.filename or "").suffix or ".png"
    name = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    dest = COMFY_INPUT / name
    with dest.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)
    return {"filename": name}


@app.post("/api/image_generate")
async def image_generate(params: ImageGenerateParams):
    """Text-to-image OR image-to-image (modify) using Flux schnell or SDXL Lightning."""
    global _active_gen, _last_error
    if _state_lock is None:
        raise HTTPException(503, "backend not ready")
    # Free the LLM from VRAM before queueing diffusion work
    await llm.unload()
    async with _state_lock:
        if _active_gen is not None:
            raise HTTPException(409, "Someone is already making something. Wait a moment.")
        if params.image_mode == "modify" and not params.image:
            raise HTTPException(400, "Modify mode needs an uploaded image")

        if params.model == "sdxl":
            workflow = build_sdxl_image_workflow(params)
        else:
            workflow = build_flux_image_workflow(params)

        gen_id = uuid.uuid4().hex[:8]
        client_id = MONITOR_CLIENT_ID
        async with httpx.AsyncClient(timeout=30) as c:
            try:
                r = await c.post(
                    f"{COMFY_HTTP}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                )
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise HTTPException(502, f"Engine rejected: {e.response.text[:300]}")
            data = r.json()

        _active_gen = GenState(
            prompt_id=data["prompt_id"],
            gen_id=gen_id,
            params=params.model_dump(),
            started_at=time.time(),
            kind="image",
        )
        _last_error = None

    return {
        "prompt_id": data["prompt_id"],
        "client_id": client_id,
        "gen_id": gen_id,
        "queue_number": data.get("number"),
    }


@app.post("/api/image_upscale")
async def image_upscale(params: UpscaleParams):
    """Upscale an image 2× or 4× using a Real-ESRGAN-style model."""
    global _active_gen, _last_error
    if _state_lock is None:
        raise HTTPException(503, "backend not ready")
    if params.factor not in (2, 4):
        raise HTTPException(400, "factor must be 2 or 4")
    await llm.unload()
    async with _state_lock:
        if _active_gen is not None:
            raise HTTPException(409, "Someone is already making something. Wait a moment.")

        workflow = build_upscale_workflow(params)
        gen_id = uuid.uuid4().hex[:8]
        client_id = MONITOR_CLIENT_ID
        async with httpx.AsyncClient(timeout=30) as c:
            try:
                r = await c.post(
                    f"{COMFY_HTTP}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                )
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise HTTPException(502, f"Engine rejected: {e.response.text[:300]}")
            data = r.json()

        params_dict = params.model_dump()
        params_dict["image_mode"] = "upscale"
        _active_gen = GenState(
            prompt_id=data["prompt_id"],
            gen_id=gen_id,
            params=params_dict,
            started_at=time.time(),
            kind="upscale",
        )
        _last_error = None

    return {
        "prompt_id": data["prompt_id"],
        "client_id": client_id,
        "gen_id": gen_id,
        "queue_number": data.get("number"),
    }


@app.get("/api/image/{filename}")
async def get_image(filename: str):
    """Serve an image file from the output dir."""
    path = COMFY_OUTPUT / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "not found")
    ext = path.suffix.lower()
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=media)


STYLE_PREFIXES = {
    "pixar":      "Pixar 3D animation style, soft warm lighting, high detail, cinematic",
    "watercolor": "watercolor children's book illustration, soft pastels, hand-painted texture",
    "anime":      "anime children's book illustration, soft pastel colors, cel-shaded, Studio Ghibli inspired",
    "cartoon":    "friendly cartoon illustration, bold outlines, bright cheerful colors",
}
STORY_ASPECT_DIMS = {
    "landscape": (832, 480),
    "square":    (640, 640),
    "portrait":  (480, 832),
}


async def _submit_comfy_and_wait(workflow: dict, timeout_s: float = 600.0) -> str:
    """Submit a workflow to ComfyUI, poll history until done, return primary output filename."""
    client_id = uuid.uuid4().hex
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{COMFY_HTTP}/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        r.raise_for_status()
        prompt_id = r.json()["prompt_id"]

    t0 = time.time()
    while True:
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"sub-gen timed out after {timeout_s}s")
        await asyncio.sleep(2)
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{COMFY_HTTP}/history/{prompt_id}")
        data = r.json()
        if prompt_id not in data:
            continue
        h = data[prompt_id]
        status = (h.get("status") or {}).get("status_str", "")
        if status == "error":
            msgs = (h.get("status") or {}).get("messages") or []
            err = "unknown error"
            for m in msgs:
                if isinstance(m, list) and m and m[0] == "execution_error":
                    err = (m[1] or {}).get("exception_message") or err
                    break
            raise RuntimeError(f"sub-gen failed: {err}")
        outputs = h.get("outputs", {})
        for _, out in outputs.items():
            for key in ("videos", "gifs", "images"):
                for item in (out.get(key) or []):
                    if item.get("filename"):
                        return item["filename"]


async def _ffmpeg_extract_frame(video_path: str, output_png: Path, last: bool):
    """Extract first or last frame of a video as a PNG."""
    if last:
        # Seek to 0.5s before end, then take 1 frame (covers any short clip)
        args = ["ffmpeg", "-y", "-sseof", "-0.5", "-i", video_path,
                "-update", "1", "-frames:v", "1", str(output_png)]
    else:
        args = ["ffmpeg", "-y", "-i", video_path,
                "-frames:v", "1", str(output_png)]
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extract failed: {stderr.decode()[:300]}")


def _build_transition_workflow(img_a_name: str, img_b_name: str, prefix: str, multiplier: int = 16) -> dict:
    """RIFE-morph from img_a → img_b over `multiplier` intermediate frames."""
    return {
        "load_a": {"class_type": "LoadImage", "inputs": {"image": img_a_name}},
        "load_b": {"class_type": "LoadImage", "inputs": {"image": img_b_name}},
        "batch": {
            "class_type": "ImageBatch",
            "inputs": {"image1": ["load_a", 0], "image2": ["load_b", 0]},
        },
        "rife": {
            "class_type": "RIFE VFI",
            "inputs": {
                "ckpt_name": "rife49.pth",
                "frames": ["batch", 0],
                "clear_cache_after_n_frames": 10,
                "multiplier": int(multiplier),
                "fast_mode": True,
                "ensemble": False,
                "scale_factor": 1.0,
                "dtype": "float16",
                "torch_compile": False,
                "batch_size": 4,
            },
        },
        "save": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["rife", 0],
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": prefix,
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    }


async def _ffprobe_duration(path: str) -> float:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    try:
        return float(out.decode().strip())
    except Exception:
        return 0.0


async def _mux_scene_with_audio(video_path: str, audio_path: Optional[str], out_path: str, min_hold: float = 1.5):
    """Combine a Wan clip with a narration WAV. Pad video (clone last frame) and audio (silence)
    so they end together. Total duration = max(video+min_hold, audio+0.5s breathing room)."""
    video_dur = await _ffprobe_duration(video_path)
    audio_dur = (await _ffprobe_duration(audio_path)) if audio_path else 0.0
    total = max(video_dur + min_hold, audio_dur + 0.5, video_dur + 0.1)
    v_pad = max(0.0, total - video_dur)
    a_pad = max(0.0, total - audio_dur)

    if audio_path and audio_dur > 0:
        filter_str = (
            f"[0:v]tpad=stop_mode=clone:stop_duration={v_pad:.3f}[v];"
            f"[1:a]apad=pad_dur={a_pad:.3f}[a]"
        )
        args = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex", filter_str,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            out_path,
        ]
    else:
        args = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"tpad=stop_mode=clone:stop_duration={v_pad:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            out_path,
        ]
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"mux scene failed: {stderr.decode()[:400]}")


async def _has_audio(video_path: str) -> bool:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    return b"audio" in (out or b"")


async def _stitch_videos(paths: list[str], output_path: str, hold_dur: float = 0.0, **_ignored):
    """Plain hard-cut concat. Inputs may carry audio (will be preserved).
    If hold_dur > 0, last frame of each clip is also tpad-extended.
    For the storybook with audio, hold is baked into each scene segment so hold_dur=0 here."""
    if not paths:
        raise ValueError("no paths to stitch")

    has_audio = await _has_audio(paths[0])

    if len(paths) == 1 and hold_dur <= 0 and has_audio:
        # Just transcode-copy (or remux). Keep audio.
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", paths[0],
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", output_path,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode()[:300]}")
        return

    # Build filter_complex
    args: list[str] = ["ffmpeg", "-y"]
    for p in paths:
        args.extend(["-i", p])
    parts: list[str] = []
    v_labels: list[str] = []
    a_labels: list[str] = []
    for i in range(len(paths)):
        v = f"v{i}"
        v_labels.append(f"[{v}]")
        if hold_dur > 0:
            parts.append(f"[{i}:v]tpad=stop_mode=clone:stop_duration={hold_dur:.3f}[{v}]")
        else:
            parts.append(f"[{i}:v]copy[{v}]")
        if has_audio:
            a_labels.append(f"[{i}:a]")
    if has_audio:
        chains = "".join(f"{v_labels[i]}{a_labels[i]}" for i in range(len(paths)))
        parts.append(f"{chains}concat=n={len(paths)}:v=1:a=1[outv][outa]")
        maps = ["-map", "[outv]", "-map", "[outa]"]
    else:
        parts.append(f"{''.join(v_labels)}concat=n={len(paths)}:v=1:a=0[outv]")
        maps = ["-map", "[outv]"]

    args.extend([
        "-filter_complex", ";".join(parts),
        *maps,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
    ])
    if has_audio:
        args.extend(["-c:a", "aac", "-b:a", "128k"])
    args.extend(["-movflags", "+faststart", output_path])

    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg stitch failed: {stderr.decode()[:500]}")


async def _run_storybook(p: StorybookParams, prompt_id: str, gen_id: str):
    """Background orchestration: plan → image per page → video per page → stitch."""
    global _active_gen, _last_error
    try:
        if _active_gen is not None:
            _active_gen.node = "planning"

        # 1) Use the LLM to plan
        plan = await llm.plan_storybook(p.story, p.n_pages, p.style)
        character = plan.get("character", "")
        scenes = plan.get("scenes") or []
        if not scenes:
            raise RuntimeError("LLM returned no scenes")

        # Surface plan in state for the frontend to show
        if _active_gen is not None:
            _active_gen.character = character
            _active_gen.scene_descriptions = [
                s.get("description") or "" for s in scenes
            ]

        # Free the LLM from VRAM before starting the long stream of Flux + Wan gens
        await llm.unload()

        style_prefix = STYLE_PREFIXES.get(p.style.lower(), STYLE_PREFIXES["pixar"])
        width, height = STORY_ASPECT_DIMS.get(p.aspect, STORY_ASPECT_DIMS["landscape"])
        seed = int(time.time()) % (2**31)

        page_videos: list[str] = []
        prev_image: Optional[str] = None
        total = len(scenes) * 2  # image + video per page
        step_done = 0

        for i, scene in enumerate(scenes):
            scene_desc = scene.get("description") or "A scene from the story."
            motion = scene.get("motion") or "gentle subtle motion"
            video_prompt = scene.get("video_prompt") or motion
            starting_pose = scene.get("starting_pose") or ""
            ending_pose = scene.get("ending_pose") or ""

            # --- Generate page image ---
            if _active_gen is not None:
                _active_gen.node = f"page-{i+1}-image"
                _active_gen.step = step_done
                _active_gen.total_steps = total

            kontext_available = (Path("/media/yunus/More Data/comfyui-models/diffusion_models") / FLUX_KONTEXT_MODEL).exists()
            ref_name = f"storybook_charref_{gen_id}.png"
            start_image_input_name = f"storybook_start_p{i}_{gen_id}.png"
            end_image_input_name = f"storybook_end_p{i}_{gen_id}.png"

            def _flux_workflow_for(prompt_text: str, this_seed: int, ref_required: bool):
                if ref_required and kontext_available and (COMFY_INPUT / ref_name).exists():
                    return build_flux_kontext_workflow(
                        prompt=prompt_text, width=width, height=height,
                        seed=this_seed, reference_image=ref_name, steps=20,
                    )
                return build_flux_image_workflow(ImageGenerateParams(
                    image_mode="create", prompt=prompt_text,
                    width=width, height=height, seed=this_seed, model="flux",
                ))

            # ---- START image ----
            # The strip thumbnail for this page IS this page's start frame:
            #  - Page 1: freshly generated start
            #  - Page N≥2: byte-perfect copy of page N-1's end (so the strip reads as a
            #    chained keyframe storyboard, exactly N thumbnails for N pages).
            if i == 0:
                start_pose_text = starting_pose or "in an initial settled pose"
                start_prompt = f"{style_prefix}. {character} {scene_desc}. {character} is {start_pose_text}."
                wf = _flux_workflow_for(start_prompt, seed, ref_required=False)
                start_out = await _submit_comfy_and_wait(wf, timeout_s=300)
                # Copy as Wan start input + as the canonical character reference for Kontext
                shutil.copyfile(str(COMFY_OUTPUT / start_out), str(COMFY_INPUT / start_image_input_name))
                shutil.copyfile(str(COMFY_OUTPUT / start_out), str(COMFY_INPUT / ref_name))
                page_thumb = start_out
            else:
                # Continuity: page N+1's start frame == page N's end frame (byte-perfect)
                shutil.copyfile(str(COMFY_OUTPUT / prev_end_image_filename), str(COMFY_INPUT / start_image_input_name))
                page_thumb = prev_end_image_filename
            if _active_gen is not None:
                _active_gen.preview_images.append(page_thumb)

            # ---- END image (Wan FLF2V target — where motion lands) ----
            end_pose_text = ending_pose or starting_pose or "in a settled, restful pose"
            end_prompt = (
                f"{style_prefix}. Same character: {character}. "
                f"{scene_desc}. {character} is now {end_pose_text}."
            )
            wf_end = _flux_workflow_for(end_prompt, seed + i * 100 + 7, ref_required=True)
            end_out = await _submit_comfy_and_wait(wf_end, timeout_s=300)
            shutil.copyfile(str(COMFY_OUTPUT / end_out), str(COMFY_INPUT / end_image_input_name))
            prev_end_image_filename = end_out
            prev_image = end_out
            step_done += 1

            # ---- Animate (Wan 2.2 FLF2V: start + end frame conditioning) ----
            if _active_gen is not None:
                _active_gen.node = f"page-{i+1}-animate"
                _active_gen.step = step_done

            pose_chain = ""
            if starting_pose and ending_pose:
                pose_chain = (
                    f" The scene starts with {character} {starting_pose}, and ends with "
                    f"{character} {ending_pose}."
                )
            elif ending_pose:
                pose_chain = f" The scene ends with {character} {ending_pose}."
            wan_full_prompt = (
                f"{video_prompt}{pose_chain} {character} "
                f"Storybook illustration style, {p.style} aesthetic, soft cinematic lighting, smooth gentle motion."
            )
            i2v_params = GenerateParams(
                prompt=wan_full_prompt,
                negative=DEFAULT_NEGATIVE,
                width=width, height=height,
                frames=49, steps=20,
                cfg=6.0, shift=5.0, seed=seed,
                fps=16, scheduler="unipc",
                noise_aug=0.0,
                image=start_image_input_name,
                end_image=end_image_input_name,  # FLF2V — Wan lands on this exact frame
                multi_gpu=True,
                attention_mode="sdpa",
                block_swap_count=15, block_swap_device="cuda:1",
                vae_tiling=False,
                keep_t5_loaded=True,
                interpolate=False,
                use_slg=True, use_feta=True, use_teacache=True,
            )
            # Wan 2.2 14B MoE — much better face fidelity than 2.1 for storybook
            wf_v = build_wan22_i2v_workflow(i2v_params)
            vid_filename = await _submit_comfy_and_wait(wf_v, timeout_s=1500)
            page_videos.append(str(COMFY_OUTPUT / vid_filename))
            step_done += 1

        # --- Generate narration audio per scene + mux into per-scene segments ---
        scene_segments: list[str] = []
        if _active_gen is not None:
            _active_gen.node = "narrating"
        narration_dir = COMFY_OUTPUT / f"_storybook_narr_{gen_id}"
        narration_dir.mkdir(exist_ok=True)

        for i, scene in enumerate(scenes):
            if _active_gen is not None:
                _active_gen.node = f"narration-{i + 1}"
            narration_text = (scene.get("narration") or "").strip()
            audio_path = narration_dir / f"page_{i}.wav"
            try:
                if narration_text:
                    await tts.synthesize_to_file(narration_text, audio_path)
                else:
                    audio_path = None  # type: ignore
            except Exception as e:
                print(f"[storybook] TTS failed for page {i+1}: {e}")
                audio_path = None  # type: ignore

            # Mux the page's Wan clip with its narration audio + a tail-hold
            seg_path = narration_dir / f"scene_{i}.mp4"
            try:
                await _mux_scene_with_audio(
                    page_videos[i],
                    str(audio_path) if audio_path else None,
                    str(seg_path),
                    min_hold=1.0,
                )
                scene_segments.append(str(seg_path))
            except Exception as e:
                print(f"[storybook] mux page {i+1} failed: {e}")
                # fall back: use the raw Wan clip (no audio, no hold)
                scene_segments.append(page_videos[i])

        # --- Final stitch (audio carries through) ---
        if _active_gen is not None:
            _active_gen.node = "stitching"
            _active_gen.step = total
            _active_gen.total_steps = total

        final_filename = f"wan_studio_storybook_{gen_id}.mp4"
        final_path = COMFY_OUTPUT / final_filename
        await _stitch_videos(scene_segments, str(final_path), hold_dur=0.0)

        # Save to history
        items = _load_json(HISTORY_FILE, [])
        items = [it for it in items if it.get("prompt_id") != prompt_id]
        items.insert(0, {
            "id": gen_id,
            "prompt_id": prompt_id,
            "filename": final_filename,
            "kind": "storybook",
            "mode": "storybook",
            "prompt": p.story,
            "params": {
                **p.model_dump(),
                "plan": plan,
                "page_videos": [os.path.basename(v) for v in page_videos],
            },
            "created_by_name": p.user_name,
            "created_by_emoji": p.user_emoji,
            "created_at": _active_gen.started_at if _active_gen else time.time(),
            "duration_s": time.time() - (_active_gen.started_at if _active_gen else time.time()),
        })
        _save_json(HISTORY_FILE, items[:200])
    except Exception as e:
        print(f"[storybook] {e}")
        _last_error = f"storybook failed: {e}"
    finally:
        _active_gen = None


@app.post("/api/storybook")
async def storybook(p: StorybookParams):
    global _active_gen, _last_error
    if _state_lock is None:
        raise HTTPException(503, "backend not ready")
    if not p.story.strip():
        raise HTTPException(400, "story is required")
    if p.n_pages < 2 or p.n_pages > 9:
        raise HTTPException(400, "n_pages must be 2..9")
    if not await llm.is_available():
        raise HTTPException(503, "Local LLM (Ollama llama3.2:3b) is not available. Start ollama and pull the model.")
    async with _state_lock:
        if _active_gen is not None:
            raise HTTPException(409, "Someone is already making something.")
        gen_id = uuid.uuid4().hex[:8]
        prompt_id = f"storybook-{gen_id}"
        _active_gen = GenState(
            prompt_id=prompt_id,
            gen_id=gen_id,
            params=p.model_dump(),
            started_at=time.time(),
            kind="storybook",
        )
        _last_error = None

    asyncio.create_task(_run_storybook(p, prompt_id, gen_id))
    return {"prompt_id": prompt_id, "gen_id": gen_id, "kind": "storybook"}


@app.post("/api/llm/improve")
async def llm_improve(p: ImprovePromptParams):
    if not await llm.is_available():
        raise HTTPException(503, "Local LLM not available")
    try:
        out = await llm.improve_prompt(p.prompt)
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {e}")
    return {"prompt": out}


@app.get("/api/state")
async def get_state():
    """Current backend state — what gen is active, last error, etc."""
    return {
        "active": _active_gen.to_dict() if _active_gen else None,
        "last_error": _last_error,
        "monitor_client_id": MONITOR_CLIENT_ID,
    }


@app.post("/api/interrupt")
async def interrupt():
    """Stop the running gen AND free model VRAM so the next request can start clean."""
    global _active_gen
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            await c.post(f"{COMFY_HTTP}/interrupt")
        except Exception as e:
            print(f"[interrupt] failed to call /interrupt: {e}")
        try:
            await c.post(
                f"{COMFY_HTTP}/free",
                json={"unload_models": True, "free_memory": True},
            )
        except Exception as e:
            print(f"[interrupt] failed to call /free: {e}")
        # Also clear ComfyUI's queue so any pending jobs don't auto-start
        try:
            await c.post(f"{COMFY_HTTP}/queue", json={"clear": True})
        except Exception:
            pass
    _active_gen = None
    return {"ok": True}


@app.websocket("/api/ws/{client_id}")
async def ws_relay(socket: WebSocket, client_id: str):
    """Forward ComfyUI WS events for one prompt_id back to the browser."""
    await socket.accept()
    upstream_url = f"{COMFY_WS}?clientId={client_id}"
    try:
        async with websockets.connect(upstream_url, max_size=2**24) as upstream:
            while True:
                try:
                    msg = await asyncio.wait_for(upstream.recv(), timeout=600)
                except asyncio.TimeoutError:
                    await socket.send_json({"type": "timeout"})
                    break
                if isinstance(msg, (bytes, bytearray)):
                    continue
                try:
                    await socket.send_text(msg)
                except WebSocketDisconnect:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await socket.send_json({"type": "ws_error", "error": str(e)})
        except Exception:
            pass


@app.get("/api/result/{prompt_id}")
async def result(prompt_id: str):
    """Once execution is finished, this returns the output filename + saves it to history."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{COMFY_HTTP}/history/{prompt_id}")
    h = r.json().get(prompt_id, {})
    if not h:
        raise HTTPException(404, "prompt not in history yet")

    outputs = h.get("outputs", {})
    filename: Optional[str] = None
    for _, out in outputs.items():
        for item in (out.get("videos") or out.get("gifs") or []):
            if item.get("filename"):
                filename = item["filename"]
                break
        if filename:
            break

    if not filename:
        raise HTTPException(404, "no video output yet")

    prompt_data = (h.get("prompt") or [None, None, {}])[2] or {}
    return {"filename": filename, "prompt_data": prompt_data}


@app.post("/api/history")
async def add_history(entry: HistoryEntry):
    items = _load_json(HISTORY_FILE, [])
    items.insert(0, entry.model_dump())
    items = items[:200]
    _save_json(HISTORY_FILE, items)
    return {"ok": True}


@app.get("/api/history")
async def list_history():
    items = _load_json(HISTORY_FILE, [])
    return {"items": [it for it in items if (COMFY_OUTPUT / it["filename"]).exists()]}


@app.delete("/api/history/{entry_id}")
async def delete_history(entry_id: str, hard: bool = False):
    items = _load_json(HISTORY_FILE, [])
    new_items = []
    deleted = None
    for it in items:
        if it.get("id") == entry_id:
            deleted = it
            continue
        new_items.append(it)
    _save_json(HISTORY_FILE, new_items)
    if deleted and hard:
        p = COMFY_OUTPUT / deleted["filename"]
        if p.exists():
            try: p.unlink()
            except Exception: pass
        tp = THUMB_DIR / (deleted["filename"] + ".jpg")
        if tp.exists():
            try: tp.unlink()
            except Exception: pass
    return {"ok": True, "deleted": bool(deleted)}


@app.get("/api/video/{filename}")
async def get_video(filename: str):
    path = COMFY_OUTPUT / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/thumb/{filename}")
async def get_thumb(filename: str):
    video_path = COMFY_OUTPUT / filename
    if not video_path.exists():
        raise HTTPException(404, "video not found")
    thumb_path = THUMB_DIR / (filename + ".jpg")
    if not thumb_path.exists():
        if not _generate_thumb(video_path, thumb_path):
            raise HTTPException(500, "thumbnail extraction failed")
    return FileResponse(thumb_path, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
