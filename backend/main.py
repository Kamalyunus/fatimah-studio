"""Fatimah Studio backend — FastAPI relay over ComfyUI.

Exposes a clean REST + WebSocket API for the React frontend:
  POST /api/storybook          plan + illustrate + animate a storybook
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

import drift
import llm

# ---------- Config ----------

COMFY_HTTP = "http://127.0.0.1:8188"
COMFY_WS = "ws://127.0.0.1:8188/ws"
COMFY_ROOT = Path("/home/yunus/Documents/comfyui")
COMFY_OUTPUT = COMFY_ROOT / "output"
COMFY_INPUT = COMFY_ROOT / "input"

STUDIO_ROOT = Path(__file__).resolve().parent
THUMB_DIR = STUDIO_ROOT / "thumbs"
HISTORY_FILE = STUDIO_ROOT / "history.json"
# Persistent character library: saved character canons + their reference images,
# so users can re-use the same protagonist across multiple storybooks.
CHARACTER_LIBRARY_FILE = STUDIO_ROOT / "characters.json"
CHARACTER_LIBRARY_DIR = STUDIO_ROOT / "character_refs"

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
                # 0.20 = Wan's recommended sweet spot. 0.25 was tried but visibly hurt
                # fine texture quality.
                "rel_l1_thresh": 0.20,
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
            # 3.5 (was 2.5) — stronger adherence to the text prompt so Kontext follows
            # the requested pose/scene change instead of just reproducing the reference.
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["ref_latent", 0], "guidance": 3.5},
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


def _append_upscale(wf: dict, image_ref: list, save_prefix: str, factor: int = 2) -> dict:
    """Chain a 4x-UltraSharp upscale onto an existing image workflow. Drops the upstream
    save node (if any) and writes a fresh one pointing at the upscaled output.

    factor=4 keeps the raw 4x output; factor=2 lanczos-downscales it (4x dims → 2x dims).
    """
    wf.pop("save", None)
    wf["upscale_model"] = {
        "class_type": "UpscaleModelLoader",
        "inputs": {"model_name": UPSCALER_MODEL},
    }
    wf["upscale"] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {"upscale_model": ["upscale_model", 0], "image": image_ref},
    }
    final_ref = ["upscale", 0]
    if factor == 2:
        wf["scale_down"] = {
            "class_type": "ImageScaleBy",
            "inputs": {"image": ["upscale", 0], "upscale_method": "lanczos", "scale_by": 0.5},
        }
        final_ref = ["scale_down", 0]
    wf["save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": final_ref, "filename_prefix": save_prefix},
    }
    return wf


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
    if p.auto_upscale:
        return _append_upscale(wf, ["decode", 0], "wan_studio_image_flux", factor=2)
    wf["save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["decode", 0], "filename_prefix": "wan_studio_image_flux"},
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
    if p.auto_upscale:
        return _append_upscale(wf, ["decode", 0], "wan_studio_image_sdxl", factor=2)
    wf["save"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["decode", 0], "filename_prefix": "wan_studio_image_sdxl"},
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
    n_pages: int = 9  # 2..12
    style: str = "pixar"  # pixar | watercolor | anime | cartoon
    aspect: str = "landscape"  # landscape | square | portrait
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

        # ----- Storybook keyframe preview gate (#2) -----
        # When the orchestrator finishes generating all Flux start+end pairs, it sets
        # node="awaiting-approval" and waits on `approval_event`. The frontend reads
        # `keyframes` to show the strip, then POSTs /api/storybook/approve or /cancel.
        self.approval_event: asyncio.Event = asyncio.Event()
        self.approval_cancelled: bool = False
        # Per-scene context, populated during the Flux phase. Each entry:
        # {scene_index, start_image, end_image, description, motion_intensity, seed,
        #  start_prompt, end_prompt}
        # Mutable so the regenerate endpoint can update individual scenes in place.
        self.keyframes: list[dict] = []

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
            # Lightweight view of the keyframe context for the frontend approval UI —
            # only the filenames + per-scene metadata it actually needs to render.
            "keyframes": [
                {
                    "scene_index": k.get("scene_index"),
                    "start_image": k.get("start_image"),
                    "end_image": k.get("end_image"),
                    "description": k.get("description"),
                    "motion_intensity": k.get("motion_intensity"),
                    "drift": k.get("drift"),
                    "drift_flagged": k.get("drift_flagged"),
                }
                for k in self.keyframes
            ],
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

async def _comfy_free() -> None:
    """Tell ComfyUI to release cached models and free VRAM. Symmetric with llm.unload():
    diffusion endpoints unload the LLM before queueing; LLM endpoints should ask ComfyUI
    to free its cached models before invoking Ollama, so the (~23 GB) LLM can actually load."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(
                f"{COMFY_HTTP}/free",
                json={"unload_models": True, "free_memory": True},
            )
    except Exception:
        pass  # don't fail the caller if ComfyUI is briefly unreachable


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


class UseAsInputParams(BaseModel):
    filename: str   # filename in COMFY_OUTPUT to copy into COMFY_INPUT for re-use


@app.post("/api/use_as_input")
async def use_as_input(p: UseAsInputParams):
    """Copy a previously generated output image into ComfyUI's input/ folder so it can
    be referenced by a follow-up modify/i2i workflow without a full upload round-trip."""
    src = COMFY_OUTPUT / p.filename
    if not src.exists() or not src.is_file():
        raise HTTPException(404, "source file not found in output")
    COMFY_INPUT.mkdir(exist_ok=True)
    ext = src.suffix or ".png"
    name = f"iterate_{uuid.uuid4().hex[:8]}{ext}"
    shutil.copyfile(str(src), str(COMFY_INPUT / name))
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

        # User-facing image gen always auto-upscales 2x; storybook page gen doesn't (the
        # storybook orchestrator builds ImageGenerateParams directly without this flag).
        params.auto_upscale = True

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
    # Bumped from 832×480 / 640×640 / 480×832 to ~30% more pixels per axis for
    # noticeably sharper output. Wan time scales roughly with pixel count.
    "landscape": (1024, 576),
    "square":    (768, 768),
    "portrait":  (576, 1024),
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


async def _stitch_videos(paths: list[str], output_path: str, hold_dur: float = 0.0):
    """Hard-cut concat of silent video clips. If hold_dur > 0, the last frame of each
    clip is tpad-extended before the cut (page-turn beat). Storybook uses hold_dur=0
    because byte-perfect FLF2V chaining already makes the cuts invisible."""
    if not paths:
        raise ValueError("no paths to stitch")

    args: list[str] = ["ffmpeg", "-y"]
    for p in paths:
        args.extend(["-i", p])
    parts: list[str] = []
    v_labels: list[str] = []
    for i in range(len(paths)):
        v = f"v{i}"
        v_labels.append(f"[{v}]")
        if hold_dur > 0:
            parts.append(f"[{i}:v]tpad=stop_mode=clone:stop_duration={hold_dur:.3f}[{v}]")
        else:
            parts.append(f"[{i}:v]copy[{v}]")
    parts.append(f"{''.join(v_labels)}concat=n={len(paths)}:v=1:a=0[outv]")

    args.extend([
        "-filter_complex", ";".join(parts),
        "-map", "[outv]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        "-movflags", "+faststart", output_path,
    ])

    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg stitch failed: {stderr.decode()[:500]}")


def _build_wan_prompt(video_prompt: str, starting_pose: str, ending_pose: str,
                      character: str, style: str) -> str:
    """Assemble the full Wan prompt from the LLM's per-scene direction. Kept as a small
    helper so the per-scene regen path can call it without duplicating the template."""
    pose_chain = ""
    if starting_pose and ending_pose:
        pose_chain = (
            f" The scene starts with {character} {starting_pose}, and ends with "
            f"{character} {ending_pose}."
        )
    elif ending_pose:
        pose_chain = f" The scene ends with {character} {ending_pose}."
    return (
        f"{video_prompt}{pose_chain} {character} "
        f"Storybook illustration style, {style} aesthetic, soft cinematic lighting, smooth gentle motion."
    )


async def _run_storybook(p: StorybookParams, prompt_id: str, gen_id: str):
    """Background orchestration: plan → image per page → video per page → stitch."""
    global _active_gen, _last_error
    try:
        if _active_gen is not None:
            _active_gen.node = "planning"

        # Free any ComfyUI-cached models (Flux from a recent image gen, etc.) so the
        # planner LLM can load without contending for GPU memory.
        await _comfy_free()

        # If the user picked a saved character, load its canon + reference image and
        # tell the planner to keep that character's canonical description verbatim.
        saved_character: Optional[dict] = None
        if p.character_id:
            library = _load_json(CHARACTER_LIBRARY_FILE, [])
            saved_character = next((c for c in library if c.get("id") == p.character_id), None)
            if saved_character:
                ref_src = CHARACTER_LIBRARY_DIR / (saved_character.get("ref_filename") or "")
                if not ref_src.exists():
                    saved_character = None  # silent fallback to fresh generation

        # 1) Use the LLM to plan (passing the saved character's canon, if any, so the
        # LLM treats it as locked rather than inventing a new protagonist).
        plan = await llm.plan_storybook(
            p.story, p.n_pages, p.style,
            existing_canon=saved_character.get("canon") if saved_character else None,
            existing_character=saved_character.get("character") if saved_character else None,
        )
        character = plan.get("character", "")
        # Render the structured character canon once and reuse it verbatim in every
        # Flux prompt — Kontext alone can drift on outfit/feature details. Fallback
        # to the prose `character` field if canon is missing.
        canon_clause = llm.render_canon(plan.get("character_canon")) or character
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

        # noise_aug per motion intensity: stiller scenes get less Wan freedom (cleaner),
        # dynamic scenes get more (Wan needs room to invent in-betweens).
        NOISE_AUG_BY_INTENSITY = {"still": 0.0, "gentle": 0.05, "dynamic": 0.10}

        kontext_available = (Path("/media/yunus/More Data/comfyui-models/diffusion_models") / FLUX_KONTEXT_MODEL).exists()
        ref_name = f"storybook_charref_{gen_id}.png"

        # Total steps: image + animate per scene; the keyframe preview gate sits between.
        total = len(scenes) * 2
        step_done = 0

        # =================== PHASE A — Generate all Flux start+end pairs ===================
        # We do this in one pass so the user can preview every keyframe before committing
        # to the heavy Wan phase.
        prev_end_image_filename: Optional[str] = None
        for i, scene in enumerate(scenes):
            scene_desc = scene.get("description") or "A scene from the story."
            starting_pose = scene.get("starting_pose") or ""
            ending_pose = scene.get("ending_pose") or ""
            intensity = (scene.get("motion_intensity") or "gentle").lower()

            if _active_gen is not None:
                _active_gen.node = f"page-{i+1}-image"
                _active_gen.step = step_done
                _active_gen.total_steps = total

            start_image_input_name = f"storybook_start_p{i}_{gen_id}.png"
            end_image_input_name = f"storybook_end_p{i}_{gen_id}.png"

            # ---- START image ----
            # Page 1: fresh Flux gen (or saved-character reference). Pages 2+: byte-perfect
            # copy of the previous page's end image (so cuts are invisible at stitch time).
            start_prompt = ""
            if i == 0:
                start_pose_text = starting_pose or "in an initial settled pose"
                if saved_character:
                    saved_ref = CHARACTER_LIBRARY_DIR / saved_character["ref_filename"]
                    shutil.copyfile(str(saved_ref), str(COMFY_INPUT / start_image_input_name))
                    shutil.copyfile(str(saved_ref), str(COMFY_INPUT / ref_name))
                    page1_seed_name = f"saved_char_p0_{gen_id}.png"
                    shutil.copyfile(str(saved_ref), str(COMFY_OUTPUT / page1_seed_name))
                    page_thumb = page1_seed_name
                else:
                    start_prompt = (
                        f"{style_prefix}. {canon_clause}. {scene_desc}. "
                        f"{character} is {start_pose_text}."
                    )
                    wf = build_flux_image_workflow(ImageGenerateParams(
                        image_mode="create", prompt=start_prompt,
                        width=width, height=height, seed=seed, model="flux",
                    ))
                    start_out = await _submit_comfy_and_wait(wf, timeout_s=300)
                    shutil.copyfile(str(COMFY_OUTPUT / start_out), str(COMFY_INPUT / start_image_input_name))
                    shutil.copyfile(str(COMFY_OUTPUT / start_out), str(COMFY_INPUT / ref_name))
                    page_thumb = start_out
            else:
                shutil.copyfile(str(COMFY_OUTPUT / prev_end_image_filename), str(COMFY_INPUT / start_image_input_name))
                page_thumb = prev_end_image_filename
            if _active_gen is not None:
                _active_gen.preview_images.append(page_thumb)

            # ---- END image (Wan FLF2V target) ----
            end_pose_text = ending_pose or starting_pose or "in a settled, restful pose"
            end_prompt = (
                f"{character} {end_pose_text}. "
                f"{scene_desc}. {style_prefix}. "
                f"Character canon (must match exactly): {canon_clause}. "
                f"The character's appearance matches the reference exactly, "
                f"but the pose, body position, and gesture are clearly different from the reference."
            )
            end_seed = seed + i * 100 + 7
            wf_end = (
                build_flux_kontext_workflow(
                    prompt=end_prompt, width=width, height=height,
                    seed=end_seed, reference_image=ref_name, steps=20,
                )
                if kontext_available and (COMFY_INPUT / ref_name).exists()
                else build_flux_image_workflow(ImageGenerateParams(
                    image_mode="create", prompt=end_prompt,
                    width=width, height=height, seed=end_seed, model="flux",
                ))
            )
            end_out = await _submit_comfy_and_wait(wf_end, timeout_s=300)
            shutil.copyfile(str(COMFY_OUTPUT / end_out), str(COMFY_INPUT / end_image_input_name))
            prev_end_image_filename = end_out

            # Cache per-scene context so the keyframe-regen endpoint can re-run this scene
            # individually, and so the Wan phase below has everything it needs without
            # re-deriving prompts.
            if _active_gen is not None:
                _active_gen.keyframes.append({
                    "scene_index": i,
                    "start_image": page_thumb,
                    "end_image": end_out,
                    "start_input_name": start_image_input_name,
                    "end_input_name": end_image_input_name,
                    "description": scene_desc,
                    "motion_intensity": intensity,
                    "start_prompt": start_prompt,   # empty for page 1 when saved_character is used
                    "end_prompt": end_prompt,
                    "end_seed": end_seed,
                    "wan_prompt": _build_wan_prompt(
                        scene.get("video_prompt") or scene.get("motion") or "gentle motion",
                        starting_pose, ending_pose, character, p.style,
                    ),
                })
            step_done += 1

        # =================== CLIP drift detection ===================
        # Score every scene's start frame against the canonical character reference and
        # attach the cosine similarity to its keyframe entry, so the UI can flag scenes
        # whose character has drifted (per #4). Runs on CPU; takes a second or two.
        try:
            ref_path = COMFY_INPUT / ref_name
            scene_paths = [COMFY_OUTPUT / kf["start_image"] for kf in _active_gen.keyframes] if _active_gen else []
            if _active_gen and ref_path.exists() and scene_paths:
                sims = await drift.score_drift(ref_path, scene_paths)
                for kf, sim in zip(_active_gen.keyframes, sims):
                    kf["drift"] = sim   # None if scoring failed
                    kf["drift_flagged"] = (sim is not None and sim < drift.DRIFT_THRESHOLD)
        except Exception as e:
            print(f"[storybook] drift detection failed (non-fatal): {e}")

        # =================== Approval gate ===================
        if _active_gen is not None:
            _active_gen.node = "awaiting-approval"
            _active_gen.step = step_done
            _active_gen.approval_event.clear()
            _active_gen.approval_cancelled = False
            await _active_gen.approval_event.wait()
            if _active_gen is None or _active_gen.approval_cancelled:
                raise RuntimeError("storybook cancelled at preview")

        # =================== PHASE B — Wan animations per scene ===================
        page_videos: list[str] = []
        for kf in (_active_gen.keyframes if _active_gen else []):
            i = kf["scene_index"]
            if _active_gen is not None:
                _active_gen.node = f"page-{i+1}-animate"
                _active_gen.step = step_done

            i2v_params = GenerateParams(
                prompt=kf["wan_prompt"],
                negative=DEFAULT_NEGATIVE,
                width=width, height=height,
                frames=81, steps=20,
                cfg=6.0, shift=5.0, seed=seed,
                fps=16, scheduler="unipc",
                noise_aug=NOISE_AUG_BY_INTENSITY.get(kf["motion_intensity"], 0.05),
                image=kf["start_input_name"],
                end_image=kf["end_input_name"],
                multi_gpu=True,
                attention_mode="sageattn",
                block_swap_count=15, block_swap_device="cuda:1",
                vae_tiling=False,
                keep_t5_loaded=True,
                use_slg=True, use_feta=True, use_teacache=True,
            )
            wf_v = build_wan22_i2v_workflow(i2v_params)
            vid_filename = await _submit_comfy_and_wait(wf_v, timeout_s=2400)
            page_videos.append(str(COMFY_OUTPUT / vid_filename))
            kf["video"] = vid_filename   # remembered so per-scene-regen (#3) can find it later
            step_done += 1

        # --- Final stitch: hard-cut concat of raw Wan clips. ---
        # Because each page's start frame is a byte-perfect copy of the previous page's
        # end frame, the cuts are invisible and the result reads as one continuous shot.
        if _active_gen is not None:
            _active_gen.node = "stitching"
            _active_gen.step = total
            _active_gen.total_steps = total

        final_filename = f"wan_studio_storybook_{gen_id}.mp4"
        final_path = COMFY_OUTPUT / final_filename
        await _stitch_videos(page_videos, str(final_path), hold_dur=0.0)

        # Save to history. We persist the full keyframe metadata so the per-scene
        # regenerate endpoint (#3) can re-animate a single scene later without
        # re-deriving prompts, seeds, or input filenames.
        scene_records = [
            {
                "scene_index": kf["scene_index"],
                "start_image": kf["start_image"],
                "end_image": kf["end_image"],
                "start_input_name": kf["start_input_name"],
                "end_input_name": kf["end_input_name"],
                "video": kf.get("video"),
                "wan_prompt": kf["wan_prompt"],
                "motion_intensity": kf["motion_intensity"],
                "description": kf["description"],
            }
            for kf in (_active_gen.keyframes if _active_gen else [])
        ]
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
                "scenes_meta": scene_records,
                "width": width,
                "height": height,
                "seed": seed,
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
    if p.n_pages < 2 or p.n_pages > 12:
        raise HTTPException(400, "n_pages must be 2..12")
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


# ---------- Storybook keyframe-preview approval gate (#2) ----------

class RegenerateKeyframeParams(BaseModel):
    scene_index: int
    frame: str = Field("end", pattern="^(start|end)$")


@app.post("/api/storybook/approve")
async def storybook_approve():
    """Tell the orchestrator that the user is happy with the keyframes — proceed to Wan."""
    if _active_gen is None or _active_gen.kind != "storybook":
        raise HTTPException(409, "no storybook awaiting approval")
    if _active_gen.node != "awaiting-approval":
        raise HTTPException(409, f"storybook is in '{_active_gen.node}' state, not waiting for approval")
    _active_gen.approval_cancelled = False
    _active_gen.approval_event.set()
    return {"ok": True}


@app.post("/api/storybook/cancel_approval")
async def storybook_cancel_approval():
    """User rejected the keyframes; abort the storybook cleanly without running Wan."""
    if _active_gen is None or _active_gen.kind != "storybook":
        raise HTTPException(409, "no storybook awaiting approval")
    if _active_gen.node != "awaiting-approval":
        raise HTTPException(409, f"storybook is in '{_active_gen.node}' state, not waiting for approval")
    _active_gen.approval_cancelled = True
    _active_gen.approval_event.set()
    return {"ok": True}


@app.post("/api/storybook/regenerate_keyframe")
async def storybook_regenerate_keyframe(p: RegenerateKeyframeParams):
    """While the storybook is paused at the approval gate, re-run a single Flux frame
    (start or end of scene N) with a fresh seed. Updates the keyframe cache in place
    so the preview strip shows the new image. Wan has not started yet, so this is cheap."""
    if _active_gen is None or _active_gen.kind != "storybook":
        raise HTTPException(409, "no storybook awaiting approval")
    if _active_gen.node != "awaiting-approval":
        raise HTTPException(409, "regen only allowed during keyframe approval")
    keyframes = _active_gen.keyframes
    if not (0 <= p.scene_index < len(keyframes)):
        raise HTTPException(400, "scene_index out of range")
    if p.frame == "start" and p.scene_index > 0:
        raise HTTPException(400, "page 2+ start frames are byte-perfect copies of the previous end; regen that end frame instead")

    kf = keyframes[p.scene_index]
    gen_id = _active_gen.gen_id
    params = _active_gen.params or {}
    width = STORY_ASPECT_DIMS.get(params.get("aspect", "landscape"), STORY_ASPECT_DIMS["landscape"])[0]
    height = STORY_ASPECT_DIMS.get(params.get("aspect", "landscape"), STORY_ASPECT_DIMS["landscape"])[1]
    ref_name = f"storybook_charref_{gen_id}.png"
    kontext_available = (
        Path("/media/yunus/More Data/comfyui-models/diffusion_models") / FLUX_KONTEXT_MODEL
    ).exists() and (COMFY_INPUT / ref_name).exists()

    new_seed = int(time.time() * 1000) % (2**31)
    await _comfy_free()
    if p.frame == "start":
        prompt_text = kf.get("start_prompt") or ""
        if not prompt_text:
            raise HTTPException(409, "this start frame is from a saved character and cannot be regenerated")
        wf = build_flux_image_workflow(ImageGenerateParams(
            image_mode="create", prompt=prompt_text,
            width=width, height=height, seed=new_seed, model="flux",
        ))
        out = await _submit_comfy_and_wait(wf, timeout_s=300)
        shutil.copyfile(str(COMFY_OUTPUT / out), str(COMFY_INPUT / kf["start_input_name"]))
        shutil.copyfile(str(COMFY_OUTPUT / out), str(COMFY_INPUT / ref_name))
        kf["start_image"] = out
        if _active_gen.preview_images:
            _active_gen.preview_images[0] = out
        await _rescore_drift_for_active()
        return {"ok": True, "filename": out}
    else:
        prompt_text = kf["end_prompt"]
        wf = (
            build_flux_kontext_workflow(
                prompt=prompt_text, width=width, height=height,
                seed=new_seed, reference_image=ref_name, steps=20,
            )
            if kontext_available
            else build_flux_image_workflow(ImageGenerateParams(
                image_mode="create", prompt=prompt_text,
                width=width, height=height, seed=new_seed, model="flux",
            ))
        )
        out = await _submit_comfy_and_wait(wf, timeout_s=300)
        shutil.copyfile(str(COMFY_OUTPUT / out), str(COMFY_INPUT / kf["end_input_name"]))
        kf["end_image"] = out
        # If a later scene took this scene's end as its start, propagate the change so
        # FLF2V chaining stays byte-perfect.
        if p.scene_index + 1 < len(keyframes):
            next_kf = keyframes[p.scene_index + 1]
            shutil.copyfile(str(COMFY_OUTPUT / out), str(COMFY_INPUT / next_kf["start_input_name"]))
            next_kf["start_image"] = out
            if _active_gen.preview_images and (p.scene_index + 1) < len(_active_gen.preview_images):
                _active_gen.preview_images[p.scene_index + 1] = out
        await _rescore_drift_for_active()
        return {"ok": True, "filename": out}


async def _rescore_drift_for_active() -> None:
    """Recompute CLIP drift scores against the canonical reference for the
    currently-active storybook gen. No-op if nothing is active."""
    if _active_gen is None or _active_gen.kind != "storybook":
        return
    ref_name = f"storybook_charref_{_active_gen.gen_id}.png"
    ref_path = COMFY_INPUT / ref_name
    if not ref_path.exists():
        return
    try:
        scene_paths = [COMFY_OUTPUT / kf["start_image"] for kf in _active_gen.keyframes]
        sims = await drift.score_drift(ref_path, scene_paths)
        for kf, sim in zip(_active_gen.keyframes, sims):
            kf["drift"] = sim
            kf["drift_flagged"] = (sim is not None and sim < drift.DRIFT_THRESHOLD)
    except Exception as e:
        print(f"[storybook] drift rescore failed: {e}")


# ---------- Per-scene Wan regenerate after a storybook has finished (#3) ----------

class RegenerateSceneParams(BaseModel):
    gen_id: str
    scene_index: int


async def _restitch_storybook_from_history(entry: dict) -> str:
    """Read scenes_meta from a history entry, concat the per-scene videos into the
    final stitched MP4 (overwriting whatever was there). Returns the final filename."""
    final_filename = entry["filename"]
    scenes_meta = (entry.get("params") or {}).get("scenes_meta") or []
    page_video_paths = [
        str(COMFY_OUTPUT / s["video"]) for s in scenes_meta if s.get("video")
    ]
    if not page_video_paths:
        raise RuntimeError("no per-scene videos found to stitch")
    await _stitch_videos(page_video_paths, str(COMFY_OUTPUT / final_filename), hold_dur=0.0)
    return final_filename


@app.post("/api/storybook/regenerate_scene")
async def storybook_regenerate_scene(p: RegenerateSceneParams):
    """Re-animate a single scene's Wan clip using the cached keyframes from the original
    run, then re-stitch the final video. The other scenes are untouched."""
    global _active_gen, _last_error
    if _state_lock is None:
        raise HTTPException(503, "backend not ready")
    items = _load_json(HISTORY_FILE, [])
    entry = next((it for it in items if it.get("id") == p.gen_id and it.get("kind") == "storybook"), None)
    if not entry:
        raise HTTPException(404, "no storybook with that gen_id in history")
    params = entry.get("params") or {}
    scenes_meta = params.get("scenes_meta") or []
    target = next((s for s in scenes_meta if int(s.get("scene_index", -1)) == p.scene_index), None)
    if not target:
        raise HTTPException(404, "scene_index not found in this storybook's metadata")
    # Sanity: input frames must still be on disk
    for fname in (target["start_input_name"], target["end_input_name"]):
        if not (COMFY_INPUT / fname).exists():
            raise HTTPException(409, f"input frame '{fname}' is no longer on disk — can't regen")

    async with _state_lock:
        if _active_gen is not None:
            raise HTTPException(409, "another generation is in progress")
        synthetic_prompt_id = f"regen-{p.gen_id}-{p.scene_index}-{int(time.time())}"
        _active_gen = GenState(
            prompt_id=synthetic_prompt_id,
            gen_id=p.gen_id,
            params={**params, "regenerating_scene": p.scene_index},
            started_at=time.time(),
            kind="storybook",
        )
        _active_gen.node = f"page-{p.scene_index+1}-animate"
        _active_gen.step = 1
        _active_gen.total_steps = 2
        _last_error = None

    asyncio.create_task(_do_regenerate_scene(p.gen_id, p.scene_index, target, params))
    return {"ok": True, "prompt_id": synthetic_prompt_id}


async def _do_regenerate_scene(gen_id: str, scene_index: int, target: dict, params: dict):
    """Background task: re-run Wan for one scene, write the new clip into the existing
    scenes_meta slot, then re-stitch the final video."""
    global _active_gen, _last_error
    NOISE_AUG_BY_INTENSITY = {"still": 0.0, "gentle": 0.05, "dynamic": 0.10}
    try:
        await llm.unload()
        i2v_params = GenerateParams(
            prompt=target["wan_prompt"],
            negative=DEFAULT_NEGATIVE,
            width=int(params.get("width") or 1024),
            height=int(params.get("height") or 576),
            frames=81, steps=20,
            cfg=6.0, shift=5.0,
            # Bump seed so the regen is actually different from the original
            seed=int(time.time() * 1000) % (2**31),
            fps=16, scheduler="unipc",
            noise_aug=NOISE_AUG_BY_INTENSITY.get(target.get("motion_intensity") or "gentle", 0.05),
            image=target["start_input_name"],
            end_image=target["end_input_name"],
            multi_gpu=True,
            attention_mode="sageattn",
            block_swap_count=15, block_swap_device="cuda:1",
            vae_tiling=False,
            keep_t5_loaded=True,
            use_slg=True, use_feta=True, use_teacache=True,
        )
        wf = build_wan22_i2v_workflow(i2v_params)
        new_video = await _submit_comfy_and_wait(wf, timeout_s=2400)

        # Patch the history entry: swap in the new per-scene video filename, then restitch.
        if _active_gen is not None:
            _active_gen.node = "stitching"
            _active_gen.step = 2
        items = _load_json(HISTORY_FILE, [])
        for it in items:
            if it.get("id") != gen_id or it.get("kind") != "storybook":
                continue
            sm = (it.get("params") or {}).get("scenes_meta") or []
            for s in sm:
                if int(s.get("scene_index", -1)) == scene_index:
                    s["video"] = new_video
                    break
            # update the params.page_videos mirror (legacy)
            it["params"]["page_videos"] = [s.get("video") for s in sm if s.get("video")]
            await _restitch_storybook_from_history(it)
            _save_json(HISTORY_FILE, items)
            break
    except Exception as e:
        print(f"[storybook] scene regen failed: {e}")
        _last_error = f"scene regen failed: {e}"
    finally:
        _active_gen = None


# ---------- Character library (re-use a protagonist across multiple stories) ----------

@app.get("/api/characters")
async def list_characters():
    return {"items": _load_json(CHARACTER_LIBRARY_FILE, [])}


@app.post("/api/characters")
async def save_character(p: SaveCharacterParams):
    """Persist the character from a completed storybook gen into the library.
    Pulls the canonical reference image (page 1's start frame, saved under
    `input/storybook_charref_<gen_id>.png`) and the canon dict from history."""
    items = _load_json(HISTORY_FILE, [])
    entry = next((it for it in items if it.get("id") == p.gen_id and it.get("kind") == "storybook"), None)
    if not entry:
        raise HTTPException(404, "no storybook with that gen_id in history")

    plan = (entry.get("params") or {}).get("plan") or {}
    canon = plan.get("character_canon") or {}
    character_prose = plan.get("character") or ""

    src_ref = COMFY_INPUT / f"storybook_charref_{p.gen_id}.png"
    if not src_ref.exists():
        raise HTTPException(404, "character reference image no longer on disk for that gen")

    CHARACTER_LIBRARY_DIR.mkdir(exist_ok=True)
    char_id = uuid.uuid4().hex[:10]
    ref_filename = f"char_{char_id}.png"
    shutil.copyfile(str(src_ref), str(CHARACTER_LIBRARY_DIR / ref_filename))

    saved = {
        "id": char_id,
        "name": p.name.strip() or (canon.get("name") or "Character").strip(),
        "canon": canon,
        "character": character_prose,
        "ref_filename": ref_filename,
        "created_at": time.time(),
        "source_gen_id": p.gen_id,
    }
    library = _load_json(CHARACTER_LIBRARY_FILE, [])
    library.insert(0, saved)
    _save_json(CHARACTER_LIBRARY_FILE, library[:100])
    return saved


@app.delete("/api/characters/{char_id}")
async def delete_character(char_id: str):
    library = _load_json(CHARACTER_LIBRARY_FILE, [])
    target = next((c for c in library if c.get("id") == char_id), None)
    library = [c for c in library if c.get("id") != char_id]
    _save_json(CHARACTER_LIBRARY_FILE, library)
    if target:
        ref = CHARACTER_LIBRARY_DIR / (target.get("ref_filename") or "")
        if ref.exists():
            try: ref.unlink()
            except Exception: pass
    return {"ok": True}


@app.get("/api/characters/{char_id}/image")
async def get_character_image(char_id: str):
    library = _load_json(CHARACTER_LIBRARY_FILE, [])
    target = next((c for c in library if c.get("id") == char_id), None)
    if not target:
        raise HTTPException(404, "character not found")
    ref_path = CHARACTER_LIBRARY_DIR / (target.get("ref_filename") or "")
    if not ref_path.exists():
        raise HTTPException(404, "reference image missing")
    return FileResponse(ref_path, media_type="image/png")


@app.post("/api/llm/improve")
async def llm_improve(p: ImprovePromptParams):
    if not await llm.is_available():
        raise HTTPException(503, "Local LLM not available")
    # Free ComfyUI's cached models so the (~23 GB) qwen3.6 can actually load on GPU.
    await _comfy_free()
    try:
        out = await llm.improve_prompt(p.prompt, style=p.style or None)
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
