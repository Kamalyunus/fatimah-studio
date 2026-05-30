"""Flux / SDXL image + upscale workflow builders (ComfyUI node graphs as plain dicts).

Pure functions: they take a *Params object and emit the node dict the caller submits.
No global state."""
from __future__ import annotations

import time

from config import (
    DEFAULT_NEGATIVE,
    FLUX_CLIP_L,
    FLUX_KONTEXT_MODEL,
    FLUX_MODEL,
    FLUX_T5,
    FLUX_VAE,
    KONTEXT_EDIT_DENOISE,
    SDXL_LIGHTNING_LORA,
    SDXL_MODEL,
    UPSCALER_MODEL,
)

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


def build_flux_kontext_edit_workflow(
    prompt: str, seed: int, edit_image: str, reference_image: str | None = None,
    steps: int = 24, denoise: float = KONTEXT_EDIT_DENOISE,
) -> dict:
    """Flux Kontext as an IMG2IMG edit (background-locked end-keyframe).

    Unlike build_flux_kontext_workflow — which samples from an EMPTY latent at denoise 1.0
    and therefore re-invents the whole scene (props/furniture reshuffle every call) — this
    starts the sampler from edit_image's OWN latent at partial denoise, so the existing
    composition, crucially the BACKGROUND, is preserved while the prompt nudges the
    character into the new pose. Used for same-location storybook end-keyframes so the room
    stays pixel-stable across the clip. reference_image (defaults to edit_image) supplies the
    Kontext ReferenceLatent that locks appearance — pass the multi-character composite when
    other characters must be held."""
    ref = reference_image or edit_image
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
        # img2img base — the page's start frame (correct room + character).
        "load_edit": {"class_type": "LoadImage", "inputs": {"image": edit_image}},
        "scale_edit": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["load_edit", 0]}},
        "encode_edit": {"class_type": "VAEEncode", "inputs": {"pixels": ["scale_edit", 0], "vae": ["vae", 0]}},
        # appearance-lock reference (Kontext context tokens).
        "load_ref": {"class_type": "LoadImage", "inputs": {"image": ref}},
        "scale_ref": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["load_ref", 0]}},
        "encode_ref": {"class_type": "VAEEncode", "inputs": {"pixels": ["scale_ref", 0], "vae": ["vae", 0]}},
        "positive": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "ref_latent": {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": ["positive", 0], "latent": ["encode_ref", 0]},
        },
        "guidance": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["ref_latent", 0], "guidance": 3.5},
        },
        "negative": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["positive", 0]},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["unet", 0],
                "positive": ["guidance", 0],
                "negative": ["negative", 0],
                "latent_image": ["encode_edit", 0],   # <-- start from the start-frame latent
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise,
                "seed": seed,
            },
        },
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
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

