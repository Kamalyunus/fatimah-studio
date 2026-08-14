"""Wan 2.2 I2V workflow builders (ComfyUI node graphs as plain dicts).

Pure functions: they take a GenerateParams and emit the node dict the orchestrator submits.
No global state, no I/O."""
from __future__ import annotations

from config import (
    CLIP_VISION_MODEL,
    DEFAULT_NEGATIVE,
    T5_MODEL,
    VACE_STRENGTH,
    VAE_MODEL,
    WAN22_I2V_HIGH,
    WAN22_I2V_LOW,
    WAN22_T2V_HIGH,
    WAN22_T2V_LOW,
    WAN22_VACE_HIGH,
    WAN22_VACE_LOW,
)

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


def _model_loader_node_named(model_filename: str, p: "GenerateParams", with_block_swap: bool, node_key: str = "model_loader",
                             quantization: str = "fp8_e4m3fn", vace_select_key: str | None = None) -> dict:
    """Wan 2.2 model loader (named so the two-expert MoE pattern can have separate nodes).
    vace_select_key wires a WanVideoVACEModelSelect node's VACEPATH into the loader,
    grafting the VACE module onto the base model."""
    inputs = {
        "model": model_filename,
        "base_precision": "bf16",
        "quantization": quantization,
        "load_device": "offload_device",   # cold-load; sampler will hot it
        "attention_mode": p.attention_mode,
    }
    if p.multi_gpu:
        inputs["compute_device"] = "cuda:0"
    if with_block_swap:
        inputs["block_swap_args"] = ["block_swap", 0]
    if vace_select_key:
        inputs["vace_model"] = [vace_select_key, 0]
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


# Gray fill for VACE frames that are to be generated (0x7F7F7F); masks are built from
# black (keep this frame) / white (generate this frame) images converted to MASK.
_VACE_GRAY = 8355711
_MASK_KEEP = 0          # black
_MASK_GENERATE = 16777215  # white


def build_wan22_vace_workflow(p: "GenerateParams") -> dict:
    """Wan 2.2 MoE + Fun-VACE modules: reference-conditioned first/last-frame animation.

    Replaces the I2V conditioning path entirely — no CLIP-vision, no i2v_encode. Instead:
      * input_frames  = [start keyframe, gray x (frames-2), end keyframe]
      * input_masks   = [keep,           generate x (frames-2), keep]
      * ref_images    = character sheet (p.vace_ref_image), padded/encoded as an extra
                        latent frame so the model holds identity in EVERY generated frame.
    The VACE modules graft onto the Wan 2.2 *T2V* experts via the loader's vace_model
    input (I2V bases are incompatible by design). Sampler/decode tail mirrors the I2V
    builder's MoE split."""
    mg = p.multi_gpu
    total_steps = max(2, int(p.steps))
    boundary = total_steps // 2
    n_frames = int(p.frames)
    block_swap = _block_swap_node(p)

    wf: dict = {}
    if block_swap:
        if mg and p.block_swap_count > 0:
            # VACE adds 15 extra transformer blocks per expert; swap them alongside the
            # base blocks so VRAM headroom matches the tuned I2V configuration.
            block_swap["inputs"]["vace_blocks_to_swap"] = min(15, p.block_swap_count)
        wf["block_swap"] = block_swap

    # VACE modules graft onto the T2V experts (KJ fp8-scaled weights → scaled quantization)
    wf["vace_sel_high"] = {"class_type": "WanVideoVACEModelSelect", "inputs": {"vace_model": WAN22_VACE_HIGH}}
    wf["vace_sel_low"] = {"class_type": "WanVideoVACEModelSelect", "inputs": {"vace_model": WAN22_VACE_LOW}}
    wf["model_high"] = _model_loader_node_named(WAN22_T2V_HIGH, p, with_block_swap=bool(block_swap),
                                                quantization="fp8_e4m3fn_scaled", vace_select_key="vace_sel_high")
    wf["model_low"] = _model_loader_node_named(WAN22_T2V_LOW, p, with_block_swap=bool(block_swap),
                                               quantization="fp8_e4m3fn_scaled", vace_select_key="vace_sel_low")

    wf["t5"] = _t5_node(p)
    wf["vae"] = {"class_type": "WanVideoVAELoader", "inputs": {"model_name": VAE_MODEL, "precision": "bf16"}}
    wf["text_encode"] = _text_encode_node(p)

    # ---- VACE control frames: start + gray middle + end, with keep/generate masks ----
    wf["load_image"] = {"class_type": "LoadImage", "inputs": {"image": p.image}}
    wf["gray_mid"] = {"class_type": "EmptyImage",
                      "inputs": {"width": p.width, "height": p.height, "batch_size": max(1, n_frames - 2), "color": _VACE_GRAY}}
    wf["frames_head"] = {"class_type": "ImageBatch", "inputs": {"image1": ["load_image", 0], "image2": ["gray_mid", 0]}}

    wf["mask_first"] = {"class_type": "EmptyImage",
                        "inputs": {"width": p.width, "height": p.height, "batch_size": 1, "color": _MASK_KEEP}}
    wf["mask_mid"] = {"class_type": "EmptyImage",
                      "inputs": {"width": p.width, "height": p.height, "batch_size": max(1, n_frames - 2), "color": _MASK_GENERATE}}
    wf["masks_head"] = {"class_type": "ImageBatch", "inputs": {"image1": ["mask_first", 0], "image2": ["mask_mid", 0]}}

    if p.end_image:
        wf["load_end_image"] = {"class_type": "LoadImage", "inputs": {"image": p.end_image}}
        wf["frames_all"] = {"class_type": "ImageBatch", "inputs": {"image1": ["frames_head", 0], "image2": ["load_end_image", 0]}}
        wf["mask_last"] = {"class_type": "EmptyImage",
                           "inputs": {"width": p.width, "height": p.height, "batch_size": 1, "color": _MASK_KEEP}}
        wf["masks_all"] = {"class_type": "ImageBatch", "inputs": {"image1": ["masks_head", 0], "image2": ["mask_last", 0]}}
        frames_key, masks_key = "frames_all", "masks_all"
    else:
        # start-frame-only: last frame is generated too (mask stays white)
        frames_key, masks_key = "frames_head", "masks_head"

    wf["masks"] = {"class_type": "ImageToMask", "inputs": {"image": [masks_key, 0], "channel": "red"}}

    vace_inputs = {
        "vae": ["vae", 0],
        "width": p.width, "height": p.height, "num_frames": n_frames,
        "strength": VACE_STRENGTH,
        "vace_start_percent": 0.0, "vace_end_percent": 1.0,
        "input_frames": [frames_key, 0],
        "input_masks": ["masks", 0],
        "tiled_vae": p.vae_tiling,
    }
    if p.vace_ref_image:
        wf["load_ref"] = {"class_type": "LoadImage", "inputs": {"image": p.vace_ref_image}}
        vace_inputs["ref_images"] = ["load_ref", 0]
    wf["vace_encode"] = {"class_type": "WanVideoVACEEncode", "inputs": vace_inputs}

    # ---- Quality knobs + MoE sampler split (mirrors build_wan22_i2v_workflow) ----
    quality_args: dict = {}
    if p.use_slg:
        wf["slg"] = {"class_type": "WanVideoSLG",
                     "inputs": {"blocks": "9", "start_percent": 0.1, "end_percent": 0.5}}
        quality_args["slg_args"] = ["slg", 0]
    if p.use_feta:
        wf["feta"] = {"class_type": "WanVideoEnhanceAVideo",
                      "inputs": {"weight": 2.0, "start_percent": 0.0, "end_percent": 1.0}}
        quality_args["feta_args"] = ["feta", 0]
    if p.use_teacache:
        wf["teacache"] = {"class_type": "WanVideoTeaCache",
                          "inputs": {"rel_l1_thresh": 0.20, "start_step": 1,
                                     "end_step": max(2, total_steps - 1),
                                     "cache_device": "main_device", "use_coefficients": True, "mode": "e"}}
        quality_args["cache_args"] = ["teacache", 0]

    sampler_inputs_common = {
        "image_embeds": ["vace_encode", 0],
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
        "inputs": {**sampler_inputs_common, "model": ["model_high", 0], "start_step": 0, "end_step": boundary},
    }
    wf["sampler_low"] = {
        "class_type": "WanVideoSamplerMultiGPU" if mg else "WanVideoSampler",
        "inputs": {**sampler_inputs_common, "model": ["model_low", 0], "samples": ["sampler_high", 0],
                   "start_step": boundary, "end_step": total_steps, "add_noise_to_samples": False},
    }

    wf["decode"] = {
        "class_type": "WanVideoDecode",
        "inputs": {"vae": ["vae", 0], "samples": ["sampler_low", 0],
                   "enable_vae_tiling": p.vae_tiling,
                   "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128},
    }
    wf["save"] = _save_node("wan_studio_vace_v22", p.fps)
    return wf
