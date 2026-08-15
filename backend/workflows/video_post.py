"""Post-processing workflows that run on the GPU through ComfyUI.

Currently just frame interpolation. Wan renders at 16fps, which is the single biggest
reason its output reads as "an AI clip" rather than film — interpolating to 32fps (and
conforming to 24 at mux time) removes most of that judder for a few seconds of GPU work
per clip, with no re-rendering of anything.
"""
from __future__ import annotations

from config import INTERP_CKPT, INTERP_MULTIPLIER, VID_FPS


def build_interpolate_workflow(
    video_path: str,
    filename_prefix: str,
    multiplier: int = INTERP_MULTIPLIER,
    source_fps: int = VID_FPS,
) -> dict:
    """Load a rendered clip, run RIFE over it, write the result back out.

    `clear_cache_after_n_frames` keeps VRAM flat on long clips; `fast_mode` and no
    ensemble are the standard speed/quality tradeoff for content this gentle.
    """
    return {
        "load_video": {
            "class_type": "VHS_LoadVideoPath",
            "inputs": {
                "video": video_path,
                "force_rate": 0.0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        },
        "rife": {
            "class_type": "RIFE VFI",
            "inputs": {
                "ckpt_name": INTERP_CKPT,
                "frames": ["load_video", 0],
                "clear_cache_after_n_frames": 10,
                "multiplier": multiplier,
                "fast_mode": True,
                "ensemble": False,
                "scale_factor": 1.0,
                "dtype": "float16",
                "torch_compile": False,
                "batch_size": 1,
            },
        },
        "save": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["rife", 0],
                "frame_rate": float(source_fps * multiplier),
                "loop_count": 0,
                "filename_prefix": filename_prefix,
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    }
