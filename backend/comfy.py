"""ComfyUI submission + media helpers.

`submit + wait` talks to ComfyUI's HTTP API; the media helpers (thumb, last-frame, probe,
stitch) shell out to ffmpeg/ffprobe. No app state — safe to import from anywhere."""
from __future__ import annotations

import asyncio
import subprocess
import time
import uuid
from pathlib import Path

import httpx

from config import COMFY_HTTP, XFADE_DUR


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


def _extract_last_frame(video_path: Path, out_path: Path) -> bool:
    """Grab a clip's FINAL decoded frame as a lossless PNG.

    Used to chain storybook pages on the ACTUAL rendered frame rather than the Kontext
    target keyframe. Wan's FLF2V undershoots its end keyframe, so page N's last frame and
    page N+1's first frame (which faithfully reproduces its start keyframe) don't match —
    that gap is the seam pop. Starting page N+1 from page N's real last frame closes it by
    construction. `-sseof -1` decodes only the last ~second and `-update 1` keeps
    overwriting the same file, so what lands on disk is the true final frame."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-sseof", "-1", "-i", str(video_path),
             "-update", "1", str(out_path)],
            check=True, capture_output=True, timeout=30,
        )
        return out_path.exists()
    except Exception as e:
        print(f"[storybook] last-frame extract failed for {video_path.name}: {e}")
        return False


def _probe_duration(path: str) -> float:
    """Clip duration in seconds (for building the crossfade offsets). Falls back to the
    default storybook clip length (81 frames @ 16fps) if ffprobe can't read it."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=20,
        )
        return float(out.stdout.strip())
    except Exception:
        return 81 / 16


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


async def _stitch_videos(paths: list[str], output_path: str):
    """Stitch silent Wan clips into one continuous video.

    Pages are chained on each clip's ACTUAL last rendered frame (see Phase B), so adjacent
    boundary frames already match. On top of that we apply a short xfade between every pair
    so the motion eases across the seam instead of stopping-and-restarting at each clip's
    static keyframe. A genuine location change is a slightly bigger but still gentle
    dissolve, which reads fine for a storybook."""
    if not paths:
        raise ValueError("no paths to stitch")

    common_out = [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
        "-movflags", "+faststart", output_path,
    ]

    # Single clip: nothing to crossfade, just transcode to the final container.
    if len(paths) == 1:
        args = ["ffmpeg", "-y", "-i", paths[0], *common_out]
    else:
        durs = [_probe_duration(p) for p in paths]
        # Clamp the fade so it can never exceed half of the shortest clip (avoids a
        # negative xfade offset on unexpectedly short clips).
        d = min(XFADE_DUR, min(durs) * 0.5)

        args = ["ffmpeg", "-y"]
        for p in paths:
            args.extend(["-i", p])

        # xfade overlays each next clip onto the running stream; offset is measured in the
        # running stream's timeline. After folding in clip k the running length grows by
        # (dur_k - d), so the next offset is (running_length - d).
        filters: list[str] = []
        prev_label = "0:v"
        running = durs[0]
        n = len(paths)
        for idx in range(1, n):
            out_label = "outv" if idx == n - 1 else f"x{idx}"
            offset = running - d
            filters.append(
                f"[{prev_label}][{idx}:v]xfade=transition=fade:"
                f"duration={d:.4f}:offset={offset:.4f}[{out_label}]"
            )
            prev_label = out_label
            running = running + durs[idx] - d

        args.extend(["-filter_complex", ";".join(filters), "-map", "[outv]", *common_out])

    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg stitch failed: {stderr.decode()[:500]}")
