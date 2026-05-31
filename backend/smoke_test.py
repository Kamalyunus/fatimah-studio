#!/usr/bin/env python3
"""Fast smoke test for the storybook coherency changes.

Fires a SHORT, LOW-STEP, auto-approved storybook run (StorybookParams.smoke=True) through
the real backend, watches progress, and on completion extracts the frames straddling every
page seam so you can eyeball:

  * background-lock  — does the room stay stable within and across clips? (img2img keyframes)
  * seam continuity  — is the last frame of clip N == first frame of clip N+1? (frame-chaining)
  * crossfade        — is the cut smooth rather than a hard pop?

Prerequisites (same as a normal run): the studio backend (uvicorn main:app :8000), ComfyUI,
and Ollama (llama3.2:3b) must all be up. Nothing here is destructive.

Usage:
    python smoke_test.py
    python smoke_test.py --story "A curious cat named Mochi who builds a tiny rocket"
    python smoke_test.py --base http://127.0.0.1:8000 --style pixar --aspect landscape
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx

# Wan writes finished videos here; matches COMFY_OUTPUT in main.py.
COMFY_OUTPUT = Path("/home/yunus/Documents/comfyui/output")
SMOKE_FRAMES = 33   # keep in sync with vid_frames in main.py's smoke mode
FPS = 16


def _seam_frames(video: Path, n_clips: int, out_dir: Path) -> list[Path]:
    """Extract the two frames either side of each page seam (last of clip k, first of k+1)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    # With a crossfade the seam isn't on an exact frame index, but the per-clip length is
    # ~SMOKE_FRAMES, so sampling around each boundary is good enough for a visual check.
    for k in range(1, n_clips):
        boundary = k * SMOKE_FRAMES
        for f in (boundary - 1, boundary):
            dst = out_dir / f"seam{k}_f{f}.jpg"
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{f})",
                 "-vframes", "1", str(dst)],
                capture_output=True,
            )
            if dst.exists():
                saved.append(dst)
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--story", default="A friendly little robot named Bolt who really wants to learn how to bake cookies.")
    ap.add_argument("--style", default="pixar")
    ap.add_argument("--aspect", default="landscape")
    ap.add_argument("--timeout", type=int, default=1800, help="max seconds to wait")
    args = ap.parse_args()

    body = {"story": args.story, "style": args.style, "aspect": args.aspect, "smoke": True}

    print(f"→ POST {args.base}/api/storybook  (smoke=True)")
    try:
        r = httpx.post(f"{args.base}/api/storybook", json=body, timeout=30)
    except Exception as e:
        print(f"!! could not reach backend: {e}\n   Is uvicorn main:app :8000 running?")
        return 2
    if r.status_code != 200:
        print(f"!! {r.status_code}: {r.text}")
        return 2
    info = r.json()
    gen_id = info["gen_id"]
    final_name = f"wan_studio_storybook_{gen_id}.mp4"
    print(f"   gen_id={gen_id}  → final video will be {final_name}")
    print("   watching progress (backend console shows per-page keyframe route: kontext / location-cut / plain-flux)…")

    last_node = None
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        try:
            s = httpx.get(f"{args.base}/api/state", timeout=10).json()
        except Exception:
            time.sleep(2)
            continue
        active = s.get("active")
        err = s.get("last_error")
        if active and active.get("node") != last_node:
            last_node = active.get("node")
            print(f"   … {last_node}")
        if not active:
            if err:
                print(f"!! run ended with error: {err}")
                return 1
            break
        time.sleep(3)
    else:
        print("!! timed out waiting for the run to finish")
        return 1

    video = COMFY_OUTPUT / final_name
    if not video.exists():
        print(f"!! finished but {video} not found — check the backend console")
        return 1

    print(f"\n✓ done: {video}")
    seam_dir = COMFY_OUTPUT / f"smoke_seams_{gen_id}"
    frames = _seam_frames(video, n_clips=3, out_dir=seam_dir)
    print(f"✓ extracted {len(frames)} seam frames into {seam_dir}")
    for f in frames:
        print(f"    {f}")
    print("\nInspect: each pair (last frame of clip N vs first of clip N+1) should show the")
    print("same room, and within a clip the background should not reshuffle. Compare to the")
    print("old run wan_studio_storybook_48a2dc74.mp4 (frames 161/162) where the room popped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
