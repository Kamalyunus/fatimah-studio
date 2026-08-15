"""Post-production: conforming clips to narration, the animatic, and audio muxing.

Everything here is ffmpeg. The organising idea is that narration is authored first and
each page's picture is cut to fit its spoken line, rather than every page being a
uniform 5 seconds — so these helpers all take a per-page duration derived from the
narration WAV.
"""
from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from comfy import _probe_duration
from config import (
    GRAIN_STRENGTH,
    LOUDNORM_I,
    LOUDNORM_LRA,
    LOUDNORM_TP,
    MUSIC_GAIN,
    NARRATION_LEAD_IN,
    NARRATION_TAIL,
    OUTPUT_FPS,
    XFADE_DUR,
)

# How far a clip may be slowed or sped to meet its narration length before we stop
# retiming and pad with a held frame instead. Beyond these the motion reads wrong.
_RETIME_MIN = 0.80
_RETIME_MAX = 1.45


async def _run_ffmpeg(args: list[str], what: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg {what} failed: {stderr.decode()[-600:]}")


def page_duration(narration_dur: float, fallback: float) -> float:
    """On-screen length for a page: its spoken line plus a beat either side.

    `fallback` is used for pages with no narration (the clip's own natural length).
    """
    if narration_dur <= 0:
        return fallback
    return NARRATION_LEAD_IN + narration_dur + NARRATION_TAIL


def frames_for_duration(dur: float, fps: int, min_frames: int, max_frames: int) -> int:
    """Frame count for a target duration, snapped to Wan's 4n+1 requirement."""
    raw = max(1, round(dur * fps))
    snapped = ((raw - 1) // 4) * 4 + 1
    if snapped < raw:
        snapped += 4
    return max(min_frames, min(max_frames, snapped))


async def conform_clip(src: Path, dst: Path, target_dur: float) -> float:
    """Retime `src` to `target_dur`, padding with a held last frame if it can't stretch far
    enough. Returns the achieved duration.

    Wan renders a fixed frame count, so a clip lands near its narration length but rarely
    on it. A gentle speed change is invisible; a big one is not, hence the clamp plus
    freeze-frame fallback (a storybook page holding a beat longer reads fine).
    """
    actual = _probe_duration(str(src))
    if actual <= 0 or target_dur <= 0:
        raise ValueError(f"cannot conform {src.name}: actual={actual} target={target_dur}")

    ratio = target_dur / actual                      # >1 means we need it longer
    ratio = max(_RETIME_MIN, min(_RETIME_MAX, ratio))
    # setpts multiplies timestamps: factor >1 slows the clip down (makes it longer).
    filters = f"setpts={ratio:.5f}*PTS"
    retimed = actual * ratio
    if retimed < target_dur - 0.02:
        # Still short after the maximum allowed slow-down — hold the final frame.
        filters += f",tpad=stop_mode=clone:stop_duration={target_dur - retimed:.4f}"

    await _run_ffmpeg(
        ["ffmpeg", "-y", "-i", str(src), "-vf", filters,
         "-t", f"{target_dur:.4f}", "-an",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", str(dst)],
        f"conform {src.name}",
    )
    return _probe_duration(str(dst))


def narration_offsets(
    page_durs: list[float],
    xfade: float = XFADE_DUR,
    dissolve_at: Optional[set[int]] = None,
) -> list[float]:
    """Start time of each page's spoken line in the stitched timeline.

    Stitching overlaps each pair of clips by the length of their seam, so a page starts
    that much earlier than a naive running sum suggests — and the seams differ now
    (hard cuts are one frame, location dissolves are longer). Mirroring the stitcher's
    per-seam overlap here is what keeps the voice from drifting out of sync page by page.
    Pass `dissolve_at=None` for a timeline joined with a uniform `xfade`.
    """
    hard_cut = 1.0 / OUTPUT_FPS
    offsets: list[float] = []
    running = 0.0
    for i, d in enumerate(page_durs):
        offsets.append(running + NARRATION_LEAD_IN)
        if i < len(page_durs) - 1:
            if dissolve_at is None:
                seam = xfade
            else:
                seam = xfade * 2.5 if (i + 1) in dissolve_at else hard_cut
            running += d - seam
        else:
            running += d
    return offsets


async def mux_narration(video: Path, wavs: list[Path | None], offsets: list[float], out: Path) -> None:
    """Lay each page's narration onto the stitched video at its computed offset."""
    present = [(w, o) for w, o in zip(wavs, offsets) if w and Path(w).exists()]
    if not present:
        # Nothing to mux — just copy the silent video through.
        await _run_ffmpeg(["ffmpeg", "-y", "-i", str(video), "-c", "copy", str(out)], "copy video")
        return

    args = ["ffmpeg", "-y", "-i", str(video)]
    for wav, _ in present:
        args.extend(["-i", str(wav)])

    parts, labels = [], []
    for idx, (_, offset) in enumerate(present, start=1):
        label = f"a{idx}"
        # adelay wants milliseconds, per channel; `all=1` applies it to every channel.
        parts.append(f"[{idx}:a]adelay={int(max(0.0, offset) * 1000)}:all=1[{label}]")
        labels.append(f"[{label}]")
    parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:dropout_transition=0:normalize=0[narr]")

    args.extend([
        "-filter_complex", ";".join(parts),
        "-map", "0:v", "-map", "[narr]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(out),
    ])
    await _run_ffmpeg(args, "mux narration")


async def build_animatic(
    images: list[Path], page_durs: list[float],
    wavs: list[Path | None], out: Path,
) -> None:
    """Assemble keyframe stills + narration into a timed animatic.

    This is the cheap preview of the whole film — it costs seconds and tells you whether
    the story and its pacing work before committing hours to animation.
    """
    if not images:
        raise ValueError("no images for animatic")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for img, dur in zip(images, page_durs):
            fh.write(f"file '{Path(img).as_posix()}'\n")
            fh.write(f"duration {dur:.4f}\n")
        # The concat demuxer ignores the final entry's duration unless the last file is
        # repeated, so repeat it.
        fh.write(f"file '{Path(images[-1]).as_posix()}'\n")
        list_path = fh.name

    silent = out.with_suffix(".silent.mp4")
    try:
        await _run_ffmpeg(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=16",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", str(silent)],
            "animatic stills",
        )
        # No crossfade in the animatic, so offsets are a plain running sum.
        await mux_narration(silent, wavs, narration_offsets(page_durs, xfade=0.0), out)
    finally:
        Path(list_path).unlink(missing_ok=True)
        silent.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Grade: make separately-generated shots look like one piece of film
# ---------------------------------------------------------------------------

def _measure_levels(video: Path) -> Optional[tuple[float, float, float]]:
    """Average Y/U/V of a clip, via ffmpeg's signalstats. None if it can't be read."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", str(video), "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=180,
        )
        y = [float(l.split("=")[-1]) for l in proc.stderr.splitlines() if "signalstats.YAVG" in l]
        if not y:
            return None
        proc2 = subprocess.run(
            ["ffmpeg", "-i", str(video), "-vf",
             "signalstats,metadata=print:key=lavfi.signalstats.UAVG",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=180,
        )
        u = [float(l.split("=")[-1]) for l in proc2.stderr.splitlines() if "signalstats.UAVG" in l]
        proc3 = subprocess.run(
            ["ffmpeg", "-i", str(video), "-vf",
             "signalstats,metadata=print:key=lavfi.signalstats.VAVG",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=180,
        )
        v = [float(l.split("=")[-1]) for l in proc3.stderr.splitlines() if "signalstats.VAVG" in l]
        return (
            sum(y) / len(y),
            sum(u) / len(u) if u else 128.0,
            sum(v) / len(v) if v else 128.0,
        )
    except Exception:
        return None


def plan_grade(levels: list[Optional[tuple[float, float, float]]]) -> list[Optional[str]]:
    """Per-clip ffmpeg filter strings that pull every shot toward the run's median look.

    Deliberately gentle and global (one brightness nudge, one colour-balance nudge per
    clip) rather than a per-frame histogram match: the goal is that shots stop
    disagreeing about exposure and white balance, not to repaint them. Corrections are
    clamped so a genuinely dark scene stays dark.
    """
    valid = [l for l in levels if l]
    if len(valid) < 2:
        return [None] * len(levels)
    med_y = sorted(l[0] for l in valid)[len(valid) // 2]
    med_u = sorted(l[1] for l in valid)[len(valid) // 2]
    med_v = sorted(l[2] for l in valid)[len(valid) // 2]

    out: list[Optional[str]] = []
    for lv in levels:
        if not lv:
            out.append(None)
            continue
        # Y is 0-255; convert the gap to ffmpeg eq's -1..1 brightness scale, then clamp
        # to a change small enough to stay invisible as an effect.
        b = max(-0.08, min(0.08, (med_y - lv[0]) / 255.0))
        # U/V are chroma offsets around 128; nudge blue/red balance the same way.
        bb = max(-0.08, min(0.08, (med_u - lv[1]) / 255.0))
        rb = max(-0.08, min(0.08, (med_v - lv[2]) / 255.0))
        if abs(b) < 0.004 and abs(bb) < 0.004 and abs(rb) < 0.004:
            out.append(None)     # already on the median — don't re-encode for nothing
        else:
            out.append(f"eq=brightness={b:.4f},colorbalance=rm={rb:.4f}:bm={bb:.4f}")
    return out


async def apply_grade(src: Path, dst: Path, filters: str) -> None:
    await _run_ffmpeg(
        ["ffmpeg", "-y", "-i", str(src), "-vf", filters, "-an",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", str(dst)],
        f"grade {src.name}",
    )


async def finish_picture(src: Path, dst: Path, fps: int = OUTPUT_FPS) -> None:
    """Final picture pass over the stitched cut: conform frame rate, add a little grain.

    Grain is the cheapest trick for making diffusion output stop looking plasticky; it
    goes on last so it sits over the whole film evenly instead of per shot.
    """
    vf = [f"fps={fps}"]
    if GRAIN_STRENGTH > 0:
        vf.append(f"noise=alls={GRAIN_STRENGTH}:allf=t+u")
    await _run_ffmpeg(
        ["ffmpeg", "-y", "-i", str(src), "-vf", ",".join(vf), "-an",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-preset", "medium", str(dst)],
        "finish picture",
    )


# ---------------------------------------------------------------------------
# Mix: narration over an optional music bed, ducked and loudness-normalised
# ---------------------------------------------------------------------------

async def mux_mix(
    video: Path,
    wavs: list[Path | None],
    offsets: list[float],
    out: Path,
    music: Optional[Path] = None,
) -> None:
    """Lay narration (and optionally a music bed) onto the cut and normalise loudness.

    With music present the bed is side-chain compressed by the narration, so the score
    steps back under the voice and swells again between lines — the thing that makes a
    mix sound produced rather than layered.
    """
    present = [(w, o) for w, o in zip(wavs, offsets) if w and Path(w).exists()]
    if not present and not music:
        await _run_ffmpeg(["ffmpeg", "-y", "-i", str(video), "-c", "copy", str(out)], "copy video")
        return

    args = ["ffmpeg", "-y", "-i", str(video)]
    for wav, _ in present:
        args.extend(["-i", str(wav)])
    if music:
        args.extend(["-stream_loop", "-1", "-i", str(music)])
        music_idx = len(present) + 1

    parts, labels = [], []
    for idx, (_, offset) in enumerate(present, start=1):
        parts.append(f"[{idx}:a]adelay={int(max(0.0, offset) * 1000)}:all=1[a{idx}]")
        labels.append(f"[a{idx}]")

    if labels:
        parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:dropout_transition=0:normalize=0[narr]")
        voice = "[narr]"
    else:
        voice = None

    if music and voice:
        # Split the voice: one copy is the audible track, one drives the ducking.
        parts.append(f"{voice}asplit=2[narr_out][duck_key]")
        parts.append(f"[{music_idx}:a]volume={MUSIC_GAIN}[bed]")
        parts.append("[bed][duck_key]sidechaincompress="
                     "threshold=0.02:ratio=10:attack=50:release=500[bed_ducked]")
        parts.append("[narr_out][bed_ducked]amix=inputs=2:dropout_transition=0:normalize=0[premix]")
        final_label = "[premix]"
    elif music:
        parts.append(f"[{music_idx}:a]volume={MUSIC_GAIN}[premix]")
        final_label = "[premix]"
    else:
        final_label = voice

    parts.append(f"{final_label}loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}[mix]")

    args.extend([
        "-filter_complex", ";".join(parts),
        "-map", "0:v", "-map", "[mix]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
    ])
    if music:
        # The bed is looped indefinitely, so something has to bound the output; the video
        # is then the shortest stream and sets the length. Without music the narration
        # track ends before the picture does, and -shortest would chop the final beat off
        # the film — so it must NOT be passed in that case.
        args.append("-shortest")
    args.extend(["-movflags", "+faststart", str(out)])
    await _run_ffmpeg(args, "mux mix")
