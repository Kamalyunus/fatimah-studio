"""Take selection: render several cheap draft clips per page, score them, keep one.

This is the single practice that most separates good generative video from mediocre
video — nobody producing decent work ships the first take of every shot. The economics
that make it affordable here: a draft keeps the FULL frame count (so its motion is
representative) but drops to half resolution and a third of the steps, which costs about
a tenth of a final render. Best-of-four therefore adds ~40% to a run rather than 4x.

The caveat worth knowing: a draft shares its seed with the final render but not its step
count or resolution, so it predicts composition and gross motion well and fine detail
poorly. We are choosing between motion arcs, not grading finished frames.
"""
from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import drift
from config import COMFY_OUTPUT

# Frame-to-frame change (mean abs luma difference, 0-255) that reads as lively but
# controlled motion. Below the floor the clip is effectively frozen; above the ceiling
# it is usually morphing, flickering, or the camera has bolted.
_MOTION_FLOOR = 0.8
_MOTION_CEILING = 9.0


@dataclass
class Take:
    index: int
    seed: int
    filename: str
    identity: Optional[float] = None    # CLIP sim, mid-clip frame vs character ref
    end_match: Optional[float] = None   # CLIP sim, last frame vs intended end keyframe
    motion: Optional[float] = None      # mean abs luma delta between sampled frames
    score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index, "seed": self.seed, "filename": self.filename,
            "identity": self.identity, "end_match": self.end_match,
            "motion": self.motion, "score": round(self.score, 4),
            "notes": list(self.notes),
        }


def _extract_frame(video: Path, out: Path, position: str) -> bool:
    """Pull one frame out of a clip. position is 'mid' or 'last'."""
    try:
        if position == "last":
            args = ["ffmpeg", "-y", "-sseof", "-0.2", "-i", str(video),
                    "-update", "1", "-q:v", "3", str(out)]
        else:
            args = ["ffmpeg", "-y", "-i", str(video), "-vf", "select='eq(n\\,0)+gte(t\\,0.5*duration)'",
                    "-frames:v", "1", "-update", "1", "-q:v", "3", str(out)]
        subprocess.run(args, check=True, capture_output=True, timeout=60)
        return out.exists()
    except Exception:
        return False


def _mean_frame_delta(video: Path) -> Optional[float]:
    """Average absolute luma change between consecutive frames.

    Uses ffmpeg's `signalstats` on a temporally-differenced stream: a frozen clip sits
    near zero, a chaotic or morphing one runs high. Cheap proxy for "is this clip
    actually animating, and is it animating sanely".
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", str(video), "-vf", "tblend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=180,
        )
        vals = [
            float(line.split("=")[-1])
            for line in proc.stderr.splitlines()
            if "lavfi.signalstats.YAVG" in line
        ]
        if not vals:
            return None
        # Drop the first entry: the first differenced frame is against a black frame.
        vals = vals[1:] or vals
        return sum(vals) / len(vals)
    except Exception:
        return None


async def score_takes(
    takes: list[Take],
    character_ref: Optional[Path],
    end_keyframe: Optional[Path],
    work_dir: Path,
) -> None:
    """Attach identity / end-match / motion measurements and a combined score, in place."""
    work_dir.mkdir(parents=True, exist_ok=True)
    mid_frames: dict[int, Path] = {}
    last_frames: dict[int, Path] = {}

    for t in takes:
        video = COMFY_OUTPUT / t.filename
        mid = work_dir / f"take{t.index}_mid.jpg"
        last = work_dir / f"take{t.index}_last.jpg"
        if _extract_frame(video, mid, "mid"):
            mid_frames[t.index] = mid
        if _extract_frame(video, last, "last"):
            last_frames[t.index] = last
        t.motion = await asyncio.to_thread(_mean_frame_delta, video)

    # CLIP passes: one batch per reference so the model loads once.
    if character_ref and character_ref.exists() and mid_frames:
        idxs = list(mid_frames)
        sims = await drift.score_drift(character_ref, [mid_frames[i] for i in idxs])
        for i, s in zip(idxs, sims):
            next(t for t in takes if t.index == i).identity = s
    if end_keyframe and end_keyframe.exists() and last_frames:
        idxs = list(last_frames)
        sims = await drift.score_drift(end_keyframe, [last_frames[i] for i in idxs])
        for i, s in zip(idxs, sims):
            next(t for t in takes if t.index == i).end_match = s

    for t in takes:
        t.score, t.notes = _combine(t)


def _combine(t: Take) -> tuple[float, list[str]]:
    """Fold the measurements into one comparable number.

    Identity carries the most weight (a drifted character is the failure users notice
    first), then landing the intended end pose, then a motion sanity band. Missing
    measurements fall back to a neutral value rather than disqualifying a take.
    """
    notes: list[str] = []
    identity = t.identity if t.identity is not None else 0.85
    end_match = t.end_match if t.end_match is not None else 0.85

    if t.motion is None:
        motion_score = 0.6
    elif t.motion < _MOTION_FLOOR:
        motion_score = 0.15
        notes.append(f"barely moves ({t.motion:.2f})")
    elif t.motion > _MOTION_CEILING:
        motion_score = 0.15
        notes.append(f"unstable motion ({t.motion:.2f})")
    else:
        # Peak in the middle of the healthy band, tapering toward both edges.
        mid = (_MOTION_FLOOR + _MOTION_CEILING) / 2
        motion_score = 1.0 - abs(t.motion - mid) / (mid - _MOTION_FLOOR)

    if t.identity is not None and t.identity < drift.DRIFT_THRESHOLD:
        notes.append(f"character drift ({t.identity:.2f})")

    score = 0.50 * identity + 0.25 * end_match + 0.25 * motion_score
    return score, notes


def pick_best(takes: list[Take]) -> Take:
    """Highest score wins; ties fall to the earlier (lower-seed) take for determinism."""
    return max(takes, key=lambda t: (t.score, -t.index))
