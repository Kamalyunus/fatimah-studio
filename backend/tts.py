"""Kokoro TTS — local 82M model, runs on CPU so it doesn't fight GPU for Wan/Flux."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

_pipeline = None
_log = logging.getLogger(__name__)

# Default voice — British female storyteller (Beatrix-Potter audiobook vibe).
# Other strong picks:
#   af_bella     — expressive American female (lively, animated)
#   af_aoede     — clear young female (good for cheerful tales)
#   bm_george    — older British male (Stephen-Fry storyteller)
#   bf_emma      — British female (current default — measured, warm)
#   am_michael   — rich American male
DEFAULT_VOICE = "bf_emma"
SAMPLE_RATE = 24000


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code="a", device="cpu")
    return _pipeline


def _synthesize_sync(text: str, voice: str, speed: float) -> Optional[np.ndarray]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        pipeline = _get_pipeline()
        chunks: list[np.ndarray] = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            if audio is None:
                continue
            arr = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
            chunks.append(arr.astype(np.float32))
        if not chunks:
            return None
        return np.concatenate(chunks)
    except Exception as e:
        _log.warning("Kokoro TTS failed for text %r: %s", text[:60], e)
        return None


async def synthesize_to_file(
    text: str,
    output_path: Path,
    voice: str = DEFAULT_VOICE,
    speed: float = 0.9,  # slightly slower than natural for storyteller cadence
) -> float:
    """Synthesise `text`, write WAV to `output_path`. Returns audio duration in seconds (0 if empty)."""
    loop = asyncio.get_event_loop()
    audio = await loop.run_in_executor(None, _synthesize_sync, text, voice, speed)
    if audio is None or audio.size == 0:
        # Write a tiny silence so downstream ffmpeg doesn't choke
        sf.write(str(output_path), np.zeros(SAMPLE_RATE // 10, dtype=np.float32), SAMPLE_RATE)
        return 0.0
    sf.write(str(output_path), audio, SAMPLE_RATE)
    return float(len(audio)) / SAMPLE_RATE


def is_available() -> bool:
    try:
        import kokoro  # noqa: F401
        return True
    except Exception:
        return False
