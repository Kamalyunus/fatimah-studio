"""Local text-to-speech for storybook narration (Kokoro-82M).

Runs on CPU on purpose: the model is tiny (~2s of compute per page) and the GPUs are
busy with diffusion for hours at a time. Narration is rendered BEFORE any video work
because each page's spoken length is what sets that page's clip duration — the picture
is cut to the voice, the way an animatic is.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from config import TTS_LANG, TTS_SAMPLE_RATE, TTS_SPEED, TTS_VOICE

_pipeline = None


def _get_pipeline():
    """Lazy singleton. Importing kokoro pulls in torch bits, so defer until first use."""
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code=TTS_LANG, repo_id="hexgrad/Kokoro-82M", device="cpu")
    return _pipeline


def clean_narration(text: str) -> str:
    """Strip anything the LLM leaked that shouldn't be spoken aloud."""
    t = (text or "").strip()
    t = re.sub(r"^(narration|page \d+|scene \d+)\s*[:\-–]\s*", "", t, flags=re.I)
    t = t.strip().strip('"').strip()
    t = re.sub(r"\s+", " ", t)
    return t


def synth(text: str, out_path: Path, voice: Optional[str] = None) -> float:
    """Render `text` to a mono WAV at out_path. Returns the duration in seconds.

    Returns 0.0 (and writes nothing) for empty text, so callers can treat a page with
    no narration as "use the default clip length".
    """
    text = clean_narration(text)
    if not text:
        return 0.0
    pipeline = _get_pipeline()
    chunks = [audio for _, _, audio in pipeline(text, voice=voice or TTS_VOICE, speed=TTS_SPEED)]
    if not chunks:
        return 0.0
    audio = np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, TTS_SAMPLE_RATE)
    return float(len(audio)) / TTS_SAMPLE_RATE
