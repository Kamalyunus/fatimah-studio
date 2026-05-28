"""CLIP-based character-drift detection for the storybook keyframe gate.

Runs a small CLIP ViT-B/32 model on CPU (so it doesn't fight Wan for GPU memory)
to score how well each scene's start frame matches the canonical character
reference image. Used to auto-flag scenes whose character has drifted.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)

# Lazy-loaded singletons
_model = None
_processor = None
_loaded = False
_load_lock = asyncio.Lock()

# openai/clip-vit-base-patch32 is ~150 MB and runs on CPU in ~50 ms / image after warm-up
_MODEL_ID = "openai/clip-vit-base-patch32"

# Below this cosine similarity, the start frame is considered drifted from the ref.
# Tuned empirically — most "looks identical" pairs sit at 0.92+, drift starts around 0.85.
DRIFT_THRESHOLD = 0.82


async def _ensure_loaded():
    global _model, _processor, _loaded
    if _loaded:
        return
    async with _load_lock:
        if _loaded:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
            loop = asyncio.get_event_loop()

            def _load():
                m = CLIPModel.from_pretrained(_MODEL_ID).eval()
                p = CLIPProcessor.from_pretrained(_MODEL_ID)
                return m, p

            _model, _processor = await loop.run_in_executor(None, _load)
            _loaded = True
            _log.info("drift: CLIP model loaded")
        except Exception as e:
            _log.warning("drift: failed to load CLIP model: %s", e)
            _loaded = False


def _embed_sync(image_paths: list[str]) -> Optional[np.ndarray]:
    """Synchronously embed a batch of images. Returns L2-normalised embeddings
    as a [N, D] numpy array, or None if anything goes wrong."""
    if _model is None or _processor is None:
        return None
    try:
        from PIL import Image
        import torch
        images = []
        for p in image_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                return None
        with torch.no_grad():
            inputs = _processor(images=images, return_tensors="pt")
            feats = _model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy()
    except Exception as e:
        _log.warning("drift: embedding failed: %s", e)
        return None


async def score_drift(reference_path: Path, scene_paths: list[Path]) -> list[Optional[float]]:
    """Return a list of cosine similarities between `reference_path` and each entry in
    `scene_paths`. Higher = more similar; values near 1.0 mean essentially identical
    character / outfit. Returns None for any path that failed to embed (so the UI can
    skip the badge instead of mis-flagging)."""
    await _ensure_loaded()
    if not _loaded:
        return [None] * len(scene_paths)
    paths = [str(reference_path)] + [str(p) for p in scene_paths]
    loop = asyncio.get_event_loop()
    embeds = await loop.run_in_executor(None, _embed_sync, paths)
    if embeds is None or embeds.shape[0] < 2:
        return [None] * len(scene_paths)
    ref = embeds[0]
    sims = embeds[1:] @ ref   # cosine sim since both are L2-normalised
    return [float(s) for s in sims]
