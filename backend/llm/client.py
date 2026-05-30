"""Ollama transport: chat completion, VRAM unload, availability check + model config."""
from __future__ import annotations

import json
from typing import Any
import re

import httpx

OLLAMA_URL = "http://127.0.0.1:11434"
# Single LLM for everything (prompt rewriting + storybook planning). qwen3.6 is heavier
# but writes noticeably better prompts; the LLM unloads from VRAM before diffusion so
# the size doesn't fight Wan/Flux at gen time.
LLM_MODEL = "qwen3.6:latest"


async def _chat(system: str, user: str, json_mode: bool = False, timeout: float = 120.0, max_tokens: int = 2048) -> str:
    """Call ollama /api/chat with a system + user message; return assistant content."""
    is_qwen3 = "qwen3" in LLM_MODEL.lower()
    sys_full = f"{system.rstrip()} /no_think" if is_qwen3 else system
    payload: dict[str, Any] = {
        "model": LLM_MODEL,
        "stream": False,
        "keep_alive": 0,
        "messages": [
            {"role": "system", "content": sys_full},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0.5 if json_mode else 0.7, "num_predict": max_tokens},
    }
    if is_qwen3:
        payload["think"] = False
    if json_mode:
        payload["format"] = "json"
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    content = (data.get("message") or {}).get("content", "").strip()
    import re
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
    return content


async def unload() -> None:
    """Force-unload every currently loaded model from VRAM. Safe to call when nothing is loaded.

    Uses /api/chat with keep_alive:0 — /api/generate was observed to not reliably evict
    qwen3.6 on this setup. We also walk /api/ps so a leftover model from a different
    name (e.g. an older small model) still gets cleared instead of holding 20+ GB.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            ps = await c.get(f"{OLLAMA_URL}/api/ps")
            loaded = [m.get("name") for m in (ps.json().get("models") or []) if m.get("name")]
            # Always include the configured model — covers the case where /api/ps is empty
            # but a stale instance is still resident.
            for name in set(loaded + [LLM_MODEL]):
                await c.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={"model": name, "messages": [], "keep_alive": 0},
                )
    except Exception:
        pass  # don't fail the caller if ollama is down




async def is_available() -> bool:
    """Quick check if Ollama + the model are reachable."""
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            data = r.json()
        names = {m.get("name") for m in (data.get("models") or [])}
        return any(n and n.startswith(LLM_MODEL.split(":")[0]) for n in names)
    except Exception:
        return False
