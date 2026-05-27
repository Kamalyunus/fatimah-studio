"""Local LLM helpers (Ollama) for prompt improvement + story decomposition."""
import json
from typing import Any, Optional

import httpx

OLLAMA_URL = "http://127.0.0.1:11434"
# Two models picked per task:
#  - qwen3:8b for "Improve prompt" UI button (fast cold-load, snappy UX)
#  - qwen3.6:latest (23 GB) for storybook planning where reasoning quality matters
#    (LLM unloads from VRAM before Wan starts, so size doesn't contend)
LLM_MODEL = "qwen3:8b"
LLM_MODEL_JSON = "qwen3.6:latest"

_IMPROVE_SYSTEM = (
    "You are a prompt engineer for AI image and video generators. "
    "Given a short user prompt, rewrite it as a single rich descriptive sentence (under 60 words) "
    "with concrete visual details: lighting, camera angle, mood, setting. "
    "Do NOT include text in quotes, do NOT add commentary — just the rewritten prompt itself."
)

_STORY_SYSTEM = (
    "You are a children's book editor and storyboard cinematographer. You will receive a "
    "short story idea and turn it into a structured plan for an animated illustrated picture "
    "book that will be read aloud. You think like a film editor — each scene flows naturally "
    "into the next, with the character's body position carrying continuity between pages.\n\n"
    "Return STRICT JSON with this shape (no markdown, no commentary):\n"
    "{\n"
    "  \"character\": \"<two-sentence description: name, species, colors, distinctive features, clothing, mood. Detailed visual canon reused for every page.>\",\n"
    "  \"scenes\": [\n"
    "    {\n"
    "      \"starting_pose\": \"<concise description of the character's body position + expression at the FIRST frame of this scene. Examples: 'standing by the window, hand on the glass, looking out', 'sitting on a log, eyes closed, smiling'. THIS MUST MATCH the previous scene's `ending_pose` word-for-word (continuity). For scene 1 this sets the opening pose.>\",\n"
    "      \"ending_pose\": \"<concise description of the character's body position + expression at the LAST frame of this scene. Same vocabulary style as starting_pose. This becomes the next scene's starting_pose.>\",\n"
    "      \"description\": \"<one-sentence visual description of the still image, MUST include the starting_pose so Flux paints the character in that exact pose. Composition, setting. Reference character by name.>\",\n"
    "      \"motion\": \"<short: what the character does. Must take character from starting_pose to ending_pose in one continuous arc.>\",\n"
    "      \"video_prompt\": \"<DETAILED cinematic video direction (40-80 words). Must include: (1) shot framing (close-up / medium / wide / overhead), (2) lighting and mood, (3) motion arc starting from the starting_pose and finishing at the ending_pose with manner (slowly, gently, eagerly), (4) one sensory environmental detail (leaves drifting, steam rising, dust motes), (5) explicit closing pose. Smooth storybook-style motion. The motion completes by the prompt's end.>\",\n"
    "      \"narration\": \"<two-to-three sentences of read-aloud narration. 25-45 words. Sensory detail, varied pacing, occasional dialogue/question/exclamation. Sounds like a warm, expressive parent reading aloud.>\"\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Generate EXACTLY the number of scenes requested.\n\n"
    "Critical rules:\n"
    "1. CONTINUITY: scene[i].starting_pose MUST equal scene[i-1].ending_pose word-for-word. Plan all `ending_pose` and `starting_pose` first so they chain.\n"
    "2. The image (`description`) AND the video (`video_prompt`) must both reflect the starting_pose, so the first frame of the animation matches the previous page's last frame.\n"
    "3. Every video_prompt traverses from starting_pose to ending_pose — no mid-action cliffhangers.\n"
    "4. Narration feels alive: vary length, add dialogue/exclamations, sensory pictures.\n"
    "5. Each page ≈ 5 seconds of viewing; narration must fit at storyteller pace.\n"
    "6. One primary action per scene. Child-friendly. Slow/subtle camera moves."
)


async def _chat(system: str, user: str, json_mode: bool = False, timeout: float = 120.0, max_tokens: int = 2048, model: Optional[str] = None) -> str:
    """Call ollama /api/chat with a system + user message; return assistant content.

    `model` defaults to LLM_MODEL (qwen3:8b for prose). Pass LLM_MODEL_JSON for structured output.
    """
    model_name = model or LLM_MODEL
    is_qwen3 = "qwen3" in model_name.lower()
    sys_full = f"{system.rstrip()} /no_think" if is_qwen3 else system
    payload: dict[str, Any] = {
        "model": model_name,
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
    """Force-unload any loaded LLM from VRAM. Safe to call when nothing is loaded."""
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            await c.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": LLM_MODEL, "keep_alive": 0},
            )
    except Exception:
        pass  # don't fail the caller if ollama is down


async def improve_prompt(short: str) -> str:
    """Take a short user prompt and rewrite it as a richer one."""
    short = short.strip()
    if not short:
        return short
    out = await _chat(_IMPROVE_SYSTEM, short)
    # Trim quotes / common artefacts
    out = out.strip().strip("\"'")
    # Sanity fallback: if LLM returned something tiny, keep original
    if len(out) < len(short) * 0.7:
        return short
    return out


async def plan_storybook(story: str, n_pages: int, style: str) -> dict:
    """Return {character, scenes:[{description, motion, narration}]} of length n_pages.

    Retries once if the LLM returns fewer scenes than requested or any scene has empty fields.
    """
    user = (
        f"Story idea: {story.strip()}\n\n"
        f"Number of scenes: EXACTLY {n_pages} (no more, no fewer)\n"
        f"Illustration style: {style}\n\n"
        f"Every scene MUST include all SIX fields: starting_pose, ending_pose, description, "
        f"motion, video_prompt, narration. The starting_pose of each scene (after scene 1) "
        f"MUST match the previous scene's ending_pose exactly. "
        f"Reference the character by name in every scene. Return STRICT JSON only."
    )
    plan = await _request_plan(user, n_pages, max_tokens=2048 + 256 * max(0, n_pages - 6))

    # Validate & repair empties / count mismatches
    needs_retry = False
    if not isinstance(plan, dict) or "scenes" not in plan:
        needs_retry = True
    else:
        scenes = plan.get("scenes") or []
        if len(scenes) != n_pages:
            needs_retry = True
        for s in scenes:
            if not isinstance(s, dict):
                needs_retry = True; break
            if not (s.get("description") or "").strip():
                needs_retry = True; break
            if not (s.get("video_prompt") or "").strip():
                needs_retry = True; break
            if not (s.get("narration") or "").strip():
                needs_retry = True; break
            if not (s.get("ending_pose") or "").strip():
                needs_retry = True; break

    if needs_retry:
        # One retry, more explicit, lower temperature
        retry_user = user + (
            "\n\nIMPORTANT: your previous response was incomplete. "
            f"Output EXACTLY {n_pages} scenes. Every scene MUST have non-empty "
            "`description`, `motion`, `video_prompt`, AND `narration` fields. Do not skip any field."
        )
        plan = await _request_plan(retry_user, n_pages, max_tokens=4096)

    plan["scenes"] = (plan.get("scenes") or [])[:n_pages]
    # Pad if still short — use the story as fallback so it's at least non-empty
    while len(plan["scenes"]) < n_pages:
        idx = len(plan["scenes"]) + 1
        char = plan.get("character", "the character").split(",")[0]
        plan["scenes"].append({
            "description": f"Another moment in the story of {char}.",
            "motion": "gentle, subtle movement",
            "video_prompt": f"A soft, warm storybook scene featuring {char}. Slow, gentle motion. Cinematic lighting.",
            "narration": story.strip()[:120] or "And the story continues...",
        })
    # Fill any individual empty fields
    fallback = {
        "starting_pose": "in a calm, settled stance",
        "ending_pose": "in a calm, settled stance",
        "description": "A quiet moment in the story.",
        "motion": "subtle motion",
        "video_prompt": "A soft warm storybook scene. Gentle, slow motion. Cinematic lighting that settles peacefully.",
        "narration": "The story unfolds.",
    }
    for i, s in enumerate(plan["scenes"]):
        if not isinstance(s, dict):
            plan["scenes"][i] = dict(fallback)
            continue
        for k, v in fallback.items():
            if not (s.get(k) or "").strip():
                s[k] = v

    # Enforce pose-chain continuity: each scene's starting_pose := previous ending_pose.
    # The LLM is asked to do this but we make it crisp here.
    for i in range(1, len(plan["scenes"])):
        prev_end = (plan["scenes"][i - 1].get("ending_pose") or "").strip()
        if prev_end:
            plan["scenes"][i]["starting_pose"] = prev_end
    return plan


async def _request_plan(user: str, n_pages: int, max_tokens: int) -> dict:
    # Use the JSON-tuned model (qwen2.5:3b) for structured output
    raw = await _chat(_STORY_SYSTEM, user, json_mode=True, timeout=180.0, max_tokens=max_tokens, model=LLM_MODEL_JSON)
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        if s >= 0 and e > s:
            try:
                plan = json.loads(raw[s : e + 1])
            except json.JSONDecodeError:
                return {"scenes": []}
        else:
            return {"scenes": []}
    # Repair any scene-shaped dicts that got generated with an empty-string key
    # (qwen3 sometimes produces {"": "...", "motion": "..."}).
    scenes = plan.get("scenes") or []
    for s in scenes:
        if isinstance(s, dict) and "" in s and "description" not in s:
            s["description"] = s.pop("") or s.get("description", "")
    plan["scenes"] = scenes
    return plan


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
