"""Local LLM helpers (Ollama) for prompt improvement + story decomposition."""
import json
from typing import Any, Optional

import httpx

OLLAMA_URL = "http://127.0.0.1:11434"
# Single LLM for everything (prompt rewriting + storybook planning). qwen3.6 is heavier
# but writes noticeably better prompts; the LLM unloads from VRAM before diffusion so
# the size doesn't fight Wan/Flux at gen time.
LLM_MODEL = "qwen3.6:latest"

_IMPROVE_SYSTEM = (
    "You are a senior prompt engineer for AI image generators (Flux, SDXL). "
    "Your job: take a short user idea and rewrite it as a single comprehensive prompt "
    "(70-90 words, one sentence preferred) that a diffusion model can render faithfully.\n\n"
    "Include, in roughly this order:\n"
    "  1. SUBJECT — concrete specifics: who/what, age/breed/material, distinguishing features, "
    "expression or posture, clothing/texture.\n"
    "  2. SETTING — where it is, time of day or season, two sensory environmental details "
    "(e.g. drifting pollen, wet cobblestones, neon haze).\n"
    "  3. COMPOSITION — shot framing (close-up / medium / wide / overhead), camera angle, "
    "and where the subject sits in frame.\n"
    "  4. LIGHTING — direction and quality (golden hour back-lighting, soft north window, "
    "harsh noon, etc.). Lighting drives mood; be specific.\n"
    "  5. ATMOSPHERE — one short mood phrase (serene, melancholic, electric, intimate).\n"
    "  6. TECHNICAL FINISH — sharpness / detail level / depth of field, plus a color palette cue.\n\n"
    "Rules:\n"
    "  - Preserve every concrete detail the user gave; never drop or contradict them.\n"
    "  - Use vivid concrete nouns; avoid generic adjectives like 'beautiful', 'amazing', 'epic'.\n"
    "  - No quotes around the output, no preamble, no bullet points, no commentary — just the "
    "    rewritten prompt itself, ready to feed to a diffusion model."
)

# Style guidance — applied on top of _IMPROVE_SYSTEM when the user picks a style chip.
# Keys are the lowercase labels the frontend sends.
_STYLE_HINTS = {
    "cinematic":
        "Style: CINEMATIC. Frame it as a film still — dramatic lighting, anamorphic feel, "
        "shallow depth of field, slight film grain. Lean into mood and atmosphere.",
    "photorealistic":
        "Style: PHOTOREALISTIC. Frame it as professional photography — sharp focus, high "
        "dynamic range, natural lighting, accurate colors, fine surface detail.",
    "anime":
        "Style: ANIME. Frame it in the style of Studio Ghibli / classic Japanese animation — "
        "hand-painted backgrounds, vibrant colors, soft cel shading, gentle mood.",
    "painting":
        "Style: OIL PAINTING. Frame it as a traditional oil painting — visible brushstrokes, "
        "rich pigments, painterly texture, classical composition.",
    "pencil sketch":
        "Style: PENCIL SKETCH. Frame it as a detailed graphite drawing — careful hatching, "
        "tonal shading, paper texture, monochrome.",
}

_STORY_SYSTEM = (
    "You are a children's book editor and storyboard cinematographer. You will receive a "
    "short story idea and turn it into a structured plan for an animated illustrated picture "
    "book that will be read aloud. You think like a film editor — each scene flows naturally "
    "into the next, with the character's body position carrying continuity between pages.\n\n"
    "Return STRICT JSON with this shape (no markdown, no commentary):\n"
    "{\n"
    "  \"character\": \"<two-sentence description: name, species, colors, distinctive features, clothing, mood. Detailed visual canon reused for every page.>\",\n"
    "  \"character_canon\": {\n"
    "    \"name\":      \"<character's name>\",\n"
    "    \"species\":   \"<species or kind: 'a small grey kitten', 'a friendly toy robot', 'a 6-year-old girl with curly red hair'>\",\n"
    "    \"colors\":    \"<key colors: 'soft grey fur with white paws and a pink nose'>\",\n"
    "    \"features\":  \"<distinctive features: 'big amber eyes, small white star on forehead, chipped left ear'>\",\n"
    "    \"clothing\":  \"<exact outfit if any: 'red wool sweater with a yellow star, blue overalls, no shoes'. Use empty string if none.>\",\n"
    "    \"accessories\": \"<persistent objects the character carries: 'a tiny leather satchel, a wooden flute'. Empty string if none.>\"\n"
    "  },\n"
    "  \"scenes\": [\n"
    "    {\n"
    "      \"starting_pose\": \"<concise description of the character's body position + expression at the FIRST frame of this scene. Examples: 'standing by the window, hand on the glass, looking out', 'sitting on a log, eyes closed, smiling'. THIS MUST MATCH the previous scene's `ending_pose` word-for-word (continuity). For scene 1 this sets the opening pose.>\",\n"
    "      \"ending_pose\": \"<concise description of the character's body position + expression at the LAST frame of this scene. Same vocabulary style as starting_pose. This becomes the next scene's starting_pose.>\",\n"
    "      \"description\": \"<one-sentence visual description of the still image, MUST include the starting_pose so Flux paints the character in that exact pose. Composition, setting. Reference character by name.>\",\n"
    "      \"motion\": \"<short: what the character does. Must take character from starting_pose to ending_pose in one continuous arc.>\",\n"
    "      \"video_prompt\": \"<CONCISE cinematic direction for ONE simple motion (25-45 words). Name: (1) shot framing (close-up / medium / wide / overhead), (2) one short motion arc from starting_pose to ending_pose with manner (slowly, gently, eagerly), (3) one ambient detail (leaves drifting, steam rising). Avoid stacking multiple actions. The motion is gentle and completes within ~5 seconds.>\",\n"
    "      \"motion_intensity\": \"<one of: 'still' (sitting, blinking, breathing), 'gentle' (waving, turning head, reaching), 'dynamic' (jumping, running steps, throwing). Default to 'gentle' if unsure.>\"\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Generate EXACTLY the number of scenes requested.\n\n"
    "Critical rules:\n"
    "1. POSE CONTINUITY: scene[i].starting_pose MUST equal scene[i-1].ending_pose word-for-word. Plan all `ending_pose` and `starting_pose` first so they chain.\n"
    "2. SETTING CONTINUITY: consecutive scenes MUST stay in the same physical setting (same room, same hill, same beach) unless a dedicated transit scene moves the character. Each scene's `description` should keep the environment from the previous scene — the camera and lighting can change, but the place does not jump.\n"
    "   - BAD: scene 3 is 'in the kitchen', scene 4 is 'in the meadow'. Background teleport → rushed feel.\n"
    "   - GOOD: scene 3 ends with 'opening the kitchen door, looking out at the meadow'. Scene 4 begins 'stepping onto the meadow grass'. The transit IS its own scene.\n"
    "   - Plan setting changes as deliberate transit beats: 'opens door', 'walks through gate', 'climbs into the spaceship'. Never skip them.\n"
    "3. The image (`description`) AND the video (`video_prompt`) must both reflect the starting_pose, so the first frame of the animation matches the previous page's last frame.\n"
    "4. ONE primary action per scene. Resist packing multiple beats into one scene — if a moment has steps (e.g. 'open the door AND step out AND look up'), that is THREE scenes, not one. Use simple verbs.\n"
    "5. The pose change from starting_pose to ending_pose should be small enough to render in ~5 seconds of natural motion (a tilt of the head, a hand reaching out, taking two steps). Avoid full-body locomotion arcs within one scene.\n"
    "6. Every video_prompt traverses from starting_pose to ending_pose — no mid-action cliffhangers.\n"
    "7. Child-friendly. Slow/subtle camera moves only (or none)."
)

_OUTLINE_SYSTEM = (
    "You are a children's story editor planning the narrative SKELETON of a short illustrated "
    "video before any scene-level writing. You return a tight outline with story beats and "
    "the number of scenes each beat will use.\n\n"
    "Return STRICT JSON with this shape (no markdown, no commentary):\n"
    "{\n"
    "  \"title\": \"<short title>\",\n"
    "  \"character\": \"<two-sentence character description — same canon used later>\",\n"
    "  \"beats\": [\n"
    "    {\"name\": \"setup\",      \"summary\": \"<who, where, what's normal>\",          \"scenes\": <int>},\n"
    "    {\"name\": \"inciting\",   \"summary\": \"<what disrupts the normal>\",            \"scenes\": <int>},\n"
    "    {\"name\": \"rising\",     \"summary\": \"<the main attempt or journey>\",         \"scenes\": <int>},\n"
    "    {\"name\": \"climax\",     \"summary\": \"<the key moment of change>\",            \"scenes\": <int>},\n"
    "    {\"name\": \"resolution\", \"summary\": \"<how things settle, what the character feels at the end>\", \"scenes\": <int>}\n"
    "  ]\n"
    "}\n\n"
    "Constraints:\n"
    "- Total of `scenes` across all beats must EQUAL the requested number of scenes exactly.\n"
    "- Allocate scenes proportionally — setup and resolution often need only 1 scene each; "
    "rising-action gets the most. Climax is usually 1-2 scenes.\n"
    "- Each beat lives in ONE primary setting; transit between settings happens within "
    "rising-action and counts as scenes of that beat.\n"
    "- Child-friendly, gentle pacing. No conflict beyond mild challenge."
)

_CRITIQUE_SYSTEM = (
    "You are a senior storyboard editor reviewing a draft children's animated picture-book plan. "
    "Your job: read the draft, find violations of the rules below, and return a REVISED plan "
    "that fixes them. If a scene already obeys the rules, keep it unchanged.\n\n"
    "Return STRICT JSON in the EXACT same shape as the input: "
    "{character, scenes:[{starting_pose, ending_pose, description, motion, video_prompt}]}\n"
    "No markdown, no commentary, no notes about what you changed — just the corrected JSON.\n\n"
    "Hard constraints:\n"
    "- Scene COUNT must stay identical to the draft. Do not add or remove scenes; rewrite in place.\n"
    "- Keep the character's canonical description intact.\n\n"
    "Things to check and fix:\n"
    "1. SETTING TELEPORTS: do consecutive scenes share the same physical setting? If scene N is in "
    "   place A and scene N+1 is in place B with no transit, REWRITE one of them so they're in "
    "   the same place. Place transit beats explicitly ('opens the door', 'climbs onto the ship').\n"
    "2. PACKED SCENES: does any scene have multiple actions joined by AND / THEN? Simplify to a "
    "   single primary action; move the dropped action to a neighboring scene if it fits.\n"
    "3. POSE CHAIN: does scene[i].starting_pose match scene[i-1].ending_pose word-for-word? If not, "
    "   align them.\n"
    "4. STORY ARC: does the sequence have a clear beginning (setup), middle (rising action), and "
    "   end (resolution)? If a scene is filler or out of order, rewrite it to advance the arc.\n"
    "5. CHARACTER CONSISTENCY: is the character described or referenced the same way throughout?\n"
    "6. ONE-MOTION RULE: each video_prompt should describe ONE simple motion completable in ~5s.\n\n"
    "If the draft is already clean, return it essentially unchanged."
)


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


async def improve_prompt(short: str, style: Optional[str] = None) -> str:
    """Rewrite a short user prompt into a richer one. If `style` matches a known
    chip (e.g. "cinematic"), the LLM is also nudged to lean into that visual style."""
    short = short.strip()
    if not short:
        return short
    system = _IMPROVE_SYSTEM
    hint = _STYLE_HINTS.get((style or "").strip().lower())
    if hint:
        system = f"{_IMPROVE_SYSTEM}\n\n{hint}"
    out = await _chat(system, short)
    out = out.strip().strip("\"'")
    if len(out) < len(short) * 0.7:
        return short
    return out


def render_canon(canon: dict | None) -> str:
    """Render the character canon dict as a single descriptive clause used verbatim in
    every Flux prompt. Empty/missing parts are skipped so we don't leak placeholder text."""
    if not isinstance(canon, dict):
        return ""
    parts: list[str] = []
    for key in ("species", "colors", "features", "clothing", "accessories"):
        v = (canon.get(key) or "").strip()
        if v:
            parts.append(v)
    if not parts:
        return ""
    name = (canon.get("name") or "").strip()
    body = ", ".join(parts)
    return f"{name} ({body})" if name else body


async def plan_storybook(
    story: str,
    n_pages: int,
    style: str,
    existing_canon: Optional[dict] = None,
    existing_character: Optional[str] = None,
) -> dict:
    """Return {character, character_canon, scenes:[{...}]} of length n_pages.

    Two-pass planning:
      1. Outline beat structure (setup → climax → resolution + scene counts per beat).
      2. Expand into per-scene plan, anchored on the outline so the arc is intentional.
    Then run a critique pass to fix setting teleports, packed scenes, broken pose chains.

    If `existing_canon` is provided (re-using a saved character from the library), the
    planner is told to lock that character verbatim instead of inventing a new one.
    """
    # Pass 1: outline
    outline = await _request_outline(story, n_pages, style)
    outline_hint = ""
    if outline.get("beats"):
        outline_hint = (
            "\n\nUse the following narrative outline (already balanced to the right scene count) "
            "to anchor the expansion. Each beat lists how many scenes belong to it:\n"
            + json.dumps(outline, ensure_ascii=False, indent=2)
        )

    locked_clause = ""
    if existing_canon or existing_character:
        canon_json = json.dumps(existing_canon or {}, ensure_ascii=False)
        prose = (existing_character or "").strip()
        locked_clause = (
            "\n\nLOCKED CHARACTER: a saved character is being re-used. Do NOT invent a new "
            "protagonist. Your `character_canon` and `character` fields MUST be set to these "
            "exact values (copy them verbatim into the plan) and every scene must reference "
            "this character.\n"
            f"  character_canon = {canon_json}\n"
            f"  character (prose) = \"{prose}\"\n"
        )

    user = (
        f"Story idea: {story.strip()}\n\n"
        f"Number of scenes: EXACTLY {n_pages} (no more, no fewer)\n"
        f"Illustration style: {style}\n\n"
        f"Every scene MUST include: starting_pose, ending_pose, description, motion, "
        f"video_prompt, motion_intensity. The character_canon object is REQUIRED and its "
        f"name/species/colors/features fields are mandatory (clothing/accessories may be empty). "
        f"The starting_pose of each scene (after scene 1) MUST match the previous scene's "
        f"ending_pose exactly. Reference the character by name in every scene. Return STRICT JSON only."
        + outline_hint
        + locked_clause
    )
    plan = await _request_plan(user, n_pages, max_tokens=3072 + 256 * max(0, n_pages - 6))

    # If a character is locked, force-overwrite whatever the LLM returned in case it drifted.
    if existing_canon:
        plan["character_canon"] = existing_canon
    if existing_character:
        plan["character"] = existing_character

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
            if not (s.get("ending_pose") or "").strip():
                needs_retry = True; break

    if needs_retry:
        # One retry, more explicit, lower temperature
        retry_user = user + (
            "\n\nIMPORTANT: your previous response was incomplete. "
            f"Output EXACTLY {n_pages} scenes. Every scene MUST have non-empty "
            "`description`, `motion`, `video_prompt`, `starting_pose`, `ending_pose` fields. "
            "Do not skip any field."
        )
        plan = await _request_plan(retry_user, n_pages, max_tokens=4096)

    # Self-critique pass: ask the model to review its own draft for setting teleports,
    # packed scenes, broken pose chains, and arc problems, then return a fixed plan.
    # The model is already warm in VRAM so this is cheap (~20-30s).
    plan = await _critique_plan(story, plan, n_pages, style)

    plan["scenes"] = (plan.get("scenes") or [])[:n_pages]
    # Pad if still short — use the story as fallback so it's at least non-empty
    while len(plan["scenes"]) < n_pages:
        char = plan.get("character", "the character").split(",")[0]
        plan["scenes"].append({
            "description": f"Another moment in the story of {char}.",
            "motion": "gentle, subtle movement",
            "video_prompt": f"A soft, warm storybook scene featuring {char}. Slow, gentle motion. Cinematic lighting.",
            "motion_intensity": "gentle",
        })
    # Fill any individual empty fields
    fallback = {
        "starting_pose": "in a calm, settled stance",
        "ending_pose": "in a calm, settled stance",
        "description": "A quiet moment in the story.",
        "motion": "subtle motion",
        "video_prompt": "A soft warm storybook scene. Gentle, slow motion. Cinematic lighting that settles peacefully.",
        "motion_intensity": "gentle",
    }
    valid_intensities = {"still", "gentle", "dynamic"}
    for i, s in enumerate(plan["scenes"]):
        if not isinstance(s, dict):
            plan["scenes"][i] = dict(fallback)
            continue
        for k, v in fallback.items():
            if not (s.get(k) or "").strip():
                s[k] = v
        # Normalise motion_intensity onto the small allowed set
        mi = (s.get("motion_intensity") or "gentle").strip().lower()
        s["motion_intensity"] = mi if mi in valid_intensities else "gentle"

    # Ensure character_canon at least exists (orchestrator falls back to plan["character"]).
    canon = plan.get("character_canon")
    if not isinstance(canon, dict):
        plan["character_canon"] = {}

    # Enforce pose-chain continuity: each scene's starting_pose := previous ending_pose.
    # The LLM is asked to do this but we make it crisp here.
    for i in range(1, len(plan["scenes"])):
        prev_end = (plan["scenes"][i - 1].get("ending_pose") or "").strip()
        if prev_end:
            plan["scenes"][i]["starting_pose"] = prev_end
    return plan


async def _request_outline(story: str, n_pages: int, style: str) -> dict:
    """Pass 1 of planning: produce a high-level beat outline whose scene counts sum to n_pages."""
    user = (
        f"Story idea: {story.strip()}\n"
        f"Total scenes to allocate: {n_pages}\n"
        f"Illustration style: {style}\n\n"
        "Return the outline as STRICT JSON."
    )
    try:
        raw = await _chat(_OUTLINE_SYSTEM, user, json_mode=True, timeout=120.0, max_tokens=1024)
        outline = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        return {}
    beats = outline.get("beats") if isinstance(outline, dict) else None
    if not isinstance(beats, list) or not beats:
        return {}
    # If the outline's scene allocation doesn't match n_pages, scale it. Don't reject —
    # any outline is better than none for guiding the expansion.
    total = sum(int(b.get("scenes") or 0) for b in beats)
    if total != n_pages and total > 0:
        # Proportional rescale, then fix rounding drift by adjusting the longest beat.
        scaled = [max(1, round(int(b.get("scenes") or 0) * n_pages / total)) for b in beats]
        drift = n_pages - sum(scaled)
        if drift != 0:
            idx = max(range(len(scaled)), key=lambda i: scaled[i])
            scaled[idx] = max(1, scaled[idx] + drift)
        for b, s in zip(beats, scaled):
            b["scenes"] = s
    return outline


async def _critique_plan(story: str, draft: dict, n_pages: int, style: str) -> dict:
    """Send the draft plan back through qwen3.6 for a one-shot review + revision.
    Returns the revised plan if the response is well-formed, otherwise the original draft."""
    draft_scenes = draft.get("scenes") or []
    if not draft_scenes:
        return draft
    user = (
        f"Original story idea: {story.strip()}\n"
        f"Illustration style: {style}\n"
        f"Scene count (MUST stay at {n_pages}): {len(draft_scenes)}\n\n"
        f"Draft plan to review:\n{json.dumps(draft, ensure_ascii=False, indent=2)}\n\n"
        "Return the corrected plan as STRICT JSON only."
    )
    try:
        raw = await _chat(
            _CRITIQUE_SYSTEM, user, json_mode=True,
            timeout=180.0, max_tokens=2048 + 256 * max(0, n_pages - 6),
        )
        revised = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        return draft
    # Sanity: revised must have the right shape and matching scene count
    if not isinstance(revised, dict):
        return draft
    new_scenes = revised.get("scenes") or []
    if not isinstance(new_scenes, list) or len(new_scenes) != n_pages:
        return draft
    # Drop revisions where any scene lost a required field
    required = ("starting_pose", "ending_pose", "description", "motion", "video_prompt")
    for s in new_scenes:
        if not isinstance(s, dict):
            return draft
        if not all((s.get(k) or "").strip() for k in required if k != "starting_pose"):
            # starting_pose for scene 0 is allowed to be empty pre-enforcement
            return draft
    # Preserve the original character description if the revision dropped it
    if not (revised.get("character") or "").strip():
        revised["character"] = draft.get("character", "")
    return revised


async def _request_plan(user: str, n_pages: int, max_tokens: int) -> dict:
    # Use the JSON-tuned model (qwen2.5:3b) for structured output
    raw = await _chat(_STORY_SYSTEM, user, json_mode=True, timeout=180.0, max_tokens=max_tokens)
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
