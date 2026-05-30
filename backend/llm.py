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
    "  \"character\": \"<two-sentence description of the PROTAGONIST: name, species, colors, distinctive features, clothing. Mirrors `characters[0]` below — kept as prose for UI display.>\",\n"
    "  \"characters\": [\n"
    "    {\n"
    "      \"name\":        \"<character's name (unique per character)>\",\n"
    "      \"role\":        \"protagonist\",  // EXACTLY ONE entry must have role='protagonist'; everyone else is 'supporting'\n"
    "      \"species\":     \"<species or kind: 'a small grey kitten', 'a 6-year-old boy with curly red hair', 'a friendly toy robot'>\",\n"
    "      \"colors\":      \"<key colors: 'soft grey fur with white paws and a pink nose'>\",\n"
    "      \"features\":    \"<distinctive features: 'big amber eyes, small white star on forehead, chipped left ear'>\",\n"
    "      \"clothing\":    \"<exact outfit, or empty string>\",\n"
    "      \"accessories\": \"<persistent objects the character carries, or empty string>\"\n"
    "    }\n"
    "    // Add a `supporting` character object for EACH other recurring character in the story.\n"
    "  ],\n"
    "  \"locations\": [\n"
    "    {\n"
    "      \"id\":          \"<short kebab-case identifier, e.g. 'kitchen', 'meadow', 'rocket-cockpit'. Used to tag scenes; must be unique.>\",\n"
    "      \"name\":        \"<human-friendly name, e.g. 'Mochi's kitchen', 'the sunny meadow'>\",\n"
    "      \"description\": \"<rich one-sentence visual description: architecture, materials, dominant colors, lighting quality, key props, time of day. This is used to paint the canonical reference of the empty environment; be concrete and visual, no characters.>\"\n"
    "    }\n"
    "    // Include EVERY distinct setting that any scene takes place in. Aim for the smallest set possible — reuse `id`s across scenes that share a setting.\n"
    "  ],\n"
    "  \"scenes\": [\n"
    "    {\n"
    "      \"location_id\":   \"<must match one of the `id`s in `locations`. Consecutive scenes should share the same id unless this is an explicit transit scene.>\",\n"
    "      \"starting_pose\": \"<concise description of the protagonist's body position + expression at the FIRST frame of this scene. Examples: 'standing by the window, hand on the glass, looking out', 'sitting on a log, eyes closed, smiling'. THIS MUST MATCH the previous scene's `ending_pose` word-for-word (continuity). For scene 1 this sets the opening pose.>\",\n"
    "      \"ending_pose\":   \"<concise description of the protagonist's body position + expression at the LAST frame of this scene. Same vocabulary style as starting_pose. This becomes the next scene's starting_pose.>\",\n"
    "      \"prev_link\":     \"<ONE short sentence describing how this scene picks up from the previous one. Reference the previous setting/action by name so the visual continuity is explicit. Example: 'Picks up the instant after Mochi opens the kitchen door — she is now stepping onto the meadow grass, the doorway visible behind her.' For scene 1, write 'Opening scene.'>\",\n"
    "      \"description\":   \"<one-sentence visual description of the still image, MUST include the protagonist's starting_pose so Flux paints them in that exact pose. Composition, setting (consistent with the location's description). Reference all named characters present.>\",\n"
    "      \"motion\":        \"<short: what the protagonist does. Must take them from starting_pose to ending_pose in one continuous arc.>\",\n"
    "      \"motion_timeline\": \"<verb-and-timing breakdown of the ~5s clip in the form '0-2s: <verb phrase>. 2-4s: <verb phrase>. 4-5s: <verb phrase>.' Each beat is ONE concrete verb, not a list. Wan follows timed verbs reliably.>\",\n"
    "      \"camera\":        \"<one of: 'static', 'slow dolly in', 'slow dolly out', 'slow pan left', 'slow pan right', 'slow tilt up', 'slow tilt down'. Pick the one that supports the moment; default to 'static' if unsure.>\",\n"
    "      \"video_prompt\":  \"<CONCISE cinematic direction for ONE simple motion (25-45 words). Name: (1) shot framing (close-up / medium / wide / overhead), (2) one short motion arc from starting_pose to ending_pose with manner (slowly, gently, eagerly), (3) one ambient detail (leaves drifting, steam rising). Avoid stacking multiple actions.>\",\n"
    "      \"motion_intensity\": \"<one of: 'still' (sitting, blinking, breathing), 'gentle' (waving, turning head, reaching), 'dynamic' (jumping, running steps, throwing). Default to 'gentle' if unsure.>\",\n"
    "      \"characters_in_scene\": [\"<exact character name>\", ...]  // ALWAYS include the protagonist. Include supporting characters ONLY when physically visible.\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Generate EXACTLY the number of scenes requested.\n\n"
    "Critical rules:\n"
    "1. POSE CONTINUITY: scene[i].starting_pose MUST equal scene[i-1].ending_pose word-for-word.\n"
    "2. LOCATIONS as FIRST-CLASS ENTITIES: enumerate every distinct setting in `locations` with a concrete visual description. Each scene gets `location_id` pointing into that list. REUSE ids whenever scenes share a setting — the same kitchen across 4 scenes is ONE location entry, not four.\n"
    "3. SETTING CONTINUITY: consecutive scenes should share the SAME `location_id` unless there's a dedicated TRANSIT scene moving the character from A to B. The transit beat is its own scene with its own location_id ('doorway', 'corridor', 'meadow-edge') and uses scene budget. Never cut directly from location A to location B with no transit.\n"
    "   - BAD: scene 3 location_id='kitchen', scene 4 location_id='meadow'. Background teleport.\n"
    "   - GOOD: scene 3 ends 'opening the kitchen door, looking out'. Scene 4 location_id='kitchen-doorway' shows stepping through. Scene 5 location_id='meadow'.\n"
    "4. NARRATIVE CONTINUITY (`prev_link`): every scene's prev_link must explicitly name the previous scene's ending state or location, so the new image obviously continues from the last one — not a hard cut.\n"
    "5. CHARACTER ROSTER: include every recurring character in `characters`; each scene's `characters_in_scene` lists ONLY characters PHYSICALLY VISIBLE. Many scenes will be protagonist-only.\n"
    "6. The image (`description`) AND the video (`video_prompt`) must both reflect the protagonist's starting_pose, so the first frame of the animation matches the previous page's last frame.\n"
    "7. ONE primary action per scene. Use the `motion_timeline` to break the 5s into 2-3 small verb beats — that is your single action, decomposed in time, NOT three separate actions. Resist packing multiple distinct actions.\n"
    "8. The pose change from starting_pose to ending_pose should be small enough to render in ~5 seconds (a tilt of the head, a hand reaching, two steps). Avoid full-body locomotion arcs.\n"
    "9. Child-friendly. Slow camera only."
)

_OUTLINE_SYSTEM = (
    "You are a children's story editor planning the narrative SKELETON of a short illustrated "
    "video before any scene-level writing. You return a tight outline with story beats, the "
    "number of scenes each beat uses, AND the set of distinct settings the story moves through.\n\n"
    "Return STRICT JSON with this shape (no markdown, no commentary):\n"
    "{\n"
    "  \"title\":     \"<short title>\",\n"
    "  \"character\": \"<two-sentence character description — same canon used later>\",\n"
    "  \"locations\": [\n"
    "    {\n"
    "      \"id\":   \"<short kebab-case id, e.g. 'kitchen', 'meadow', 'cave-entrance'>\",\n"
    "      \"name\": \"<human-friendly name>\",\n"
    "      \"description\": \"<one rich visual sentence: architecture, materials, dominant colors, lighting, key props, time of day. No characters.>\"\n"
    "    }\n"
    "  ],\n"
    "  \"beats\": [\n"
    "    {\"name\": \"setup\",      \"summary\": \"<who, where, what's normal>\",          \"location_id\": \"<id>\", \"scenes\": <int>},\n"
    "    {\"name\": \"inciting\",   \"summary\": \"<what disrupts the normal>\",            \"location_id\": \"<id>\", \"scenes\": <int>},\n"
    "    {\"name\": \"rising\",     \"summary\": \"<the main attempt or journey>\",         \"location_id\": \"<id>\", \"scenes\": <int>},\n"
    "    {\"name\": \"climax\",     \"summary\": \"<the key moment of change>\",            \"location_id\": \"<id>\", \"scenes\": <int>},\n"
    "    {\"name\": \"resolution\", \"summary\": \"<how things settle, what the character feels at the end>\", \"location_id\": \"<id>\", \"scenes\": <int>}\n"
    "  ]\n"
    "}\n\n"
    "Constraints:\n"
    "- Total of `scenes` across all beats must EQUAL the requested number of scenes exactly.\n"
    "- Allocate scenes proportionally — setup and resolution often need only 1 scene each; "
    "rising-action gets the most. Climax is usually 1-2 scenes.\n"
    "- Each beat lives in ONE primary `location_id`. If a beat crosses settings, it must have "
    "≥2 scenes (the extras are transit scenes that get their own location ids).\n"
    "- Keep `locations` SMALL — typically 2-5 entries. Re-use the same id across beats whenever "
    "the story returns to that setting. Don't proliferate per-scene locations.\n"
    "- Child-friendly, gentle pacing. No conflict beyond mild challenge."
)

_CRITIQUE_SYSTEM = (
    "You are a senior storyboard editor reviewing a draft children's animated picture-book plan. "
    "Your job: read the draft, find violations of the rules below, and return a REVISED plan "
    "that fixes them. If a scene already obeys the rules, keep it unchanged.\n\n"
    "Return STRICT JSON in the EXACT same shape as the input. Preserve EVERY field on every scene "
    "(location_id, starting_pose, ending_pose, prev_link, description, motion, motion_timeline, "
    "camera, video_prompt, motion_intensity, characters_in_scene). Preserve the `characters` and "
    "`locations` arrays at the top level. No markdown, no commentary, no notes about what you "
    "changed — just the corrected JSON.\n\n"
    "Hard constraints:\n"
    "- Scene COUNT must stay identical to the draft. Do not add or remove scenes; rewrite in place.\n"
    "- Keep the character canon and the locations list intact (you may refine prose, but ids stay).\n\n"
    "Things to check and fix:\n"
    "1. LOCATION TELEPORTS: do consecutive scenes share the same `location_id`? If scene N has "
    "   location A and scene N+1 has location B without an intermediate transit, REWRITE one of "
    "   them so they share a location id, OR convert it into an explicit transit (doorway, edge, "
    "   threshold). Update both `location_id` and the scene's description+prev_link to match.\n"
    "2. PREV_LINK QUALITY: does each scene's `prev_link` reference the previous scene's ending "
    "   state by name? If it's generic ('Continues the story.'), rewrite it to name the previous "
    "   action or location concretely.\n"
    "3. PACKED SCENES: does any scene have multiple actions joined by AND / THEN? Simplify to a "
    "   single primary action; move the dropped action to a neighboring scene if it fits.\n"
    "4. POSE CHAIN: does scene[i].starting_pose match scene[i-1].ending_pose word-for-word? If "
    "   not, align them.\n"
    "5. MOTION TIMELINE: does each scene have a `motion_timeline` with 2-3 timed verb beats "
    "   ('0-2s: ...')? If missing or vague, decompose the scene's motion into timed beats.\n"
    "6. CAMERA: is each scene's `camera` set to a valid value? If missing, set 'static'.\n"
    "7. STORY ARC: does the sequence have a clear beginning, middle, and end? Rewrite filler.\n"
    "8. ONE-MOTION RULE: each video_prompt describes ONE simple motion completable in ~5s.\n\n"
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
    """Render a single character canon dict as a descriptive clause used verbatim in
    Flux prompts. Empty/missing parts are skipped so we don't leak placeholder text."""
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


def render_cast(characters: list, names: list[str] | None = None) -> str:
    """Render multiple character canons as a single combined clause. If `names` is
    given, only render the characters whose `name` matches one in the list. Joined
    with semicolons so Flux/Kontext parses each canon distinctly."""
    if not isinstance(characters, list) or not characters:
        return ""
    if names is not None:
        wanted = {n.strip().lower() for n in names if n}
        characters = [c for c in characters if isinstance(c, dict)
                      and (c.get("name") or "").strip().lower() in wanted]
    clauses = [render_canon(c) for c in characters]
    return "; ".join(c for c in clauses if c)


def coerce_characters(plan: dict) -> list[dict]:
    """Normalise the `characters` list from a plan dict. Accepts:
      - new format: plan["characters"] = [{name, role, species, ...}, ...]
      - old format: plan["character_canon"] = {name, species, ...} (single character)
    Always returns a list with the protagonist first."""
    chars = plan.get("characters")
    if isinstance(chars, list) and chars:
        out = []
        for c in chars:
            if isinstance(c, dict) and (c.get("name") or "").strip():
                # Default role inference: protagonist if it's the first one and unspecified
                role = (c.get("role") or "").strip().lower()
                if role not in ("protagonist", "supporting"):
                    role = "protagonist" if not out else "supporting"
                out.append({**c, "role": role})
        # Ensure exactly one protagonist (first one wins; others demoted)
        seen_protagonist = False
        for c in out:
            if c["role"] == "protagonist":
                if seen_protagonist:
                    c["role"] = "supporting"
                seen_protagonist = True
        return out
    # Fall back to the old single-canon shape
    legacy = plan.get("character_canon")
    if isinstance(legacy, dict) and (legacy.get("name") or "").strip():
        return [{**legacy, "role": "protagonist"}]
    return []


def protagonist_of(characters: list[dict]) -> dict | None:
    """Return the protagonist character (first one with role='protagonist') or None."""
    for c in characters:
        if isinstance(c, dict) and c.get("role") == "protagonist":
            return c
    return characters[0] if characters else None


def _slugify_location_id(s: str) -> str:
    s = (s or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in " -_":
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "place"


def coerce_locations(plan: dict) -> list[dict]:
    """Normalise the `locations` list. Ensures every entry has non-empty id/name/description
    and ids are unique slugified strings. Returns at least one fallback entry if empty."""
    raw = plan.get("locations")
    out: list[dict] = []
    seen_ids: set[str] = set()
    if isinstance(raw, list):
        for loc in raw:
            if not isinstance(loc, dict):
                continue
            lid = _slugify_location_id(loc.get("id") or loc.get("name") or "")
            if not lid or lid in seen_ids:
                # de-dup by suffixing
                base = lid or "place"
                i = 2
                while f"{base}-{i}" in seen_ids:
                    i += 1
                lid = f"{base}-{i}" if base else f"place-{i}"
            seen_ids.add(lid)
            out.append({
                "id":          lid,
                "name":        (loc.get("name") or lid.replace("-", " ")).strip(),
                "description": (loc.get("description") or "").strip(),
            })
    if not out:
        out.append({"id": "scene", "name": "the scene", "description": ""})
    return out


def location_by_id(locations: list[dict], lid: str) -> dict | None:
    """Find a location dict by id; case-insensitive, slug-normalised."""
    if not lid:
        return None
    target = _slugify_location_id(lid)
    for loc in locations:
        if isinstance(loc, dict) and _slugify_location_id(loc.get("id") or "") == target:
            return loc
    return None


def render_location(loc: dict | None) -> str:
    """Render a location as a descriptive clause for prompts. Empty parts skipped."""
    if not isinstance(loc, dict):
        return ""
    name = (loc.get("name") or "").strip()
    desc = (loc.get("description") or "").strip()
    if name and desc:
        return f"{name} — {desc}"
    return name or desc


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
        # Wrap the legacy single-canon shape into the new array form for the locked clause
        protagonist_locked = {**(existing_canon or {}), "role": "protagonist"}
        prose = (existing_character or "").strip()
        locked_clause = (
            "\n\nLOCKED PROTAGONIST: a saved character is being re-used. The first entry of "
            "`characters` (role='protagonist') MUST be set to the exact object below (copy it "
            "verbatim) and `character` MUST be set to the exact prose below. You may still add "
            "supporting characters appropriate to the story, but the protagonist is locked.\n"
            f"  characters[0] = {json.dumps(protagonist_locked, ensure_ascii=False)}\n"
            f"  character (prose) = \"{prose}\"\n"
        )

    user = (
        f"Story idea: {story.strip()}\n\n"
        f"Number of scenes: EXACTLY {n_pages} (no more, no fewer)\n"
        f"Illustration style: {style}\n\n"
        f"Every scene MUST include: location_id, starting_pose, ending_pose, prev_link, "
        f"description, motion, motion_timeline, camera, video_prompt, motion_intensity, "
        f"characters_in_scene. The `locations` array is REQUIRED — enumerate every distinct "
        f"setting (typically 2-5 entries) and tag each scene with one of those ids. The "
        f"`characters` array is REQUIRED and contains the protagonist plus every supporting "
        f"character that appears; each character needs non-empty name/species/colors/features "
        f"(clothing/accessories may be empty). Each scene's `characters_in_scene` lists only "
        f"characters physically present (always include the protagonist). The starting_pose of "
        f"each scene (after scene 1) MUST match the previous scene's ending_pose exactly. The "
        f"prev_link must explicitly reference the previous scene's ending state by name. "
        f"Return STRICT JSON only."
        + outline_hint
        + locked_clause
    )
    plan = await _request_plan(user, n_pages, max_tokens=3072 + 256 * max(0, n_pages - 6))

    # If a character is locked, force-overwrite the protagonist entry in case the model drifted.
    if existing_canon or existing_character:
        chars = coerce_characters(plan)
        if not chars:
            chars = [{"name": (existing_canon or {}).get("name") or "the character",
                      "role": "protagonist"}]
        if existing_canon:
            # Replace protagonist's canon with the locked one (keep role='protagonist').
            chars[0] = {**(existing_canon or {}), "role": "protagonist"}
        plan["characters"] = chars
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
        "starting_pose":    "in a calm, settled stance",
        "ending_pose":      "in a calm, settled stance",
        "prev_link":        "Continues directly from the previous moment.",
        "description":      "A quiet moment in the story.",
        "motion":           "subtle motion",
        "motion_timeline":  "0-2s: gentle breath and small adjustment. 2-5s: the small motion completes.",
        "camera":           "static",
        "video_prompt":     "A soft warm storybook scene. Gentle, slow motion. Cinematic lighting that settles peacefully.",
        "motion_intensity": "gentle",
    }
    valid_intensities = {"still", "gentle", "dynamic"}
    valid_cameras = {
        "static", "slow dolly in", "slow dolly out",
        "slow pan left", "slow pan right",
        "slow tilt up", "slow tilt down",
    }
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
        # Normalise camera onto the small allowed set
        cam = (s.get("camera") or "static").strip().lower()
        s["camera"] = cam if cam in valid_cameras else "static"

    # Normalise the characters array (also handles old-format saved-character locks).
    plan["characters"] = coerce_characters(plan)
    if not plan["characters"]:
        # Last-resort fallback: synthesise a generic protagonist so the orchestrator
        # always has something to inject. This should be very rare.
        plan["characters"] = [{
            "name": "Hero",
            "role": "protagonist",
            "species": "the main character",
            "colors": "",
            "features": "",
            "clothing": "",
            "accessories": "",
        }]

    # Sync the prose `character` field with the protagonist for UI display.
    proto = protagonist_of(plan["characters"])
    if proto and not (plan.get("character") or "").strip():
        plan["character"] = render_canon(proto)
    # Keep `character_canon` populated as the protagonist's canon, for backward compat
    # with the character library save endpoint that reads that single field.
    if proto:
        plan["character_canon"] = {k: v for k, v in proto.items() if k != "role"}

    # Enforce characters_in_scene includes the protagonist + only references known names.
    known_names = {(c.get("name") or "").strip() for c in plan["characters"]}
    protagonist_name = (proto.get("name") or "").strip() if proto else ""
    for s in plan["scenes"]:
        in_scene = s.get("characters_in_scene")
        if not isinstance(in_scene, list):
            in_scene = []
        # Keep only names that match known characters; preserve order.
        in_scene = [n for n in in_scene if isinstance(n, str) and n.strip() in known_names]
        # Always include the protagonist.
        if protagonist_name and protagonist_name not in in_scene:
            in_scene.insert(0, protagonist_name)
        s["characters_in_scene"] = in_scene

    # Normalise locations + assign each scene a valid location_id. If the planner emitted
    # no locations, synthesise one from the protagonist's setting so downstream code always
    # has something to render.
    plan["locations"] = coerce_locations(plan)
    known_loc_ids = {loc["id"] for loc in plan["locations"]}
    default_loc_id = plan["locations"][0]["id"]
    last_loc_id = default_loc_id
    for s in plan["scenes"]:
        lid_raw = (s.get("location_id") or "").strip()
        lid = _slugify_location_id(lid_raw) if lid_raw else ""
        if lid not in known_loc_ids:
            # Fall back to the previous scene's location id (keeps continuity) rather than
            # the first location — better than randomly snapping to setup-time location.
            lid = last_loc_id
        s["location_id"] = lid
        last_loc_id = lid

    # Enforce pose-chain continuity: each scene's starting_pose := previous ending_pose.
    for i in range(1, len(plan["scenes"])):
        prev_end = (plan["scenes"][i - 1].get("ending_pose") or "").strip()
        if prev_end:
            plan["scenes"][i]["starting_pose"] = prev_end

    # Scene 1's prev_link is always "opening scene" — overwrite if the LLM didn't.
    if plan["scenes"]:
        first_link = (plan["scenes"][0].get("prev_link") or "").strip().lower()
        if not first_link or "scene 1" in first_link or "previous" in first_link:
            plan["scenes"][0]["prev_link"] = "Opening scene."
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
    # If a critical field is missing on the revised scene, copy it from the draft
    # rather than discarding the whole revision (the LLM often drops a couple of
    # newer fields like motion_timeline even when the rest of the revision is solid).
    merge_keys = (
        "starting_pose", "ending_pose", "prev_link", "description", "motion",
        "video_prompt", "motion_timeline", "camera", "motion_intensity",
        "location_id", "characters_in_scene",
    )
    for idx, s in enumerate(new_scenes):
        if not isinstance(s, dict):
            return draft
        draft_s = draft_scenes[idx] if idx < len(draft_scenes) else {}
        for k in merge_keys:
            v = s.get(k)
            if (isinstance(v, str) and not v.strip()) or v is None:
                fallback = draft_s.get(k) if isinstance(draft_s, dict) else None
                if fallback is not None:
                    s[k] = fallback
    # Hard floor: the essentials must end up populated. If not, fall back to draft.
    essentials = ("description", "motion", "video_prompt")
    for s in new_scenes:
        if not all((s.get(k) or "").strip() for k in essentials):
            return draft
    # Preserve the original character description / locations if the revision dropped them.
    if not (revised.get("character") or "").strip():
        revised["character"] = draft.get("character", "")
    if not isinstance(revised.get("locations"), list) or not revised.get("locations"):
        revised["locations"] = draft.get("locations", [])
    if not isinstance(revised.get("characters"), list) or not revised.get("characters"):
        revised["characters"] = draft.get("characters", [])
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
