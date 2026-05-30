"""Prompt improvement and multi-step storybook planning (outline → plan → critique)."""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from llm.client import _chat
from llm.prompts import (
    _CRITIQUE_SYSTEM,
    _IMPROVE_SYSTEM,
    _OUTLINE_SYSTEM,
    _STORY_SYSTEM,
    _STYLE_HINTS,
)
from llm.render import (
    coerce_characters,
    coerce_locations,
    protagonist_of,
    render_canon,
    _slugify_location_id,
)

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




async def plan_storybook(
    story: str,
    n_pages: int,
    style: str,
    existing_canon: Optional[dict] = None,
    existing_character: Optional[str] = None,
) -> dict:
    """Return {character, characters:[...], locations:[...], scenes:[{...}]} of length n_pages.

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
        f"characters_in_scene, objects_in_hand, object_change. The `locations` array is REQUIRED — enumerate every distinct "
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
        # Retry re-emits the WHOLE plan after a truncation, so scale with pages and give
        # more headroom than the initial attempt (base 4096 vs 3072) — the previous run
        # already proved this plan runs long.
        plan = await _request_plan(retry_user, n_pages, max_tokens=4096 + 256 * max(0, n_pages - 6))

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
        "object_change":    "none",
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

    # Normalise per-scene object lists: objects_in_hand is the SET of things the protagonist
    # is holding at the END of the scene; object_change is the diff verb. We do not hard-reject
    # the plan when the diff is inconsistent (qwen3.6 plans are noisy), but we do enforce a
    # cheap consistency repair: if objects_in_hand changed but object_change is "none", flip
    # it to a generic pickup/putdown so downstream prompts at least mention the handling.
    prev_objects: list[str] = []
    for s in plan["scenes"]:
        raw = s.get("objects_in_hand")
        if isinstance(raw, list):
            objs = [str(o).strip() for o in raw if isinstance(o, (str, int, float)) and str(o).strip()]
        elif isinstance(raw, str) and raw.strip():
            objs = [raw.strip()]
        else:
            objs = []
        s["objects_in_hand"] = objs
        change = (s.get("object_change") or "").strip().lower()
        prev_set, curr_set = set(prev_objects), set(objs)
        if prev_set != curr_set and change in ("", "none"):
            added = curr_set - prev_set
            dropped = prev_set - curr_set
            if added and dropped:
                s["object_change"] = f"swaps {', '.join(sorted(dropped))} for {', '.join(sorted(added))}"
            elif added:
                s["object_change"] = f"picks up {', '.join(sorted(added))}"
            elif dropped:
                s["object_change"] = f"puts down {', '.join(sorted(dropped))}"
        elif not change:
            s["object_change"] = "none"
        prev_objects = objs

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
        # The critique re-emits the FULL revised plan (same shape/size as the input), so it
        # needs at least the plan cap plus margin — the old 2048 base ran ~96% full on busy
        # 12-page plans and could truncate, silently dropping the whole quality pass.
        raw = await _chat(
            _CRITIQUE_SYSTEM, user, json_mode=True,
            timeout=180.0, max_tokens=4096 + 256 * max(0, n_pages - 6),
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
        "location_id", "characters_in_scene", "objects_in_hand", "object_change",
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


