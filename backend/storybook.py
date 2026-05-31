"""Storybook orchestration: plan → per-page keyframes (Flux/Kontext) → per-page Wan
animation → stitch. Also the post-hoc per-scene regenerate + restitch helpers.

Mutates the shared singleton via `state.active_gen` so the routes can report progress and
drive the keyframe-approval gate."""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Optional

import drift
import llm
import state
from comfy import _comfy_free, _extract_last_frame, _stitch_videos, _submit_comfy_and_wait
from config import (
    CHARACTER_LIBRARY_DIR,
    CHARACTER_LIBRARY_FILE,
    COMFY_INPUT,
    COMFY_OUTPUT,
    DEFAULT_NEGATIVE,
    FLUX_KONTEXT_MODEL,
    HISTORY_FILE,
    SMOKE_PAGES,
    STORY_ASPECT_DIMS,
    STYLE_PREFIXES,
)
from models import GenerateParams, ImageGenerateParams, StorybookParams
from store import load_json, save_json
from workflows import (
    build_flux_image_workflow,
    build_flux_kontext_edit_workflow,
    build_flux_kontext_workflow,
    build_wan22_i2v_workflow,
)


_POSE_STOPWORDS = {
    "the", "and", "with", "his", "her", "its", "their", "they", "are", "for", "into",
    "onto", "from", "out", "off", "but", "not", "near", "over", "this", "that",
    "slightly", "gently", "softly", "faintly", "slowly", "still", "while", "then",
}


def _poses_differ(start_pose: str, end_pose: str, threshold: float = 0.7) -> bool:
    """True when start and end poses describe a meaningfully different body state.

    Compares content-word sets (Jaccard similarity). The cookie-story LLM leaves
    object_change empty, but always emits rich pose text, so the pose delta — not the
    object fields — is the reliable signal for 'does this scene have a real beat that needs
    a Kontext end-keyframe?'. Near-identical poses (ambient beats) return False and the
    scene animates as pure background-locked I2V instead."""
    def toks(s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]+", (s or "").lower())
                if w not in _POSE_STOPWORDS and len(w) > 2}
    a, b = toks(start_pose), toks(end_pose)
    if not a and not b:
        return False
    if not a or not b:
        return True
    sim = len(a & b) / len(a | b)
    return sim < threshold


def _sanitize_char_name(name: str) -> str:
    """Lowercase + strip non-alphanumerics so character names become safe filename parts."""
    return "".join(c.lower() if c.isalnum() else "_" for c in (name or "")).strip("_") or "char"


def _build_model_sheet_prompt(canon_clause: str, style_prefix: str) -> str:
    """Prompt template that asks Flux for a neutral 'character model sheet' headshot —
    used as the canonical reference image fed to Kontext for that character later."""
    return (
        f"Character model sheet: {canon_clause}. "
        f"Full body shot, neutral standing pose facing forward, plain off-white background, "
        f"soft even studio lighting, no shadows, no scenery, no other characters. "
        f"{style_prefix}."
    )


def _composite_refs(
    char_paths: list[Path],
    output_path: Path,
    location_path: Optional[Path] = None,
) -> None:
    """Build the Kontext reference image for a scene by concatenating refs side-by-side.

    Layout: [ location (~50% width) | char1 | char2 | ... ]

    Kontext only takes one image; a wide left-to-right strip preserves all the visual
    cues. Location goes first (left) when supplied so Kontext keys the *setting* off it,
    then characters. With one character and no location the image is copied through
    verbatim so single-protagonist scenes behave identically to the original single-ref
    flow."""
    from PIL import Image
    if not char_paths and not location_path:
        raise ValueError("no refs to composite")
    if not char_paths and location_path:
        shutil.copyfile(str(location_path), str(output_path))
        return
    if len(char_paths) == 1 and location_path is None:
        shutil.copyfile(str(char_paths[0]), str(output_path))
        return

    images: list = []
    if location_path is not None:
        images.append(Image.open(location_path).convert("RGB"))
    images.extend(Image.open(p).convert("RGB") for p in char_paths)

    # Normalise heights so the strip looks coherent.
    target_h = min(img.height for img in images)
    resized = []
    for idx, img in enumerate(images):
        if img.height != target_h:
            new_w = int(img.width * target_h / img.height)
            img = img.resize((new_w, target_h), Image.LANCZOS)
        # Cap the location panel width so a wide landscape location doesn't drown out
        # the character refs. Aim for ≤ total character width.
        if idx == 0 and location_path is not None and len(char_paths) >= 1:
            chars_total_w = 0
            for c in images[1:]:
                cw = int(c.width * target_h / c.height) if c.height != target_h else c.width
                chars_total_w += cw
            max_loc_w = max(target_h, chars_total_w)
            if img.width > max_loc_w:
                left = (img.width - max_loc_w) // 2
                img = img.crop((left, 0, left + max_loc_w, target_h))
        resized.append(img)

    total_w = sum(img.width for img in resized)
    combined = Image.new("RGB", (total_w, target_h), (245, 245, 240))
    x = 0
    for img in resized:
        combined.paste(img, (x, 0))
        x += img.width
    combined.save(str(output_path), "PNG")


def _build_wan_prompt(
    video_prompt: str,
    starting_pose: str,
    ending_pose: str,
    character: str,
    style: str,
    motion_timeline: str = "",
    camera: str = "",
    location_clause: str = "",
    objects_clause: str = "",
    object_change: str = "",
) -> str:
    """Assemble the full Wan prompt from the LLM's per-scene direction.

    Wan 2.2 follows timed verbs and camera cues well, so the timeline and camera get
    woven in explicitly. The location clause anchors the setting so Wan doesn't drift
    the background across the 5s clip."""
    pose_chain = ""
    if starting_pose and ending_pose:
        pose_chain = (
            f" The scene starts with {character} {starting_pose}, and ends with "
            f"{character} {ending_pose}."
        )
    elif ending_pose:
        pose_chain = f" The scene ends with {character} {ending_pose}."

    setting = f" Setting: {location_clause}." if location_clause else ""
    timeline = f" Timeline — {motion_timeline}" if motion_timeline else ""
    cam = camera.strip().lower() if camera else ""
    cam_clause = ""
    if cam and cam != "static":
        cam_clause = f" Camera: {cam}."
    elif cam == "static":
        cam_clause = " Camera: locked, no movement."

    obj_clause = ""
    if objects_clause:
        obj_clause = f" The character is holding {objects_clause} the entire shot; the object does not morph or change."
    elif objects_clause == "":
        # explicit empty-hands signal only when the LLM emitted empty list (callers can
        # pass "" to mean "don't say anything")
        pass
    change_clause = ""
    if object_change and object_change.lower() != "none":
        change_clause = f" Action note: {object_change}."

    return (
        f"{video_prompt}{pose_chain}{setting}{timeline}{cam_clause}{obj_clause}{change_clause} "
        f"{character}. "
        f"Storybook illustration style, {style} aesthetic, soft cinematic lighting, "
        f"smooth gentle motion, background remains stable and consistent throughout the shot."
    )


async def _run_storybook(p: StorybookParams, prompt_id: str, gen_id: str):
    """Background orchestration: plan → image per page → video per page → stitch."""
    try:
        if state.active_gen is not None:
            state.active_gen.node = "planning"

        # Free any ComfyUI-cached models (Flux from a recent image gen, etc.) so the
        # planner LLM can load without contending for GPU memory.
        await _comfy_free()

        # If the user picked a saved character, load its canon + reference image and
        # tell the planner to keep that character's canonical description verbatim.
        saved_character: Optional[dict] = None
        if p.character_id:
            library = load_json(CHARACTER_LIBRARY_FILE, [])
            saved_character = next((c for c in library if c.get("id") == p.character_id), None)
            if saved_character:
                ref_src = CHARACTER_LIBRARY_DIR / (saved_character.get("ref_filename") or "")
                if not ref_src.exists():
                    saved_character = None  # silent fallback to fresh generation

        # Smoke-test knobs: a few short pages, low Wan steps/frames, low Kontext steps.
        # The dominant cost is Wan, so steps 20→8 and frames 81→33 give the biggest cut;
        # n_pages is capped to keep at least one same-location beat (→ exercises the img2img
        # background-lock) and ≥1 seam (→ exercises chaining + crossfade).
        smoke = bool(getattr(p, "smoke", False))
        plan_pages = SMOKE_PAGES if smoke else p.n_pages
        kontext_steps = 10 if smoke else 20
        vid_steps = 8 if smoke else 20
        vid_frames = 33 if smoke else 81
        if smoke:
            print(f"[storybook] SMOKE MODE: pages={plan_pages} wan_steps={vid_steps} "
                  f"wan_frames={vid_frames} kontext_steps={kontext_steps} (auto-approve on)")

        # 1) Use the LLM to plan (passing the saved character's canon, if any, so the
        # LLM treats it as locked rather than inventing a new protagonist).
        plan = await llm.plan_storybook(
            p.story, plan_pages, p.style,
            existing_canon=saved_character.get("canon") if saved_character else None,
            existing_character=saved_character.get("character") if saved_character else None,
        )
        character = plan.get("character", "")
        characters = llm.coerce_characters(plan)
        protagonist = llm.protagonist_of(characters) or {}
        locations = llm.coerce_locations(plan)
        scenes = plan.get("scenes") or []
        if not scenes:
            raise RuntimeError("LLM returned no scenes")
        if smoke:
            scenes = scenes[:SMOKE_PAGES]   # cap in case the planner over-produced

        # Surface plan in state for the frontend to show
        if state.active_gen is not None:
            state.active_gen.character = character
            state.active_gen.scene_descriptions = [
                s.get("description") or "" for s in scenes
            ]

        # Free the LLM from VRAM before starting the long stream of Flux + Wan gens
        await llm.unload()

        style_prefix = STYLE_PREFIXES.get(p.style.lower(), STYLE_PREFIXES["pixar"])
        width, height = STORY_ASPECT_DIMS.get(p.aspect, STORY_ASPECT_DIMS["landscape"])
        seed = int(time.time()) % (2**31)

        NOISE_AUG_BY_INTENSITY = {"still": 0.0, "gentle": 0.05, "dynamic": 0.10}
        kontext_available = (Path("/media/yunus/More Data/comfyui-models/diffusion_models") / FLUX_KONTEXT_MODEL).exists()

        # Total steps: image + animate per scene; the keyframe preview gate sits between.
        total = len(scenes) * 2
        step_done = 0

        # =================== PHASE 0 — Per-character canonical reference images ===================
        # Protagonist is NOT pre-generated as a model sheet — that regresses overall
        # coherency because the neutral-background sheet pulls Kontext toward "studio
        # portrait" aesthetic and away from the storybook style. Instead, the
        # protagonist's canonical ref is set to page 1's ACTUAL scene render below
        # (matches what worked in the single-character version).
        #
        # Supporting characters DO get model sheets here, because we need *some* visual
        # anchor for them before they appear, and we don't have a scene-render to use.
        # These sheets feed only into the composite for scenes where they appear.
        protagonist_name = (protagonist.get("name") or "").strip()
        char_refs: dict[str, str] = {}

        if saved_character and protagonist_name:
            # Saved-character flow: use the saved ref as the protagonist's canonical ref
            # immediately, so the composite for page 1 already has the locked character.
            src = CHARACTER_LIBRARY_DIR / saved_character["ref_filename"]
            ref_filename = f"char_ref_{_sanitize_char_name(protagonist_name)}_{gen_id}.png"
            shutil.copyfile(str(src), str(COMFY_INPUT / ref_filename))
            char_refs[protagonist_name] = ref_filename

        # Supporting characters: one model-sheet Flux T2I each, used in composites.
        supporting_count = 0
        for char in characters:
            name = (char.get("name") or "").strip()
            if not name or char.get("role") == "protagonist":
                continue
            if state.active_gen is not None:
                state.active_gen.node = f"casting-{name}"
            ref_filename = f"char_ref_{_sanitize_char_name(name)}_{gen_id}.png"
            clause = llm.render_canon(char) or name
            prompt = _build_model_sheet_prompt(clause, style_prefix)
            wf_ref = build_flux_image_workflow(ImageGenerateParams(
                image_mode="create", prompt=prompt,
                width=width, height=height,
                seed=seed + abs(hash(name)) % 100000,
                model="flux",
            ))
            ref_out = await _submit_comfy_and_wait(wf_ref, timeout_s=300)
            shutil.copyfile(str(COMFY_OUTPUT / ref_out), str(COMFY_INPUT / ref_filename))
            char_refs[name] = ref_filename
            supporting_count += 1

        # Publish the cast (so far — protagonist gets a placeholder until page 1 lands).
        if state.active_gen is not None:
            state.active_gen.cast = [
                {
                    "name": (c.get("name") or "").strip(),
                    "role": c.get("role") or "supporting",
                    "species": c.get("species") or "",
                    "ref_filename": char_refs.get((c.get("name") or "").strip(), ""),
                }
                for c in characters
                if (c.get("name") or "").strip()
            ]

        # =================== PHASE 0.5 — Per-location canonical reference images ===================
        # One Flux T2I per UNIQUE location the story passes through. These get composited
        # into the Kontext reference for every scene set in that location, so the
        # background reads identically across scenes (the fix for "environment changes
        # abruptly"). Locations referenced by zero scenes are skipped to save Flux calls.
        loc_refs: dict[str, str] = {}
        used_loc_ids = {(s.get("location_id") or "").strip() for s in scenes}
        used_loc_ids.discard("")
        for loc in locations:
            lid = loc.get("id") or ""
            if lid not in used_loc_ids:
                continue
            if state.active_gen is not None:
                state.active_gen.node = f"location-{lid}"
            loc_filename = f"loc_ref_{_sanitize_char_name(lid)}_{gen_id}.png"
            loc_clause = llm.render_location(loc) or lid
            loc_prompt = (
                f"{style_prefix}. Wide establishing shot of {loc_clause}. "
                f"Empty environment — no characters, no people, no animals. "
                f"Soft cinematic lighting, rich background detail, coherent palette."
            )
            wf_loc = build_flux_image_workflow(ImageGenerateParams(
                image_mode="create", prompt=loc_prompt,
                width=width, height=height,
                seed=seed + abs(hash(lid)) % 100000,
                model="flux",
            ))
            loc_out = await _submit_comfy_and_wait(wf_loc, timeout_s=300)
            shutil.copyfile(str(COMFY_OUTPUT / loc_out), str(COMFY_INPUT / loc_filename))
            loc_refs[lid] = loc_filename

        # =================== PHASE A — Generate all Flux start+end pairs ===================
        # We do this in one pass so the user can preview every keyframe before committing
        # to the heavy Wan phase. Every keyframe runs through Flux Kontext with a
        # composite reference image built from the characters present in that scene —
        # this is what locks supporting characters' appearance across scenes.
        # Background-anchor chain: scenes 2+ within the SAME location_id (and within the cap)
        # use the previous scene's end image as the leftmost composite panel — this locks
        # the background visually instead of letting Kontext re-invent the kitchen on every
        # scene. The cap (LOC_CHAIN_CAP) bounds compounding drift; once exceeded, we re-anchor
        # to the canonical location ref.
        # Effectively unbounded: within a single continuous location run we ALWAYS inherit
        # the previous scene's end frame as the background anchor (with the "preserve the
        # EXACT room" Kontext prompt). The old cap of 3 forced a re-anchor to the canonical
        # location ref every 3 pages, which produced a FRESH Kontext render of the room —
        # i.e. a periodic background reshuffle right at a seam (the pops at pages 7→8 and
        # 10→11 in the cookie story). Re-anchoring still happens automatically on a genuine
        # location change because loc_id != prev_loc_id resets the chain.
        LOC_CHAIN_CAP = 10**9
        prev_loc_id: str = ""
        loc_chain_len: int = 0
        prev_end_image_filename: Optional[str] = None
        prev_objects: list[str] = []
        for i, scene in enumerate(scenes):
            scene_desc = scene.get("description") or "A scene from the story."
            starting_pose = scene.get("starting_pose") or ""
            ending_pose = scene.get("ending_pose") or ""
            intensity = (scene.get("motion_intensity") or "gentle").lower()
            chars_in_scene = scene.get("characters_in_scene") or ([protagonist_name] if protagonist_name else [])
            # Location resolution: scene tags a location_id; we look up the canon and the
            # pre-generated ref. Fall back gracefully if the LLM emitted an unknown id.
            loc_id = (scene.get("location_id") or "").strip()
            loc_obj = llm.location_by_id(locations, loc_id) if loc_id else None
            loc_clause = llm.render_location(loc_obj)
            loc_ref_filename = loc_refs.get(loc_id) or ""
            prev_link = (scene.get("prev_link") or "").strip()
            motion_timeline = (scene.get("motion_timeline") or "").strip()
            # Camera is FORCED static for stitched storybooks. Each page is an independent
            # Wan clip with no shared camera trajectory, so a per-clip dolly/pan ends at one
            # framing and the next clip starts a different move from a different framing —
            # that mismatch is a major source of the seam "pops". A locked frame also helps
            # Wan hold the background stable across the 5s shot. We still record the LLM's
            # intended camera in the keyframe meta for reference.
            planned_camera = (scene.get("camera") or "static").strip()
            camera = "static"
            objects_in_hand = scene.get("objects_in_hand") or []
            if not isinstance(objects_in_hand, list):
                objects_in_hand = []
            objects_clause = llm.render_objects(objects_in_hand)
            object_change = (scene.get("object_change") or "none").strip()

            if state.active_gen is not None:
                state.active_gen.node = f"page-{i+1}-image"
                state.active_gen.step = step_done
                state.active_gen.total_steps = total

            start_image_input_name = f"storybook_start_p{i}_{gen_id}.png"
            end_image_input_name = f"storybook_end_p{i}_{gen_id}.png"

            # Per-scene canon clause (text-level): lists every character present + their canon.
            # This is the load-bearing piece for supporting-character consistency since
            # Flux follows text strongly.
            scene_canon = llm.render_cast(characters, names=chars_in_scene) or (character or "")

            # ---- START image ----
            # Page 1: plain Flux T2I (no Kontext) so the protagonist's canonical look is
            # established by a real scene render — the OLD working behavior. That image
            # then becomes the protagonist's canonical ref for every later Kontext call.
            # Pages 2+: byte-perfect copy of the previous page's end image (so cuts are
            # invisible at stitch time).
            multi_char = len(chars_in_scene) > 1
            setting_clause = f"Setting: {loc_clause}. " if loc_clause else ""
            start_prompt = ""
            if i == 0:
                start_pose_text = starting_pose or "in an initial settled pose"
                # For single-character scenes, only describe the protagonist — pluralised
                # phrasing tends to make Flux invent extra companions out of thin air.
                cast_clause = scene_canon if multi_char else (llm.render_canon(protagonist) or character)
                start_prompt = (
                    f"{style_prefix}. {setting_clause}{cast_clause}. {scene_desc}. "
                    f"{character} is {start_pose_text}."
                )
                if saved_character:
                    # Saved-character flow: the saved ref IS the protagonist's canonical
                    # look, so use it as a Kontext ref for page 1 — this anchors the saved
                    # appearance into the scene.
                    proto_ref = char_refs.get(protagonist_name, "")
                    if kontext_available and proto_ref and (COMFY_INPUT / proto_ref).exists():
                        wf = build_flux_kontext_workflow(
                            prompt=start_prompt, width=width, height=height,
                            seed=seed, reference_image=proto_ref, steps=kontext_steps,
                        )
                    else:
                        wf = build_flux_image_workflow(ImageGenerateParams(
                            image_mode="create", prompt=start_prompt,
                            width=width, height=height, seed=seed, model="flux",
                        ))
                else:
                    wf = build_flux_image_workflow(ImageGenerateParams(
                        image_mode="create", prompt=start_prompt,
                        width=width, height=height, seed=seed, model="flux",
                    ))
                start_out = await _submit_comfy_and_wait(wf, timeout_s=300)
                shutil.copyfile(str(COMFY_OUTPUT / start_out), str(COMFY_INPUT / start_image_input_name))
                # Page 1's render becomes the protagonist's canonical ref for every later
                # Kontext call (single-character flow). Saved-character flow already set
                # char_refs[protagonist] in Phase 0.
                if protagonist_name and not saved_character:
                    proto_ref_filename = f"char_ref_{_sanitize_char_name(protagonist_name)}_{gen_id}.png"
                    shutil.copyfile(str(COMFY_OUTPUT / start_out), str(COMFY_INPUT / proto_ref_filename))
                    char_refs[protagonist_name] = proto_ref_filename
                    if state.active_gen is not None:
                        for entry in state.active_gen.cast:
                            if entry["name"] == protagonist_name:
                                entry["ref_filename"] = proto_ref_filename
                                break
                page_thumb = start_out
            else:
                shutil.copyfile(str(COMFY_OUTPUT / prev_end_image_filename), str(COMFY_INPUT / start_image_input_name))
                page_thumb = prev_end_image_filename
            if state.active_gen is not None:
                state.active_gen.preview_images.append(page_thumb)

            # ---- Background-anchor selection (Fix #1: prev-end-as-ref with cap) ----
            # Default: use the canonical location ref as the leftmost composite panel.
            # When the location is unchanged from the previous scene AND we're within the
            # cap, swap in the previous scene's end image instead — this locks the exact
            # kitchen Bolt was just in, eliminating the "kitchen reshuffles on every scene"
            # drift. The cap bounds compounding drift to 3 consecutive inherited backgrounds;
            # after that we re-anchor to the canonical location ref to refresh.
            if i == 0:
                loc_chain_len = 1
                bg_anchor_kind = "loc_ref"
            elif loc_id and loc_id == prev_loc_id and loc_chain_len < LOC_CHAIN_CAP:
                loc_chain_len += 1
                bg_anchor_kind = "prev_end"
            else:
                # Location changed OR cap exceeded → re-anchor to canonical location ref.
                loc_chain_len = 1
                bg_anchor_kind = "loc_ref"

            bg_anchor_path: Optional[Path] = None
            if bg_anchor_kind == "prev_end" and prev_end_image_filename:
                # Use the previous scene's end image (which is also THIS scene's start image
                # via FLF2V chaining) as the background anchor for Kontext.
                candidate = COMFY_INPUT / start_image_input_name
                if candidate.exists():
                    bg_anchor_path = candidate
            if bg_anchor_path is None and loc_ref_filename and (COMFY_INPUT / loc_ref_filename).exists():
                bg_anchor_path = COMFY_INPUT / loc_ref_filename
                bg_anchor_kind = "loc_ref"

            # ---- Composite reference for THIS scene ----
            # Kontext can only take one image, so we concatenate the relevant refs into a
            # single strip: [ bg_anchor | char1 | char2 | ... ]. bg_anchor is either the
            # canonical location ref OR the previous scene's end frame; character panels
            # lock appearance.
            ref_paths_this_scene = [
                COMFY_INPUT / char_refs[n] for n in chars_in_scene if n in char_refs
            ]
            composite_ref_name = f"composite_p{i}_{gen_id}.png"
            try:
                if ref_paths_this_scene or bg_anchor_path:
                    _composite_refs(
                        ref_paths_this_scene,
                        COMFY_INPUT / composite_ref_name,
                        location_path=bg_anchor_path,
                    )
            except Exception as e:
                print(f"[storybook] composite ref build failed for page {i+1}: {e}")
                composite_ref_name = ""

            use_kontext = (
                kontext_available
                and composite_ref_name
                and (COMFY_INPUT / composite_ref_name).exists()
            )

            # ---- Keyframe routing: img2img background-lock vs from-scratch ----
            # EVERY page renders its own end keyframe so each page advances visually — a
            # storybook page is always a distinct beat even when its internal motion is
            # small. (An earlier "pure-I2V for ambient pages" optimization skipped the
            # render and reused the previous image, which froze runs of low-motion pages
            # onto a single frame — never do that.) The only choice is *how* to render it:
            #   - same location → img2img edit of the start frame (background preserved)
            #   - location cut  → from-scratch render of the new room
            is_location_cut = i > 0 and bg_anchor_kind == "loc_ref"

            # ---- END image (Wan FLF2V target) ----
            # Kontext with the composite reference. Lead with the pose change so Kontext
            # doesn't just reproduce the reference image verbatim. The location panel of
            # the composite anchors the background; the text leads with setting so Kontext
            # also keys the *style* of the location off the reference. Switch between
            # singular and plural phrasing so Flux doesn't invent extra characters in
            # protagonist-only scenes.
            end_pose_text = ending_pose or starting_pose or "in a settled, restful pose"
            proto_clause = llm.render_canon(protagonist) or (character or "")
            # The setting-lock language differs based on what the leftmost composite panel
            # is. When it's the previous scene's end frame, we tell Kontext to literally
            # preserve the room layout — that's the strongest possible background lock.
            # When it's the canonical location ref, we phrase it as a style/setting match.
            if bg_anchor_kind == "prev_end":
                setting_lock = (
                    "Setting: this scene takes place in the EXACT same room shown in the "
                    "leftmost panel of the reference — preserve the wall colors, shelves, "
                    "windows, furniture, and prop placement; only the protagonist's pose changes. "
                )
            elif loc_clause:
                setting_lock = (
                    f"Setting (must match the location panel of the reference exactly): {loc_clause}. "
                )
            else:
                setting_lock = ""
            link_clause = (
                f"Narrative continuity: {prev_link} " if prev_link and i > 0 else ""
            )
            # Object continuity: tell Flux exactly what the protagonist is holding at the
            # END of the scene and (if this scene changes the inventory) what the change is.
            # This is the load-bearing piece for "no props appearing from thin air."
            if objects_clause:
                hold_clause = f"At the end of the scene the protagonist is holding {objects_clause}. "
            else:
                hold_clause = "At the end of the scene the protagonist's hands are empty. "
            change_clause = ""
            if object_change and object_change.lower() != "none" and i > 0:
                change_clause = f"During this scene: {object_change}. "
            if multi_char:
                end_prompt = (
                    f"{character} {end_pose_text}. "
                    f"{scene_desc}. {style_prefix}. "
                    f"{setting_lock}"
                    f"{link_clause}"
                    f"{hold_clause}{change_clause}"
                    f"Protagonist (must match the reference exactly): {proto_clause}. "
                    f"Other characters present: {scene_canon}. "
                    f"Every character keeps their appearance from the reference, but their "
                    f"poses, positions, and gestures are clearly different from the reference."
                )
            else:
                end_prompt = (
                    f"{character} {end_pose_text}. "
                    f"{scene_desc}. {style_prefix}. "
                    f"{setting_lock}"
                    f"{link_clause}"
                    f"{hold_clause}{change_clause}"
                    f"The character must match the reference exactly: {proto_clause}. "
                    f"The character's appearance is identical to the reference, but their "
                    f"pose, body position, and gesture are clearly different from the reference."
                )
            end_seed = seed + i * 100 + 7
            # From-scratch Kontext for every end keyframe. The composite reference already
            # carries the background anchor (the location ref, or the previous scene's end
            # frame for same-location continuity) plus the character panels, so Kontext keeps
            # the room and the cast while having full freedom to render THIS scene's pose.
            #
            # (We tried an img2img edit of the start frame to pixel-lock the background, but
            # at any denoise low enough to hold the room it also reproduced the pose — and
            # chaining each page off the previous page's output made a same-location run
            # converge to one frozen frame. Background continuity across clips is handled
            # instead by frame-chaining in Phase B + Wan I2V anchoring to the start frame.)
            if use_kontext:
                _route = "kontext" + (" (location cut)" if is_location_cut else "")
                wf_end = build_flux_kontext_workflow(
                    prompt=end_prompt, width=width, height=height,
                    seed=end_seed, reference_image=composite_ref_name, steps=kontext_steps,
                )
            else:
                _route = "plain-flux (no kontext)"
                wf_end = build_flux_image_workflow(ImageGenerateParams(
                    image_mode="create", prompt=end_prompt,
                    width=width, height=height, seed=end_seed, model="flux",
                ))
            print(f"[storybook] page {i+1}: {_route}")
            end_out = await _submit_comfy_and_wait(wf_end, timeout_s=300)
            shutil.copyfile(str(COMFY_OUTPUT / end_out), str(COMFY_INPUT / end_image_input_name))
            end_input_for_kf = end_image_input_name
            prev_end_image_filename = end_out
            # Bookkeeping for the next iteration's bg-anchor decision.
            prev_loc_id = loc_id
            prev_objects = list(objects_in_hand)

            # Cache per-scene context so the keyframe-regen endpoint can re-run this scene
            # individually, and so the Wan phase below has everything it needs without
            # re-deriving prompts.
            if state.active_gen is not None:
                state.active_gen.keyframes.append({
                    "scene_index": i,
                    "start_image": page_thumb,
                    "end_image": end_out,
                    "start_input_name": start_image_input_name,
                    "end_input_name": end_input_for_kf,
                    "use_flf2v": True,   # every page now renders an FLF2V end keyframe
                    "description": scene_desc,
                    "motion_intensity": intensity,
                    "start_prompt": start_prompt,
                    "end_prompt": end_prompt,
                    "end_seed": end_seed,
                    "composite_ref": composite_ref_name,   # used by keyframe-regen + drift rescore
                    "bg_anchor_kind": bg_anchor_kind,      # "loc_ref" or "prev_end"
                    "characters_in_scene": list(chars_in_scene),
                    "location_id": loc_id,
                    "location_clause": loc_clause,
                    "prev_link": prev_link,
                    "motion_timeline": motion_timeline,
                    "camera": camera,
                    "planned_camera": planned_camera,
                    "objects_in_hand": list(objects_in_hand),
                    "object_change": object_change,
                    "wan_prompt": _build_wan_prompt(
                        scene.get("video_prompt") or scene.get("motion") or "gentle motion",
                        starting_pose, ending_pose, character, p.style,
                        motion_timeline=motion_timeline,
                        camera=camera,
                        location_clause=loc_clause,
                        objects_clause=objects_clause,
                        object_change=object_change,
                    ),
                })
            step_done += 1

        # =================== CLIP drift detection ===================
        # Score every scene's start frame against the protagonist's canonical reference
        # and attach the cosine similarity to its keyframe entry, so the UI can flag
        # scenes whose character has drifted. Runs on CPU; takes a second or two.
        try:
            proto_ref_path = (
                COMFY_INPUT / char_refs[protagonist_name]
                if protagonist_name and protagonist_name in char_refs else None
            )
            scene_paths = [COMFY_OUTPUT / kf["start_image"] for kf in state.active_gen.keyframes] if state.active_gen else []
            if state.active_gen and proto_ref_path and proto_ref_path.exists() and scene_paths:
                sims = await drift.score_drift(proto_ref_path, scene_paths)
                for kf, sim in zip(state.active_gen.keyframes, sims):
                    kf["drift"] = sim   # None if scoring failed
                    kf["drift_flagged"] = (sim is not None and sim < drift.DRIFT_THRESHOLD)
        except Exception as e:
            print(f"[storybook] drift detection failed (non-fatal): {e}")

        # =================== Approval gate ===================
        # Smoke mode auto-approves so the run completes unattended.
        if state.active_gen is not None and not smoke:
            state.active_gen.node = "awaiting-approval"
            state.active_gen.step = step_done
            state.active_gen.approval_event.clear()
            state.active_gen.approval_cancelled = False
            await state.active_gen.approval_event.wait()
            if state.active_gen is None or state.active_gen.approval_cancelled:
                raise RuntimeError("storybook cancelled at preview")
        elif smoke:
            print("[storybook] SMOKE MODE: auto-approving keyframe gate, starting Wan phase")

        # =================== PHASE B — Wan animations per scene ===================
        page_videos: list[str] = []
        # The previous clip's ACTUAL last rendered frame (a filename in COMFY_INPUT). We
        # start each page from this instead of its Kontext end-keyframe so the seam is
        # invisible by construction. Reset to None at a genuine location change (a real cut),
        # where chaining the previous room's frame would be wrong.
        prev_chain_frame: Optional[str] = None
        for kf in (state.active_gen.keyframes if state.active_gen else []):
            i = kf["scene_index"]
            if state.active_gen is not None:
                state.active_gen.node = f"page-{i+1}-animate"
                state.active_gen.step = step_done

            # A page re-anchored to the canonical location ref (bg_anchor_kind == "loc_ref")
            # past page 0 is a genuine location change → hard cut, don't chain frames.
            is_location_cut = i > 0 and kf.get("bg_anchor_kind") == "loc_ref"
            if prev_chain_frame and not is_location_cut and (COMFY_INPUT / prev_chain_frame).exists():
                start_input = prev_chain_frame
            else:
                start_input = kf["start_input_name"]
            kf["chain_start_input_name"] = start_input   # persisted so regen reuses the same start

            i2v_params = GenerateParams(
                prompt=kf["wan_prompt"],
                negative=DEFAULT_NEGATIVE,
                width=width, height=height,
                frames=vid_frames, steps=vid_steps,
                cfg=6.0, shift=5.0, seed=seed,
                fps=16, scheduler="unipc",
                noise_aug=NOISE_AUG_BY_INTENSITY.get(kf["motion_intensity"], 0.05),
                image=start_input,
                # Empty for pure-I2V (ambient) pages → Wan animates from the start frame
                # only and holds its background. Set for FLF2V (beat) pages.
                end_image=kf.get("end_input_name") or "",
                multi_gpu=True,
                attention_mode="sageattn",
                block_swap_count=15, block_swap_device="cuda:1",
                vae_tiling=False,
                keep_t5_loaded=True,
                use_slg=True, use_feta=True, use_teacache=True,
            )
            wf_v = build_wan22_i2v_workflow(i2v_params)
            vid_filename = await _submit_comfy_and_wait(wf_v, timeout_s=2400)
            page_videos.append(str(COMFY_OUTPUT / vid_filename))
            kf["video"] = vid_filename   # remembered so per-scene-regen (#3) can find it later

            # Extract this clip's real last frame to chain the NEXT page onto it.
            chain_name = f"chain_last_p{i}_{gen_id}.png"
            if _extract_last_frame(COMFY_OUTPUT / vid_filename, COMFY_INPUT / chain_name):
                prev_chain_frame = chain_name
            else:
                prev_chain_frame = None   # fall back to the next page's own start keyframe
            step_done += 1

        # --- Final stitch: short crossfade between clips. ---
        # Each page starts from the previous clip's actual last frame, so adjacent boundary
        # frames already match; the crossfade smooths the residual motion-velocity seam.
        if state.active_gen is not None:
            state.active_gen.node = "stitching"
            state.active_gen.step = total
            state.active_gen.total_steps = total

        final_filename = f"wan_studio_storybook_{gen_id}.mp4"
        final_path = COMFY_OUTPUT / final_filename
        await _stitch_videos(page_videos, str(final_path))

        # Save to history. We persist the full keyframe metadata so the per-scene
        # regenerate endpoint (#3) can re-animate a single scene later without
        # re-deriving prompts, seeds, or input filenames.
        scene_records = [
            {
                "scene_index": kf["scene_index"],
                "start_image": kf["start_image"],
                "end_image": kf["end_image"],
                "start_input_name": kf["start_input_name"],
                # The frame actually fed to Wan as the start (previous clip's real last
                # frame, except at a location cut). Falls back to start_input_name.
                "chain_start_input_name": kf.get("chain_start_input_name", kf["start_input_name"]),
                "end_input_name": kf["end_input_name"],
                "use_flf2v": kf.get("use_flf2v", bool(kf["end_input_name"])),
                "video": kf.get("video"),
                "wan_prompt": kf["wan_prompt"],
                "motion_intensity": kf["motion_intensity"],
                "description": kf["description"],
                "composite_ref": kf.get("composite_ref"),
                "characters_in_scene": kf.get("characters_in_scene", []),
                "location_id": kf.get("location_id", ""),
                "location_clause": kf.get("location_clause", ""),
                "prev_link": kf.get("prev_link", ""),
                "motion_timeline": kf.get("motion_timeline", ""),
                "camera": kf.get("camera", ""),
                "objects_in_hand": kf.get("objects_in_hand", []),
                "object_change": kf.get("object_change", "none"),
                "bg_anchor_kind": kf.get("bg_anchor_kind", "loc_ref"),
            }
            for kf in (state.active_gen.keyframes if state.active_gen else [])
        ]
        locations_records = [
            {"id": l.get("id"), "name": l.get("name"), "description": l.get("description"),
             "ref_filename": loc_refs.get(l.get("id") or "", "")}
            for l in locations if (l.get("id") or "") in {r["location_id"] for r in scene_records if r.get("location_id")}
        ]
        items = load_json(HISTORY_FILE, [])
        items = [it for it in items if it.get("prompt_id") != prompt_id]
        items.insert(0, {
            "id": gen_id,
            "prompt_id": prompt_id,
            "filename": final_filename,
            "kind": "storybook",
            "mode": "storybook",
            "prompt": p.story,
            "params": {
                **p.model_dump(),
                "plan": plan,
                "scenes_meta": scene_records,
                "locations_meta": locations_records,
                "protagonist_name": protagonist_name,
                "protagonist_ref_filename": char_refs.get(protagonist_name, ""),
                "width": width,
                "height": height,
                "seed": seed,
            },
            "created_by_name": p.user_name,
            "created_by_emoji": p.user_emoji,
            "created_at": state.active_gen.started_at if state.active_gen else time.time(),
            "duration_s": time.time() - (state.active_gen.started_at if state.active_gen else time.time()),
        })
        save_json(HISTORY_FILE, items[:200])
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[storybook] FAILED: {type(e).__name__}: {e}\n{tb}")
        state.last_error = f"storybook failed: {type(e).__name__}: {e}".rstrip(": ")
    finally:
        state.active_gen = None



async def _rescore_drift_for_active() -> None:
    """Recompute CLIP drift scores against the protagonist's canonical reference for
    the currently-active storybook gen. No-op if nothing is active."""
    if state.active_gen is None or state.active_gen.kind != "storybook":
        return
    proto_ref = (state.active_gen.params or {}).get("protagonist_ref_filename") or ""
    if not proto_ref:
        # Active gen mid-run: pull from cast entries instead.
        for entry in state.active_gen.cast:
            if entry.get("role") == "protagonist":
                proto_ref = entry.get("ref_filename") or ""
                break
    if not proto_ref:
        return
    ref_path = COMFY_INPUT / proto_ref
    if not ref_path.exists():
        return
    try:
        scene_paths = [COMFY_OUTPUT / kf["start_image"] for kf in state.active_gen.keyframes]
        sims = await drift.score_drift(ref_path, scene_paths)
        for kf, sim in zip(state.active_gen.keyframes, sims):
            kf["drift"] = sim
            kf["drift_flagged"] = (sim is not None and sim < drift.DRIFT_THRESHOLD)
    except Exception as e:
        print(f"[storybook] drift rescore failed: {e}")



async def _restitch_storybook_from_history(entry: dict) -> str:
    """Read scenes_meta from a history entry, concat the per-scene videos into the
    final stitched MP4 (overwriting whatever was there). Returns the final filename."""
    final_filename = entry["filename"]
    scenes_meta = (entry.get("params") or {}).get("scenes_meta") or []
    page_video_paths = [
        str(COMFY_OUTPUT / s["video"]) for s in scenes_meta if s.get("video")
    ]
    if not page_video_paths:
        raise RuntimeError("no per-scene videos found to stitch")
    await _stitch_videos(page_video_paths, str(COMFY_OUTPUT / final_filename))
    return final_filename



async def _do_regenerate_scene(gen_id: str, scene_index: int, target: dict, params: dict):
    """Background task: re-run Wan for one scene, write the new clip into the existing
    scenes_meta slot, then re-stitch the final video."""
    NOISE_AUG_BY_INTENSITY = {"still": 0.0, "gentle": 0.05, "dynamic": 0.10}
    try:
        await llm.unload()
        i2v_params = GenerateParams(
            prompt=target["wan_prompt"],
            negative=DEFAULT_NEGATIVE,
            width=int(params.get("width") or 1024),
            height=int(params.get("height") or 576),
            frames=81, steps=20,
            cfg=6.0, shift=5.0,
            # Bump seed so the regen is actually different from the original
            seed=int(time.time() * 1000) % (2**31),
            fps=16, scheduler="unipc",
            noise_aug=NOISE_AUG_BY_INTENSITY.get(target.get("motion_intensity") or "gentle", 0.05),
            image=target.get("_resolved_start_input") or target["start_input_name"],
            end_image=target.get("end_input_name") or "",
            multi_gpu=True,
            attention_mode="sageattn",
            block_swap_count=15, block_swap_device="cuda:1",
            vae_tiling=False,
            keep_t5_loaded=True,
            use_slg=True, use_feta=True, use_teacache=True,
        )
        wf = build_wan22_i2v_workflow(i2v_params)
        new_video = await _submit_comfy_and_wait(wf, timeout_s=2400)

        # Patch the history entry: swap in the new per-scene video filename, then restitch.
        if state.active_gen is not None:
            state.active_gen.node = "stitching"
            state.active_gen.step = 2
        items = load_json(HISTORY_FILE, [])
        for it in items:
            if it.get("id") != gen_id or it.get("kind") != "storybook":
                continue
            sm = (it.get("params") or {}).get("scenes_meta") or []
            for s in sm:
                if int(s.get("scene_index", -1)) == scene_index:
                    s["video"] = new_video
                    break
            await _restitch_storybook_from_history(it)
            save_json(HISTORY_FILE, items)
            break
    except Exception as e:
        print(f"[storybook] scene regen failed: {e}")
        state.last_error = f"scene regen failed: {e}"
    finally:
        state.active_gen = None


