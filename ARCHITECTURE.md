# Fatimah Studio — Architecture

This document covers the design decisions behind the stack, how the storybook pipeline
chains together, and the tricks used to get character-consistent, motion-coherent output
on a dual-3090 home rig.

## Stack

| Layer                  | Component                                                | Why                                                                          |
|------------------------|----------------------------------------------------------|------------------------------------------------------------------------------|
| UI                     | React + Tailwind, Vite dev server (port 3000)            | Snappy, mobile-friendly, easy to make kid-readable                           |
| Backend / orchestrator | FastAPI on 8000                                          | One service that builds workflow JSON and runs the storybook pipeline        |
| Diffusion runtime      | ComfyUI on 8188                                          | Kijai's WanVideoWrapper + MultiGPU                                           |
| Video model            | Wan 2.2 14B I2V (MoE: high-noise + low-noise expert)     | SOTA quality at FP8 in 2024–2025, robust FLF2V via end-frame guidance        |
| Image model            | Flux schnell + Flux Kontext                              | Schnell is fast; Kontext locks character + setting via reference image       |
| LLM                    | Ollama: `qwen3.6:latest` (Qwen3.5-MoE 36B, Q4_K_M)       | Single model — prompt improvement *and* storybook planning. JSON-capable.    |
| Attention              | SageAttention                                            | Lossless ~30% Wan speedup on this hardware                                   |
| CLIP-vision            | `openai/clip-vit-base-patch32` (CPU)                     | Cheap drift scoring against the protagonist's canonical reference            |
| Stitching              | ffmpeg                                                   | Crossfade concat of silent Wan clips (chained on each clip's real last frame) |
| Upscale                | 4x-UltraSharp                                            | Solid photographic upscaler with low artifacts                               |
| Remote access          | NordVPN Meshnet                                          | Peer-to-peer, no public IP exposure                                          |
| LAN discovery          | avahi-daemon (mDNS)                                      | `fatimahstudio.local` resolves on the local network                          |

## Backend module layout

The backend is a small package of single-responsibility modules with a clean
dependency DAG (`config ← store/models ← workflows/comfy ← state ← storybook ← main`),
so there are no circular imports:

| Module          | Responsibility                                                          |
|-----------------|-------------------------------------------------------------------------|
| `config.py`     | ComfyUI endpoints, on-disk paths, model filenames, tunables             |
| `store.py`      | Atomic JSON load/save                                                    |
| `models.py`     | Pydantic request models                                                  |
| `workflows/wan.py`  | Wan 2.2 I2V workflow builder                                         |
| `workflows/flux.py` | Flux Kontext / Kontext-edit / Flux / SDXL / upscale builders        |
| `comfy.py`      | ComfyUI submit-and-wait + ffmpeg media helpers (thumb, last-frame, probe, stitch) |
| `state.py`      | `GenState`, the shared active-gen singleton, ComfyUI monitor + poll loops, history persistence |
| `storybook.py`  | The `_run_storybook` orchestrator + keyframe/regen/restitch/drift helpers |
| `llm/`          | Ollama package — `prompts`, `client`, `render`, `planning`               |
| `main.py`       | FastAPI app, lifespan, and all ~25 routes (thin route layer)            |

The one piece of shared mutable state — the in-flight `GenState`, the asyncio lock,
and the last error — lives in `state.py` and is accessed *qualified* (`state.active_gen`)
from every module so reassignment is visible everywhere.

## Request/response shape

The backend exposes a small, deliberately flat REST + WebSocket API. Notable endpoints:

```
POST /api/storybook                      start a storybook generation
POST /api/storybook/approve              user OK'd the keyframes → proceed to Wan
POST /api/storybook/cancel_approval      user rejected the keyframes → abort
POST /api/storybook/regenerate_keyframe  re-roll one Flux start/end frame mid-gate
POST /api/storybook/regenerate_scene     re-animate one scene's Wan clip + restitch

POST /api/image_generate    Flux/SDXL text-to-image or image-to-image
POST /api/image_upscale     run UltraSharp
POST /api/llm/improve       rewrite a short prompt with the LLM
POST /api/interrupt         cancel the currently running gen
POST /api/upload            stash an upload in ComfyUI's input/
POST /api/use_as_input      copy a previous output back into ComfyUI's input/

GET  /api/characters                 list saved characters (re-use across stories)
POST /api/characters                 save the protagonist from a finished storybook
DEL  /api/characters/{id}            forget a saved character
GET  /api/characters/{id}/image      stream the saved character's reference PNG

GET  /api/state                      current active gen + last error (polled by frontend)
WS   /api/ws/{client_id}             relay ComfyUI progress events
GET  /api/history                    list past generations
GET  /api/video|image|thumb/{name}   stream artifacts
```

The frontend polls `/api/state` every 1.5–4 s (adaptive) instead of holding a single
long WebSocket — simpler, survives connection blips, easy to inspect by hand.

## Singleton execution

The backend allows exactly **one active generation at a time**, guarded by an asyncio
lock and a `_active_gen` `GenState` object. Concurrent submissions get an HTTP 409.
Rationale: the 3090s are already saturated by a single Wan job; queueing only invites
OOM. The UI surfaces "Someone is already making something" as a friendly message.

## VRAM choreography

Two 24 GB cards. A single Wan 2.2 14B FP8 step (high or low expert) fits with
block-swap. Both experts plus the T5 encoder plus the LLM do not.

Strategy:
- **LLM unloads first.** Every diffusion endpoint awaits `llm.unload()`, which uses
  `/api/chat` with `keep_alive: 0` plus a walk of `/api/ps` to evict every loaded
  model (Ollama's `/api/generate` was observed not to reliably evict qwen3.6).
- **Block swap to cuda:1.** The model loader offloads N transformer blocks to the
  second GPU. Default is 15 blocks for storybook scenes.
- **MoE swap.** Wan 2.2's high-noise expert handles steps `0..total/2`, then is
  offloaded; the low-noise expert handles `total/2..total`. Only one expert sits on
  cuda:0 at a time.
- **T5 + CLIP-vision live on cuda:1.** Encoded once per scene; results cached on
  cuda:0 for the sampler.
- **CLIP-vit-base-patch32 stays on CPU.** Used only for drift scoring; small enough
  not to need a GPU and stays out of the way during Wan.

## Storybook pipeline

`POST /api/storybook` kicks off `_run_storybook`, a long-running async coroutine.
Five phases:

### Phase 0 — Planning (qwen3.6)

Two-pass with a critique pass:

1. **Outline** — emit a 5-beat skeleton (setup → inciting → rising → climax →
   resolution) plus a `locations` list (typically 2–5 distinct settings). Scene
   counts are scaled to match the requested total exactly.
2. **Expand** — turn the outline into N scenes. Every scene carries:
   - `location_id` → reference into the top-level `locations`
   - `starting_pose`, `ending_pose` (pose chain across scenes)
   - `prev_link` (one sentence linking to the previous scene's ending state)
   - `description`, `motion`, `video_prompt`
   - `motion_timeline` ("0-2s: ... 2-4s: ... 4-5s: ...")
   - `camera` (static / slow dolly in / slow pan left / etc.)
   - `motion_intensity` (still / gentle / dynamic)
   - `characters_in_scene` (only physically visible characters)
   - `objects_in_hand` + `object_change` (object continuity)
3. **Critique** — the same model gets the draft back with rules: check location
   teleports, object continuity (no props appearing from nowhere), prev_link
   quality, packed scenes, pose chain, motion timeline, camera. Returns a revised
   plan; the merge is field-by-field so partial-fill failures don't discard a good
   revision.

If a saved character is in play, the planner is told to lock that character verbatim
as `characters[0]` with role='protagonist'.

### Phase 0a — Character casting

One Flux model-sheet T2I per supporting character (neutral pose, plain background)
into `input/char_ref_<name>_<gen_id>.png`. The protagonist is **not** model-sheeted:
their canonical reference comes from page 1's actual scene render (see below), which
keeps the scene aesthetic instead of pulling toward studio-portrait neutrality.

If a saved character is in play, the saved PNG replaces the protagonist's model
sheet at this step.

### Phase 0b — Location casting

One Flux T2I per unique location used by any scene, into
`input/loc_ref_<id>_<gen_id>.png`. Prompt is "wide establishing shot, empty
environment, no characters" — a pure background plate. Scenes that share a location
share the ref.

### Phase A — Per-scene Flux keyframes

For each scene, in order:

1. **START frame.**
   - Page 1: plain Flux T2I from `description + starting_pose`. The result is
     copied as the protagonist's canonical reference for every later Kontext call.
   - Pages 2+: byte-perfect copy of the previous page's end image. (This seeds the
     keyframe preview; the *actual* Wan start image is later swapped for the previous
     clip's real last rendered frame — see Phase B chaining.)

2. **Composite Kontext reference.** Kontext only takes one image, so we build a
   left-to-right strip `[ bg_anchor | char1 | char2 | ... ]` where:
   - `bg_anchor` is either the canonical **location ref** (location change, or
     start of a new same-location chain), or the **previous scene's end image**
     (same location, within the chain cap). The 3-scene cap bounds compounding
     drift; after that we re-anchor.
   - The character panels lock each visible character's appearance.

3. **END frame (hybrid).** A pose-delta classifier (`_poses_differ`, plus
   `motion_intensity == dynamic`, an object change, or a location cut) decides
   whether this scene even *needs* an end keyframe:
   - **Ambient beat** (start ≈ end pose): **no end frame** — the scene animates as
     pure image-to-video and Wan holds the start frame's background.
   - **Real beat, same location:** the end frame is an **img2img Kontext edit of the
     start frame** (`build_flux_kontext_edit_workflow`, denoise `0.6`) — it samples
     from the start frame's own latent rather than empty noise, so the background is
     preserved and only the pose changes. This is the fix for "the room reshuffles
     every page."
   - **Location cut:** a from-scratch Kontext render of the new room.

   In every case the prompt leads with the pose change, then setting-lock language,
   then `prev_link`, then the object clause.

4. **Cache.** Every scene's prompt, seed, composite ref name, keyframe filenames,
   motion timeline, camera, objects, and assembled Wan prompt are stashed on the
   active `GenState` for later use.

### Phase A1 — CLIP drift scoring

CPU-side cosine similarity between the protagonist's canonical reference and every
scene's start frame. Scenes scoring below `DRIFT_THRESHOLD = 0.82` get a
`drift_flagged: true` on their keyframe entry for the UI to badge.

### Phase A2 — Approval gate

The orchestrator sets `node = "awaiting-approval"` and awaits an
`asyncio.Event`. The frontend renders the full keyframe strip with drift badges; the
user can:

- Hit **approve** → proceed to Wan.
- Hit **regenerate** on any keyframe → re-runs Flux Kontext with a new seed,
  updates the cache in place, and re-scores drift. If the regen is an end-frame, it
  propagates to the next scene's start to keep FLF2V chaining byte-perfect.
- Hit **cancel** → abort cleanly without any Wan work.

### Phase B — Wan animation

Scenes are animated **sequentially** so each can chain on the previous one. Per scene,
Wan 2.2 14B I2V:

- `image` = the previous clip's **actual last rendered frame** (extracted with
  `ffmpeg -sseof -1`), not the Kontext keyframe. Wan's FLF2V undershoots its end
  target, so chaining on the *rendered* frame is what makes the seam invisible by
  construction. (Page 1, and the page right after a location cut, start from their
  own keyframe.)
- `end_image` = the page's end frame for **FLF2V** beats; **empty** for ambient
  beats, which run as pure I2V and hold the background.
- **Camera is forced static** (`Camera: locked, no movement`). Independent per-clip
  dolly/pan moves end at one framing and the next clip starts a different move from a
  different framing — a major source of seam pops — so they're stripped (the LLM's
  intended camera is still recorded as `planned_camera`).
- 81 frames @ 16 fps ≈ 5 s (Wan 2.2's trained sweet spot); smoke mode uses 33.
- `noise_aug` is selected from `motion_intensity` (still 0.0, gentle 0.05, dynamic 0.10)
- Wan prompt is the assembled string: video_prompt + pose chain + setting +
  timeline + locked-camera + objects-held clause + "background remains stable" tail.

Quality flags on by default: `use_slg`, `use_feta`, `use_teacache`. SageAttention.
Block swap to cuda:1.

### Phase C — Stitch

`ffmpeg xfade` of the silent Wan clips with a short ~0.18 s crossfade per seam
(`XFADE_DUR`). Because each clip already starts on the previous clip's real last
frame, the boundary frames match; the crossfade just absorbs the residual
motion-velocity discontinuity so the action eases across the cut instead of
stop-starting. Output: `wan_studio_storybook_<gen_id>.mp4`.

The history entry persists: the full plan, `scenes_meta` (every keyframe context,
including `use_flf2v` and the chained start frame), `locations_meta`, the
protagonist's reference filename. This is enough to drive the per-scene Wan-regen
flow without re-running anything else.

### Smoke-test mode

`StorybookParams.smoke = true` runs the *real* pipeline end to end but fast: capped
to `SMOKE_PAGES` (3), Wan at 8 steps / 33 frames, Kontext at 10 steps, and the
keyframe approval gate auto-approved so it completes unattended in a few minutes
instead of hours. The backend logs each page's keyframe route, and `smoke_test.py`
posts the job, watches `/api/state`, and extracts the frames either side of every
seam into `output/smoke_seams_<id>/`. Purpose: validate the background-lock, the
frame chaining, and the crossfade before committing to a full-length run.

## Background continuity

The default failure mode of generating each Kontext scene from empty noise is that
the background reshuffles every page — the kitchen Bolt was just in becomes a
*different* kitchen on the next page (a fridge appears, jars rearrange). Three
mechanisms, in order of impact, keep the room stable:

- **img2img end keyframes (the load-bearing fix).** For a same-location beat, the end
  frame is generated by `build_flux_kontext_edit_workflow`, which seeds the sampler
  from the start frame's *own* latent at denoise `0.6` instead of empty noise. The
  existing composition — crucially the background — survives; only the character's
  pose moves. The old path sampled from `EmptySD3LatentImage` at denoise `1.0` and
  re-invented the whole room every call.
- **Prev-end as the background anchor.** The leftmost panel of the Kontext composite
  is scene N-1's end image when the location is unchanged, with hard "preserve the
  EXACT room" prompt language. Re-anchoring to the canonical location ref now happens
  **only on a genuine location change** (the old 3-scene chain cap that forced a
  periodic re-render — and a periodic background pop — is removed).
- **Frame chaining into Wan.** Even with matched keyframes, the *rendered* clips are
  what the viewer sees, so each clip starts from the previous clip's real last frame
  (Phase B) and Wan I2V holds that background forward.

## Object continuity

Each scene has `objects_in_hand` (list of object names at the END of the scene) and
`object_change` (`none` / `picks up X` / `puts down X` / `swaps X for Y`).

- The planner's hard rule: `scene[i].objects_in_hand` may differ from
  `scene[i-1].objects_in_hand` only if the description visibly shows a pickup,
  putdown, or swap.
- The critique pass enforces this in revision: a tray of cookies cannot appear in
  scene N's description if no earlier scene shows them being made.
- Post-LLM repair: if the diff is inconsistent (objects changed but
  `object_change == "none"`), we auto-fill the verb so the prompts at least narrate
  the change.
- Both Flux end prompt and Wan prompt embed the held-object clause: Flux says
  "At the end of the scene the protagonist is holding X"; Wan says "the character
  is holding X the entire shot; the object does not morph or change."

## Character consistency

The chain that produced kid-readable continuity, in order of impact:

1. **Page 1's render as the protagonist's canonical reference.** A plain T2I render
   of the first scene, *not* a neutral model sheet — using the real scene aesthetic
   prevents Kontext from pulling toward studio-portrait neutrality.
2. **Composite Kontext ref with all visible characters.** Side-by-side strip of
   per-character refs (one panel per character) so multi-character scenes carry
   every face into Kontext.
3. **Last-frame chaining.** Each Wan clip starts from the previous clip's *actual
   last rendered frame*, so there is no re-illustration gap and the character carries
   forward exactly across the seam.
4. **CLIP drift scoring.** Cheap CPU check that flags scenes where the character
   drifted from the canonical look; the UI exposes a regen button for those scenes.
5. **Saved-character library.** A finished storybook's protagonist can be saved to
   `characters.json`; subsequent stories lock it via Kontext from page 1.

## Wan 2.2 quality knobs

| Flag           | Class                    | What it does                                            |
|----------------|--------------------------|---------------------------------------------------------|
| `use_slg`      | `WanVideoSLG`            | Skip Layer Guidance, sharper output                     |
| `use_feta`     | `WanVideoEnhanceAVideo`  | Enhance-A-Video — smoother motion, weight 2.0           |
| `use_teacache` | `WanVideoTeaCache`       | Caches denoising deltas, ~1.5–2× speedup, minor cost    |
| `sageattn`     | `attention_mode`         | SageAttention — lossless ~30% speedup                   |

All wired into both the high-noise and low-noise samplers. Frame count is 81 for a
full run (Wan 2.2's trained sweet spot for ~5 s); smoke mode drops to 33 frames /
8 steps. `KONTEXT_EDIT_DENOISE` (0.6) and `XFADE_DUR` (0.18 s) are the two visual
tunables for the background-lock strength and the seam crossfade.

## LLM behavior

A single Ollama model (`qwen3.6:latest` — a Qwen3.5-MoE 36B, Q4_K_M) handles both
prompt improvement and storybook planning. It's heavier than a small JSON model but
writes noticeably better prompts and plans; the unload protocol keeps it out of VRAM
during diffusion.

- **Context length** is *not* set by the backend, so Ollama loads the model at its
  default — the full **262 144 (256 K)** native window. Input never truncates;
  planning prompts are a few KB. (Ollama doesn't pre-allocate the KV cache, so the
  big default costs no extra VRAM in practice.)
- **Output cap** (`num_predict`) is set per call and scales with page count, sized so
  the plan/retry/critique passes — each of which re-emits the *whole* plan — don't
  truncate. Plan and retry use `~3072–4096 + 256·(pages−6)`; the critique uses the
  same headroom (it returns a full revised plan). A truncated/invalid critique falls
  back to the un-critiqued draft rather than failing.

Server-side post-processing on every plan:
- `coerce_characters` / `coerce_locations` normalise both arrays and assign
  defaults if the model dropped fields.
- Pose-chain enforcement: `scene[i].starting_pose` is overwritten with
  `scene[i-1].ending_pose` if the model drifted.
- `characters_in_scene` is filtered to known names with the protagonist always
  prepended.
- `location_id`s are slug-normalised; unknown ids fall back to the previous scene's
  location id (preserves continuity rather than snapping to the first location).
- `objects_in_hand` is normalised; `object_change` is auto-filled when the
  in-hand diff is inconsistent.
- A scene-1 `prev_link` of "Opening scene." is forced.
- JSON repair handles a known qwen3 glitch where it sometimes emits
  `{"": "...", "motion": "..."}` (empty-string key) → maps to `description`.

`qwen3` system messages append `/no_think` and the API payload sets `think: False`
to skip the chain-of-thought block (faster, cleaner JSON).

## Cancellation

`POST /api/interrupt` calls ComfyUI's `/interrupt` and clears `_active_gen`. The
storybook orchestrator wraps every long step in an "is cancelled?" check and exits
cleanly without leaving half-rendered scenes in history. The approval gate also
honours cancel via `cancel_approval`.

## History

`history.json` lives next to `main.py`. Each storybook entry records:

- The final stitched filename
- The full plan (with characters, locations, every scene's metadata)
- `scenes_meta` — every keyframe's filenames, prompts, motion timeline, camera,
  objects, composite ref, drift score
- `locations_meta` — every location id + ref filename actually used
- `protagonist_name` and `protagonist_ref_filename` — for the save-character flow
- Width / height / seed
- Avatar attribution + duration

This is enough to drive per-scene Wan-regen without re-running the LLM or Flux.

Capped at 200 entries.

The history drawer renders thumbnails; for videos these are extracted on first
request via ffmpeg and cached as JPG in `studio/backend/thumbs/`.

## Frontend conventions

- **One context, no Redux.** `StudioProvider` holds the small global state (server
  poll results, current result, history, theme, profile). Each panel keeps its own
  local form state.
- **Adaptive polling.** `/api/state` every 1.5 s while a gen is active, 4 s when
  idle.
- **Friendly progress messages.** `ProgressDisplay.tsx` translates internal node
  names (`page-3-image`, `page-3-animate`, `casting-Mochi`, `location-kitchen`,
  `awaiting-approval`, `stitching`, …) into kid-readable strings.
- **Approval gate UI.** When `node == "awaiting-approval"`, the output panel
  switches to a strip of keyframe pairs with regen buttons and drift badges; the
  user clicks Approve to release the Wan phase.
- **Avatar attribution.** A first-visit modal asks for a name + emoji; this is
  stored in `localStorage` and sent with every job.

## Why no public exposure?

The frontend is the only thing listening on `0.0.0.0` and the backend / ComfyUI /
Ollama all stay on `127.0.0.1`. Remote access goes through Meshnet — peer-to-peer,
encrypted, no ports opened to the public internet. Every device that wants access
needs Meshnet installed, but for a family use-case that's a feature.

## Things deliberately left simple

- No multi-user queueing. One generation at a time, family-scale.
- No accounts. Just an avatar in `localStorage`.
- No cloud storage. Everything is on the box. History caps at 200 entries; old
  artifacts are also pruned from the ComfyUI `output/` folder when the history
  entry rolls off.
- No streaming previews of frames in flight. The keyframe approval gate gives the
  user a *much* better preview than a streaming sampler thumbnail would.
