# Fatimah Studio — Architecture

This document covers the design decisions behind the stack, how the storybook pipeline
chains together, and the tricks used to get character-consistent, motion-coherent output
on a dual-3090 home rig.

## Stack

| Layer                  | Component                                             | Why                                                                    |
|------------------------|-------------------------------------------------------|------------------------------------------------------------------------|
| UI                     | React + Tailwind, Vite dev server (port 3000)         | Snappy, mobile-friendly, easy to make kid-readable                     |
| Backend / orchestrator | FastAPI on 8000                                       | One service that builds workflow JSON and runs the storybook pipeline  |
| Diffusion runtime      | ComfyUI on 8188                                       | Kijai's WanVideoWrapper + MultiGPU + frame-interpolation nodes         |
| Video model            | Wan 2.2 14B I2V (MoE: high-noise + low-noise expert)  | SOTA quality at FP8 in 2024–2025, robust FLF2V via end-frame guidance  |
| Image model            | Flux schnell + Flux Kontext                           | Schnell is fast; Kontext locks character identity across pages         |
| LLM                    | Ollama: `qwen3.6:latest` + `qwen3:8b`                 | Local, JSON-capable, unloads from VRAM before Wan starts               |
| TTS                    | Kokoro 82M (CPU)                                      | Tiny, expressive, runs on CPU so it doesn't fight GPUs                 |
| Stitching              | ffmpeg + ffprobe                                      | Hard cuts + `tpad` hold per page + per-scene audio mux                 |
| Upscale                | 4x-UltraSharp                                         | Solid photographic upscaler with low artifacts                         |
| Remote access          | NordVPN Meshnet                                       | Peer-to-peer, no public IP exposure                                    |
| LAN discovery          | avahi-daemon (mDNS)                                   | `fatimahstudio.local` resolves on the local network                    |

## Request/response shape

The backend exposes a small, deliberately flat REST + WebSocket API. Notable endpoints:

```
POST /api/storybook         start a storybook generation (planning → pages → stitch)
POST /api/image_generate    Flux/SDXL text-to-image or image-to-image
POST /api/image_upscale     run UltraSharp
POST /api/llm/improve       rewrite a short prompt with the small LLM
POST /api/interrupt         cancel the currently running gen
POST /api/upload            stash an upload in ComfyUI's input/
GET  /api/state             current active gen + last error (polled by frontend)
WS   /api/ws/{client_id}    relay ComfyUI progress events
GET  /api/history           list past generations
GET  /api/video|image|thumb/{filename}  stream artifacts
```

The frontend polls `/api/state` every 1.5–4 s (adaptive) instead of holding a single long
WebSocket — simpler, survives connection blips, easy to inspect by hand.

## Singleton execution

The backend allows exactly **one active generation at a time**, guarded by an asyncio lock
and a `_active_gen` `GenState` object. Concurrent submissions get an HTTP 409. Rationale:
the 3090s are already saturated by a single Wan job; queueing only invites OOM. The UI
surfaces "Someone is already making something" as a friendly message.

The active job's progress is tracked by `_monitor_loop` (subscribes to ComfyUI's
WebSocket as a known `MONITOR_CLIENT_ID`) plus `_queue_poll_loop` (polls
`/queue` + `/history`) — the combination handles the edge cases where ComfyUI's WS
silently drops a `done` event.

## VRAM choreography

Two 24 GB cards. A single Wan 2.2 14B FP8 step (high or low expert) fits with block-swap.
Both experts plus the T5 encoder plus the LLM do not.

Strategy:
- **LLM unloads first**. Every diffusion endpoint awaits `llm.unload()` (Ollama's
  `keep_alive: 0` API) before queueing the ComfyUI job.
- **Block swap to cuda:1**. The model loader offloads N transformer blocks to the second
  GPU. Default is 15 blocks for storybook (832×480), 20 for higher resolutions.
- **MoE swap**. Wan 2.2's high-noise expert handles steps `0..total/2`, then is offloaded;
  the low-noise expert handles `total/2..total`. Only one expert is on cuda:0 at a time.
- **T5 + CLIP-vision lives on cuda:1**. Encoded once at the start of each scene; results
  cached on cuda:0 for the sampler.
- **Kokoro stays on CPU**. 82 M params, decoding a few seconds of audio per page — no
  reason to compete with the GPUs.

## Storybook pipeline

`POST /api/storybook` kicks off `_run_storybook`, which is a long-running async coroutine.
The flow per page:

```
1. Plan       qwen3.6  → JSON with N scenes
                          { character, scenes:[{ starting_pose, ending_pose,
                            description, motion, video_prompt, narration }] }
              [continuity guarantee: scene[i].starting_pose ≡ scene[i-1].ending_pose,
               enforced server-side after the model returns]

2. Per page i, in sequence:

   (a) START frame
       i == 0  → Flux schnell from scratch using `description + starting_pose`
                 Save the output as the character canonical reference for Kontext.
       i >= 1  → byte-perfect copy of page (i-1)'s END frame
                 (this is the key to seamless inter-page continuity)
       Copy the resulting PNG into ComfyUI's input/ as `storybook_start_pN_<gen>.png`.

   (b) END frame
       Flux Kontext, conditioned on the canonical character reference,
       prompted with `description + ending_pose`.
       Copy into input/ as `storybook_end_pN_<gen>.png`.

   (c) Animate (Wan 2.2 14B I2V, FLF2V mode)
       start_image = the page's start frame
       end_image   = the page's end frame
       49 frames @ 16 fps = ~3 s of motion that lands exactly on the end pose.
       Quality flags on by default: SLG, FETA (Enhance-A-Video), TeaCache.

   (d) Narrate (Kokoro, voice = bf_emma, speed 0.9)
       Synthesize the page's narration text to WAV.

   (e) Mux scene
       ffmpeg: video → 1.5 s `tpad` hold on the final frame → mux audio.
       If narration is longer than the video, the hold extends to cover it.

3. Stitch all per-scene MP4s with ffmpeg concat. Save under `wan_studio_storybook_<id>.mp4`.
```

Live progress: each step bumps `_active_gen.step` and `_active_gen.node`
(`page-N-image`, `page-N-animate`, `narration-N`, `stitching`). The frontend renders a
thumbnail strip from `preview_images`, which holds the **start frame of each page**
(byte-perfect equal to the previous page's end frame, so the strip reads as a chained
keyframe storyboard with exactly N thumbnails for N pages).

## Character consistency strategy

This took a few iterations:

1. **Same-seed prompting** — drifted noticeably across pages.
2. **Flux Redux (style/IP-Adapter-style conditioning)** — better, but still drifted.
3. **Flux Kontext + canonical reference** — page 1 generated from scratch, then saved as
   the character reference for every subsequent page. Currently shipped.
4. **FLF2V chaining on top of (3)** — each page's start frame is a byte-perfect copy of
   the previous page's end frame, so Wan's first frame is **identical** to the previous
   page's last frame. No re-illustration gap between pages.

The result: kid-readable continuity. The same character moves through the story without
"why does she look different now" moments.

## Wan 2.2 quality knobs (default-on for storybook)

| Flag         | Class                     | What it does                                            |
|--------------|---------------------------|---------------------------------------------------------|
| `use_slg`    | `WanVideoSLG`             | Skip Layer Guidance, 0.1–0.5 of denoising — sharper.    |
| `use_feta`   | `WanVideoEnhanceAVideo`   | Enhance-A-Video — smoother motion, weight 2.0.          |
| `use_teacache` | `WanVideoTeaCache`      | Caches denoising deltas, ~1.5–2× speedup, minor cost.   |

All three are wired into **both** the high-noise and low-noise samplers via
`slg_args`/`feta_args`/`cache_args`. Frame count is fixed at 49 (Wan 2.2's trained sweet
spot) — variable per-page length was tested and rejected because drifting outside the
training distribution costs more quality than it buys.

## Transitions: the "kid reading a book" decision

Several transition styles were tried and rejected for storybook output:

- **RIFE frame morph** — dreamy melt, identity drift.
- **Wipe right** — jarring, breaks immersion.
- **Crossfade** — wrong vibe, looks like a slideshow.

**Final: hard cut + 1.5 s `tpad` hold on each page's final frame.** This matches the pacing
of a parent flipping pages in a book and gives the narration time to land. It's also free
(no extra inference) and survives narration durations that vary wildly per page.

## LLM behavior

Two models, picked per task:

- **`qwen3:8b`** — small, fast cold-load. Used by the "Improve prompt" button in the UI.
  Fits the snappy-UX requirement.
- **`qwen3.6:latest` (23 GB)** — used for storybook planning, where reasoning matters
  (continuity, scene structure, narration tone). Loaded only for the planning step, then
  unloaded before Wan starts.

A pose-chain validator runs server-side after each LLM response: `scene[i].starting_pose`
is overwritten with `scene[i-1].ending_pose` if the model drifted. There's also a
JSON-repair pass for a known qwen3 glitch where it sometimes emits `{"": "...", "motion":
"..."}` (empty-string key) — the repair maps that to `description`.

`qwen3` prompts append `/no_think` to the system message and set `think: False` in the
API payload to skip the chain-of-thought block (faster + cleaner JSON).

## Cancellation

`POST /api/interrupt` calls ComfyUI's `/interrupt` and clears `_active_gen`. The storybook
orchestrator wraps every long step in an "is cancelled?" check (the orchestrator notices
`_active_gen` becoming `None`) and exits cleanly without leaving half-rendered scenes in
history.

## History

`history.json` lives next to `main.py`. Each entry records the filename of the final
artifact, the prompt, full parameter blob, who made it (avatar name + emoji from the
profile cookie), and duration. Capped at 200 entries; the oldest fall off.

The history drawer renders thumbnails — for videos these are extracted on first request
via ffmpeg and cached as JPG in `studio/backend/thumbs/`. Hovering plays a muted preview.

## Frontend conventions

- **One context, no Redux.** `StudioProvider` holds the small global state (server poll
  results, current result, history, theme, profile). Each panel keeps its own local form
  state.
- **Adaptive polling.** `/api/state` every 1.5 s while a gen is active, 4 s when idle.
- **Friendly progress messages.** `ProgressDisplay.tsx` translates internal node names
  (`sampler`, `clip_vision_encode`, `page-3-animate`, …) into kid-readable strings like
  "Animating page 3…". Sampler progress shows a cycling status ("Sketching the scene…",
  "Painting in the details…", …).
- **Avatar attribution.** A first-visit modal asks for a name + emoji; this is stored in
  `localStorage` and sent with every job so the history drawer can credit the creator.

## Why no public exposure?

The frontend is the only thing listening on `0.0.0.0` and the backend / ComfyUI / Ollama
all stay on `127.0.0.1`. Remote access goes through Meshnet — peer-to-peer, encrypted, no
ports opened to the public internet. The trade-off is that every device that wants access
needs Meshnet installed, but for a family use-case that's a feature, not a friction.

## Things deliberately left simple

- No multi-user queueing. One generation at a time, family-scale.
- No accounts. Just an avatar in `localStorage`.
- No cloud storage. Everything is on the box. History caps at 200 entries; old artifacts
  are also pruned from the ComfyUI `output/` folder when the history entry rolls off.
- No streaming previews of frames in flight. ComfyUI doesn't make it easy; the live
  thumbnail strip of completed pages turned out to be enough storytelling.
