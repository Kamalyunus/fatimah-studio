# Fatimah Studio

A local, family-friendly AI creative studio for generating illustrated storybook
movies and pictures — built on top of ComfyUI with Wan 2.2, Flux schnell, Flux
Kontext, and a local Ollama LLM. No cloud, no API keys, no per-generation cost;
runs entirely on a home machine with dual NVIDIA GPUs.

The headline feature is **Storybook Movie Maker**: type a one-line idea, and the
studio writes the story, records a warm narrator reading it, illustrates every page with
consistent characters and locked backgrounds, animates each page, and cuts the whole
thing to the voice track — delivering a narrated 24fps MP4 you can watch on the couch.

> Built for my family. The UI is simple enough for kids; the pipeline underneath
> is tuned for quality and character coherency.

## Highlights

- **Storybook Movie Maker** — story → narration recorded first → planned scenes
  (locations, characters, beats) → illustrated keyframes → animatic → best-of-N animated
  takes → interpolated, graded and mixed into one narrated MP4. One unattended run.
- **Narration-first timing** — the LLM writes real picture-book prose for each page, a
  local voice reads it, and each page's clip length is cut to fit its line. The film is
  cut to the voice, the way an animatic is, instead of every page being a flat 5 seconds.
- **Animatic preview** — as soon as the keyframes exist, the stills are cut against the
  narration into a watchable preview of the whole story. It appears in the UI minutes
  in, while the hours-long animation phase is still running.
- **Best-of-N takes** — every page is drafted several times at low cost and scored on
  character identity, whether it lands the intended end pose, and motion sanity; only
  the winning seed gets a full-quality render. Generating once and hoping is what made
  the old output mediocre.
- **Shot variety** — pages carry a wide / medium / close shot size; a change of framing
  deliberately breaks the frame chain and re-frames the scene, which is what stops a
  whole location playing as one locked-off take.
- **Real finishing chain** — RIFE interpolation to 24fps, per-shot colour matching
  toward the run's median look, film grain, hard cuts within a location and dissolves
  only where the place changes, then a loudness-normalised mix (with optional ducked
  music bed).
- **Picture maker** — Flux schnell text-to-image and image-to-image at up to
  1280×768.
- **Photo enhancer** — 4× upscale with 4x-UltraSharp.
- **Character consistency** — Flux Kontext locks the protagonist's appearance
  across all scenes; supporting characters get model-sheet refs; each clip starts
  from the *actual last rendered frame* of the previous clip so pose continuity is
  exact. CLIP-vision drift detection flags scenes where the character has drifted.
- **Layout-locked keyframes** — a page's end keyframe is generated as a Kontext
  *edit of its own start frame* ("same room, same lighting, only the pose changes"),
  so start and end agree on the room by construction and Wan never has to morph the
  background mid-clip. From-scratch rendering is reserved for genuine location cuts
  and pages where a new character enters.
- **Cast lock during animation** — every Wan prompt states that only the characters
  already visible may appear, backed by a storybook negative prompt that pushes hard
  against extra people, crowds, and bystanders wandering through a shot.
- **Background continuity** — locations are first-class entities. Each scene's
  Kontext keyframe is anchored on a consistent background panel (the location ref,
  or the previous scene's end frame for same-location runs), and the *animated*
  continuity is carried by frame-chaining: every Wan clip starts from the previous
  clip's real last frame and Wan's I2V holds that background forward — so the
  kitchen Bolt is in stays the same kitchen across the cut.
- **Seamless cuts** — pages chain on the previous clip's real last frame and stitch
  with a short crossfade, so the page-to-page seams read as one continuous shot
  instead of popping. The camera is locked per clip (no per-shot dolly/pan that
  would jump at every cut).
- **Object continuity** — every scene declares what the protagonist is holding;
  the LLM critique pass refuses plans where props appear from thin air.
- **Character library** — save a protagonist from a finished storybook and re-use
  it as the locked main character of a future story.
- **Per-scene regenerate** — re-animate a single Wan scene after the storybook is
  done, without redoing the rest.
- **Fast smoke-test mode** — a `smoke` flag runs a short, low-step 3-page
  generation in a few minutes, so the coherency pipeline can be eyeballed before
  committing to a full-length, full-quality run.
- **Quality stack on by default** — Wan 2.2 14B MoE (two-expert I2V chain) with
  Skip Layer Guidance, Enhance-A-Video, TeaCache, and SageAttention. (A VACE T2V
  route ships behind the `USE_VACE` flag, off by default — it holds character
  identity well but under-constrains the middle of each clip; see ARCHITECTURE.md.)
- **Dual-GPU aware** — block-swap to a second GPU keeps each 24 GB card from
  running out of VRAM. LLM unloads from VRAM before diffusion starts.
- **Local & private** — Ollama (`qwen3.6:latest` for everything LLM), no
  telemetry, no external calls.
- **Friendly UI** — React + Tailwind, mobile-friendly, kid-readable. Avatar
  picker for attribution. Live thumbnail strip during illustration so you can watch
  the story take shape.
- **Remote access** — mDNS hostname for the LAN; NordVPN Meshnet for "from
  anywhere" without exposing anything to the public internet.

## Architecture at a glance

```
                  ┌────────────────────────────────────────────────────┐
   browser ─────► │ React + Tailwind UI  • Storybook tab • Picture tab │
                  │ static build, served by the backend (no dev server)│
                  └─────────────────────┬──────────────────────────────┘
                                        │ same origin, /api/*
                                        ▼
                  ┌────────────────────────────────────────────────────┐
                  │ FastAPI backend (8000) — serves the built UI + /api│
                  │   • workflows/  builds ComfyUI workflow JSON       │
                  │   • storybook   orchestrates the pipeline          │
                  │   • comfy/state submit + track the active gen      │
                  │   • llm/        Ollama planner & prompt-improve     │
                  └─────┬─────────────────┬──────────────────────────┬─┘
                        │                 │                          │
                        ▼                 ▼                          ▼
                ┌──────────────┐  ┌──────────────┐         ┌──────────────────┐
                │ ComfyUI 8188 │  │ Ollama 11434 │         │ CLIP-vit (CPU)   │
                │ Wan 2.2 I2V  │  │ qwen3.6      │         │ drift scoring    │
                │ Flux schnell │  │              │         │                  │
                │ Flux Kontext │  │              │         │                  │
                │ Upscaler     │  │              │         │                  │
                └──────────────┘  └──────────────┘         └──────────────────┘
```

For the deeper walkthrough of the storybook pipeline, the start-frame-edit keyframe
trick, and VRAM choreography, see
[ARCHITECTURE.md](ARCHITECTURE.md).

## Hardware

Currently tuned for and tested on:

- 2× NVIDIA RTX 3090 (48 GB total VRAM)
- ~100 GB free disk for model weights (Wan 2.2 + Flux + supporting encoders)
- Linux (Ubuntu / Pop!_OS), Python 3.10+, Node 20+

It will work on a single 24 GB card with quality reduced (lower resolution,
smaller block swap count). It will not work comfortably below 16 GB.

## Repo layout

```
studio/
├── backend/
│   ├── main.py          FastAPI app + all HTTP/WS routes (thin route layer)
│   ├── config.py        Endpoints, paths, model filenames, tunables
│   ├── models.py        Pydantic request models
│   ├── store.py         Atomic JSON load/save
│   ├── comfy.py         ComfyUI submit + ffmpeg media helpers (thumb, stitch, …)
│   ├── state.py         GenState + active-gen singleton + ComfyUI monitor loops
│   ├── storybook.py     Storybook orchestrator (plan → keyframes → Wan → stitch)
│   ├── workflows/       Builders — wan.py (I2V+VACE), flux.py (Flux/SDXL), video_post.py (RIFE)
│   ├── tts.py           Kokoro narration synthesis (CPU)
│   ├── takes.py         Draft-take scoring + selection
│   ├── finishing.py     Animatic, conform, grade, grain, narration/music mix
│   ├── narration/       Per-page narration WAVs (auto-created)
│   ├── llm/             Ollama package — prompts, client, render, planning
│   ├── drift.py         CLIP-vision drift scoring (CPU)
│   ├── smoke_test.py    Fast end-to-end smoke runner (smoke-mode storybook)
│   ├── characters.json  Saved-character library (auto-created)
│   ├── character_refs/  PNG refs for the saved-character library (auto-created)
│   ├── history.json     Local history of generations (auto-created)
│   └── thumbs/          JPG thumbnails for the history drawer (auto-created)
└── frontend/
    ├── src/
    │   ├── App.tsx                 Tabbed shell (Storybook + Picture)
    │   ├── components/
    │   │   ├── StorybookPanel.tsx     story input → duration → submit
    │   │   ├── ImagePanel.tsx         create / modify
    │   │   ├── HomeScreen.tsx         landing tiles
    │   │   ├── OutputPanel.tsx        result + thumbnail strip + per-scene regen
    │   │   └── ProgressDisplay.tsx    friendly progress messaging
    │   ├── lib/{api,store,presets,user,utils}.ts
    │   └── types.ts
    ├── vite.config.ts              build config (dev-server proxy kept for `npm run dev`)
    └── dist/                       production build — `npm run build`; served by the backend at :8000
```

## Prerequisites

1. **ComfyUI** installed at `/home/yunus/Documents/comfyui` with these custom nodes:
   - [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (Kijai)
   - [ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU)
   - SageAttention (compiled wheel or pip)
2. **Model weights** on a large drive (mine: `/media/yunus/More Data/comfyui-models/`),
   wired up via `extra_model_paths.yaml`:
   - Wan 2.2 I2V high + low noise FP8 (`wan2.2_i2v_*_14B_fp8_scaled.safetensors`)
   - Wan 2.2 T2V A14B high + low FP8-scaled (`Wan2_2-T2V-A14B*_fp8_e4m3fn_scaled_KJ.safetensors`)
   - Wan 2.2 Fun-VACE modules high + low (`Wan2_2_Fun_VACE_module_A14B_*_fp8_e4m3fn_scaled_KJ.safetensors`)
   - Wan 2.1 VAE + UMT5 XXL text encoder + Wan CLIP-vision
   - Flux schnell + Flux Kontext + Flux T5 + Flux CLIP-L + Flux VAE
   - SDXL base + SDXL Lightning LoRA (optional, fallback image model)
   - 4x-UltraSharp upscaler
3. **Ollama** running on `127.0.0.1:11434` with:
   - `qwen3.6:latest` (~23 GB — used for prompt improvement *and* storybook planning)
3b. **Narration + interpolation**:
   - `kokoro` (pip, ~330 MB of weights on first use) for the narrator voice
   - [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)
     for RIFE — the `rife47.pth` checkpoint downloads on first use
4. **Python deps**: `fastapi`, `uvicorn`, `httpx`, `websockets`, `pydantic`,
   `pillow`, `transformers`, `torch`. (Living in `/home/yunus/Documents/comfyui/venv/`.)
5. **System**: `ffmpeg`, `ffprobe`, `avahi-daemon` (for `fatimahstudio.local` mDNS).

## Running locally

ComfyUI must already be up on port 8188 with the custom nodes loaded. Then:

```bash
# Frontend — build once (re-run after any UI code change; there is no hot reload)
cd studio/frontend
npm install
npm run build            # → frontend/dist/

# Backend — serves the built UI *and* the /api routes on a single port
cd ../backend
/home/yunus/Documents/comfyui/venv/bin/python main.py    # binds 127.0.0.1:8000
```

Open `http://localhost:8000/`. The backend serves the compiled frontend and the
`/api` routes from the same origin, so there is **no separate dev-server process** to
run or keep alive. (Editing UI code? Re-run `npm run build`; the backend picks up the
new files immediately, no restart needed. For live hot-reload while developing, you
can still `npm run dev` — it proxies `/api` to `:8000` on its own port.)

(On my box, the backend is a `fatimah-backend.service` user unit that depends on
the `More Data` drive being mounted via fstab so the model weights are reachable.)

### Smoke test

With the backend, ComfyUI, and Ollama all up, a fast end-to-end check (3 short pages,
low steps — a few minutes instead of hours) validates the coherency
pipeline and drops seam frames for inspection:

```bash
cd studio/backend
/home/yunus/Documents/comfyui/venv/bin/python smoke_test.py
```

It posts a `smoke: true` storybook, watches progress, then writes the seam frames to
`output/smoke_seams_<id>/`. The backend log prints each page's keyframe route
(`kontext` / `kontext (location cut)` / `plain-flux`).

## Remote access (optional)

By default everything binds `127.0.0.1` (local-only) — the backend that serves the UI
on `:8000`, ComfyUI, and Ollama. For LAN or "from anywhere" access, bind the backend
to all interfaces: change the `uvicorn.run(..., host="127.0.0.1")` line at the bottom
of `backend/main.py` to `host="0.0.0.0"` (or launch it as
`uvicorn main:app --host 0.0.0.0 --port 8000`). Then reach it over **NordVPN Meshnet**
(free, peer-to-peer, nothing exposed to the public internet): enable Meshnet on this
machine and any phone/laptop, then visit `http://<meshnet-ip>:8000/`.

## Status

Personal project, actively used by my family. No external contributions expected,
but the code is MIT-friendly and the architecture doc should make forks
straightforward.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and pipeline details.
