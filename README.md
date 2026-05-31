# Fatimah Studio

A local, family-friendly AI creative studio for generating illustrated storybook
movies and pictures — built on top of ComfyUI with Wan 2.2, Flux schnell, Flux
Kontext, and a local Ollama LLM. No cloud, no API keys, no per-generation cost;
runs entirely on a home machine with dual NVIDIA GPUs.

The headline feature is **Storybook Movie Maker**: type a one-line idea, and the
studio plans a multi-scene picture book, illustrates each scene with consistent
characters and locked backgrounds, lets you approve the keyframes before any heavy
work, animates every scene with cinematic motion, and stitches it all into a single
silent MP4 you can watch on the couch.

> Built for my family. The UI is simple enough for kids; the pipeline underneath
> is tuned for quality and character coherency.

## Highlights

- **Storybook Movie Maker** — story → planned scenes (locations, characters, story
  beats) → illustrated keyframes → approval gate → animated clips with first/last
  frame guidance → ffmpeg-stitched into one cinematic MP4.
- **Picture maker** — Flux schnell text-to-image and image-to-image at up to
  1280×768.
- **Photo enhancer** — 4× upscale with 4x-UltraSharp.
- **Character consistency** — Flux Kontext locks the protagonist's appearance
  across all scenes; supporting characters get model-sheet refs; each clip starts
  from the *actual last rendered frame* of the previous clip so pose continuity is
  exact. CLIP-vision drift detection flags scenes where the character has drifted.
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
- **Per-scene regenerate** — re-roll a single keyframe at the approval gate, or
  re-animate a single Wan scene after the storybook is done, without redoing
  the rest.
- **Fast smoke-test mode** — a `smoke` flag runs a short, low-step, auto-approved
  3-page generation in a few minutes, so the coherency pipeline can be eyeballed
  before committing to a full-length, full-quality run.
- **Quality stack on by default** — Wan 2.2 14B MoE (two-expert chain) with Skip
  Layer Guidance, Enhance-A-Video, TeaCache, and SageAttention.
- **Dual-GPU aware** — block-swap to a second GPU keeps each 24 GB card from
  running out of VRAM. LLM unloads from VRAM before diffusion starts.
- **Local & private** — Ollama (`qwen3.6:latest` for everything LLM), no
  telemetry, no external calls.
- **Friendly UI** — React + Tailwind, mobile-friendly, kid-readable. Avatar
  picker for attribution. Live thumbnail strip during illustration and a full
  approval gate before animation kicks off.
- **Remote access** — mDNS hostname for the LAN; NordVPN Meshnet for "from
  anywhere" without exposing anything to the public internet.

## Architecture at a glance

```
                  ┌────────────────────────────────────────────────────┐
   browser ─────► │ React + Tailwind frontend (Vite, 3000)             │
                  │   • Storybook tab   • Picture tab                  │
                  └─────────────────────┬──────────────────────────────┘
                                        │ /api/*
                                        ▼
                  ┌────────────────────────────────────────────────────┐
                  │ FastAPI backend (8000) — main.py + modules         │
                  │   • workflows/  builds ComfyUI workflow JSON       │
                  │   • storybook   orchestrates the pipeline          │
                  │   • comfy/state submit + track the active gen      │
                  │   • llm/        Ollama planner & prompt-improve     │
                  └─────┬─────────────────┬──────────────────────────┬─┘
                        │                 │                          │
                        ▼                 ▼                          ▼
                ┌──────────────┐  ┌──────────────┐         ┌──────────────────┐
                │ ComfyUI 8188 │  │ Ollama 11434 │         │ CLIP-vit (CPU)   │
                │ Wan 2.2 14B  │  │ qwen3.6      │         │ drift scoring    │
                │ Flux schnell │  │              │         │                  │
                │ Flux Kontext │  │              │         │                  │
                │ Upscaler     │  │              │         │                  │
                └──────────────┘  └──────────────┘         └──────────────────┘
```

For the deeper walkthrough of the storybook pipeline, the prev-end-as-Kontext-ref
trick, the approval gate, and VRAM choreography, see
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
│   ├── workflows/       Workflow builders — wan.py (Wan I2V), flux.py (Flux/SDXL)
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
    │   │   ├── OutputPanel.tsx        result + thumbnail strip + approval gate UI
    │   │   └── ProgressDisplay.tsx    friendly progress messaging
    │   ├── lib/{api,store,presets,user,utils}.ts
    │   └── types.ts
    └── vite.config.ts              listens on 0.0.0.0:3000, proxies /api → 127.0.0.1:8000
```

## Prerequisites

1. **ComfyUI** installed at `/home/yunus/Documents/comfyui` with these custom nodes:
   - [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (Kijai)
   - [ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU)
   - SageAttention (compiled wheel or pip)
2. **Model weights** on a large drive (mine: `/media/yunus/More Data/comfyui-models/`),
   wired up via `extra_model_paths.yaml`:
   - Wan 2.2 I2V high + low noise FP8 (`wan2.2_i2v_*_14B_fp8_scaled.safetensors`)
   - Wan 2.1 VAE + UMT5 XXL text encoder + Wan CLIP-vision
   - Flux schnell + Flux Kontext + Flux T5 + Flux CLIP-L + Flux VAE
   - SDXL base + SDXL Lightning LoRA (optional, fallback image model)
   - 4x-UltraSharp upscaler
3. **Ollama** running on `127.0.0.1:11434` with:
   - `qwen3.6:latest` (~23 GB — used for prompt improvement *and* storybook planning)
4. **Python deps**: `fastapi`, `uvicorn`, `httpx`, `websockets`, `pydantic`,
   `pillow`, `transformers`, `torch`. (Living in `/home/yunus/Documents/comfyui/venv/`.)
5. **System**: `ffmpeg`, `ffprobe`, `avahi-daemon` (for `fatimahstudio.local` mDNS).

## Running locally

ComfyUI must already be up on port 8188 with the custom nodes loaded. Then:

```bash
# Backend
cd studio/backend
/home/yunus/Documents/comfyui/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000

# Frontend
cd studio/frontend
npm install
npm run dev -- --host 0.0.0.0
```

Open `http://fatimahstudio.local:3000/` on the LAN, or `http://localhost:3000/`.

(On my box, the backend is a `fatimah-backend.service` user unit that depends on
the `More Data` drive being mounted via fstab so the model weights are reachable.)

### Smoke test

With the backend, ComfyUI, and Ollama all up, a fast end-to-end check (3 short pages,
low steps, auto-approved — a few minutes instead of hours) validates the coherency
pipeline and drops seam frames for inspection:

```bash
cd studio/backend
/home/yunus/Documents/comfyui/venv/bin/python smoke_test.py
```

It posts a `smoke: true` storybook, watches progress, then writes the seam frames to
`output/smoke_seams_<id>/`. The backend log prints each page's keyframe route
(`kontext` / `kontext (location cut)` / `plain-flux`).

## Remote access (optional)

The frontend listens on `0.0.0.0:3000`; the backend, ComfyUI, and Ollama all stay
on `127.0.0.1`. For "from anywhere" access without exposing anything to the public
internet, the studio runs over **NordVPN Meshnet** (free, peer-to-peer). Enable
Meshnet on this machine and any phone/laptop you want to use, then visit
`http://<meshnet-ip>:3000/`.

## Status

Personal project, actively used by my family. No external contributions expected,
but the code is MIT-friendly and the architecture doc should make forks
straightforward.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and pipeline details.
