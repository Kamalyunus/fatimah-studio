# Fatimah Studio

A local, family-friendly AI creative studio for generating illustrated storybook movies,
images, and photo enhancements — built on top of ComfyUI with Wan 2.2, Flux, Kokoro TTS,
and a local Ollama LLM. No cloud, no API keys, no per-generation cost; runs entirely on a
home machine with dual NVIDIA GPUs.

The headline feature is **Storybook Movie Maker**: type a one-line idea, and the studio
plans a multi-page picture book, illustrates each page with consistent characters,
animates every page with cinematic motion, narrates it with a warm storyteller voice,
and stitches it all into a single MP4 you can watch on the couch.

> Built for my family. The UI is simple enough for kids; the pipeline underneath is tuned
> for quality and character coherency.

## Highlights

- **Storybook Movie Maker** — story → planned scenes → illustrated pages → animated clips
  with first/last-frame guidance → narrated with Kokoro TTS → ffmpeg-stitched into one
  cinematic MP4.
- **Picture maker** — Flux schnell text-to-image and image-to-image at up to 1280×768.
- **Photo enhancer** — 4× upscale with 4x-UltraSharp.
- **Character consistency** — Flux Kontext locks the protagonist's appearance across all
  pages; byte-perfect end-frame → next-page start-frame chaining keeps pose continuity.
- **Quality stack on by default** — Wan 2.2 14B MoE (two-expert chain) with Skip Layer
  Guidance, Enhance-A-Video, and TeaCache for sharper motion and ~1.5–2× speedup.
- **Dual-GPU aware** — block-swap to a second GPU keeps each 24 GB card from running out
  of VRAM at high resolution. LLM unloads from VRAM before diffusion starts.
- **Local & private** — Ollama (`qwen3.6:latest` for planning, `qwen3:8b` for "Improve
  prompt"), Kokoro TTS on CPU, no telemetry, no external calls.
- **Friendly UI** — React + Tailwind, mobile-friendly, kid-readable. Avatar picker for
  attribution. Live thumbnail strip while the storybook is being illustrated.
- **Remote access** — mDNS hostname for the LAN; NordVPN Meshnet for "from anywhere"
  without exposing anything to the public internet.

## Architecture at a glance

```
                  ┌────────────────────────────────────────────────────┐
   browser ─────► │ React + Tailwind frontend (Vite, 3000)             │
                  │   • Storybook tab   • Picture tab                  │
                  └─────────────────────┬──────────────────────────────┘
                                        │ /api/*
                                        ▼
                  ┌────────────────────────────────────────────────────┐
                  │ FastAPI backend (8000) — main.py                   │
                  │   • Builds ComfyUI workflow JSON                   │
                  │   • Orchestrates storybook pipeline                │
                  │   • Calls Ollama, Kokoro, ffmpeg                   │
                  └─────┬─────────────────┬────────────────┬───────────┘
                        │                 │                │
                        ▼                 ▼                ▼
                ┌──────────────┐  ┌──────────────┐ ┌──────────────┐
                │ ComfyUI 8188 │  │ Ollama 11434 │ │ Kokoro (CPU) │
                │ Wan 2.2 14B  │  │ qwen3.6      │ │ bf_emma TTS  │
                │ Flux/Kontext │  │ qwen3:8b     │ │              │
                │ Upscaler/etc │  │              │ │              │
                └──────────────┘  └──────────────┘ └──────────────┘
```

For a deeper walkthrough of the storybook pipeline, character-consistency strategy, and
VRAM choreography, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Hardware

Currently tuned for and tested on:

- 2× NVIDIA RTX 3090 (48 GB total VRAM)
- ~100 GB free disk for model weights (Wan 2.2 + Flux + supporting encoders)
- Linux (Ubuntu / Pop!_OS), Python 3.10+, Node 20+

It will work on a single 24 GB card with quality reduced (lower resolution, smaller block
swap count). It will not work comfortably below 16 GB.

## Repo layout

```
studio/
├── backend/
│   ├── main.py        FastAPI app, ComfyUI workflow builders, storybook orchestrator
│   ├── llm.py         Ollama wrapper (prompt-improve + storybook planner)
│   ├── tts.py         Kokoro TTS wrapper (CPU-only, runs in a thread executor)
│   ├── history.json   Local history of generations (auto-created)
│   └── thumbs/        JPG thumbnails for the history drawer (auto-created)
└── frontend/
    ├── src/
    │   ├── App.tsx               Tabbed shell (Storybook + Picture)
    │   ├── components/
    │   │   ├── StorybookPanel.tsx   story input → submit
    │   │   ├── ImagePanel.tsx       create / modify / enhance modes
    │   │   ├── HomeScreen.tsx       landing tiles
    │   │   ├── OutputPanel.tsx      result preview + live thumbnail strip
    │   │   └── ProgressDisplay.tsx  friendly progress messaging
    │   ├── lib/{api,store,presets,user,utils}.ts
    │   └── types.ts
    └── vite.config.ts            listens on 0.0.0.0:3000, proxies /api → 127.0.0.1:8000
```

## Prerequisites

1. **ComfyUI** installed at `/home/yunus/Documents/comfyui` with the following custom nodes:
   - [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (Kijai)
   - [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) (for RIFE)
   - [ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU)
2. **Model weights** on a large drive (mine: `/media/yunus/More Data/comfyui-models/`),
   wired up via `extra_model_paths.yaml`:
   - Wan 2.2 I2V high + low noise FP8 (`wan2.2_i2v_*_14B_fp8_scaled.safetensors`)
   - Wan 2.1 VAE + UMT5 XXL text encoder + Wan CLIP-vision
   - Flux schnell + Flux Kontext + Flux T5 + Flux CLIP-L + Flux VAE
   - SDXL base + SDXL Lightning LoRA (optional, fallback image model)
   - 4x-UltraSharp upscaler
3. **Ollama** running on `127.0.0.1:11434` with:
   - `qwen3.6:latest` (~23 GB — storybook planning)
   - `qwen3:8b` (~5 GB — prompt improvement)
4. **Python deps**: `fastapi`, `uvicorn`, `httpx`, `websockets`, `pydantic`, `kokoro`,
   `soundfile`, `numpy`. (Living in `/home/yunus/Documents/comfyui/venv/`.)
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

## Remote access (optional)

The frontend listens on `0.0.0.0:3000`; the backend, ComfyUI, and Ollama all stay on
`127.0.0.1`. For "from anywhere" access without exposing anything to the public internet,
the studio runs over **NordVPN Meshnet** (free, peer-to-peer). Enable Meshnet on this
machine and any phone/laptop you want to use, then visit `http://<meshnet-ip>:3000/`.

## Status

Personal project, actively used by my family. No external contributions expected, but the
code is MIT-friendly and the architecture doc should make forks straightforward.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and pipeline details.
