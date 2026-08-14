"""Fatimah Studio backend — FastAPI relay over ComfyUI.

Thin HTTP/WebSocket layer: parses requests, builds ComfyUI workflows (see workflows/),
submits them (see comfy.py), and tracks the single in-flight gen (see state.py). Storybook
orchestration lives in storybook.py.

  POST /api/storybook          plan + illustrate + animate a storybook
  POST /api/image_generate     Flux/SDXL text-to-image (or image-to-image)
  POST /api/image_upscale      4x-UltraSharp upscale
  POST /api/llm/improve        rewrite a short prompt into a richer one
  POST /api/interrupt          cancel the currently running gen
  POST /api/upload             upload an image (returns {filename})
  GET  /api/state              current active gen + last error
  WS   /api/ws/{client_id}     relay of ComfyUI progress
  GET  /api/history            list past generations
  GET  /api/video|image|thumb/{filename}  stream artifacts
"""
from __future__ import annotations

import asyncio
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import websockets
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import llm
import state
import storybook
from config import *            # noqa: F401,F403  — COMFY_* paths, model names, dims, tunables
from models import *            # noqa: F401,F403  — request models
from workflows import *         # noqa: F401,F403  — workflow builders
from comfy import (
    _comfy_free,
    _generate_thumb,
    _submit_comfy_and_wait,
)
from state import GenState
from store import load_json, save_json

@asynccontextmanager
async def lifespan(app: FastAPI):
    THUMB_DIR.mkdir(exist_ok=True)
    state.state_lock = asyncio.Lock()
    await state._recover_active_gen()
    state.monitor_task = asyncio.create_task(state._monitor_loop())
    state.poll_task = asyncio.create_task(state._queue_poll_loop())
    yield
    for t in (state.monitor_task, state.poll_task):
        if t:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            r = await c.get(f"{COMFY_HTTP}/system_stats")
        return {"ok": True, "comfy": r.json().get("system", {}).get("comfyui_version")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    COMFY_INPUT.mkdir(exist_ok=True)
    ext = Path(file.filename or "").suffix or ".png"
    name = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    dest = COMFY_INPUT / name
    with dest.open("wb") as fp:
        shutil.copyfileobj(file.file, fp)
    return {"filename": name}



@app.post("/api/use_as_input")
async def use_as_input(p: UseAsInputParams):
    """Copy a previously generated output image into ComfyUI's input/ folder so it can
    be referenced by a follow-up modify/i2i workflow without a full upload round-trip."""
    src = COMFY_OUTPUT / p.filename
    if not src.exists() or not src.is_file():
        raise HTTPException(404, "source file not found in output")
    COMFY_INPUT.mkdir(exist_ok=True)
    ext = src.suffix or ".png"
    name = f"iterate_{uuid.uuid4().hex[:8]}{ext}"
    shutil.copyfile(str(src), str(COMFY_INPUT / name))
    return {"filename": name}


@app.post("/api/image_generate")
async def image_generate(params: ImageGenerateParams):
    """Text-to-image OR image-to-image (modify) using Flux schnell or SDXL Lightning."""
    if state.state_lock is None:
        raise HTTPException(503, "backend not ready")
    # Free the LLM from VRAM before queueing diffusion work
    await llm.unload()
    async with state.state_lock:
        if state.active_gen is not None:
            raise HTTPException(409, "Someone is already making something. Wait a moment.")
        if params.image_mode == "modify" and not params.image:
            raise HTTPException(400, "Modify mode needs an uploaded image")

        # User-facing image gen always auto-upscales 2x; storybook page gen doesn't (the
        # storybook orchestrator builds ImageGenerateParams directly without this flag).
        params.auto_upscale = True

        if params.model == "sdxl":
            workflow = build_sdxl_image_workflow(params)
        else:
            workflow = build_flux_image_workflow(params)

        gen_id = uuid.uuid4().hex[:8]
        client_id = MONITOR_CLIENT_ID
        async with httpx.AsyncClient(timeout=30) as c:
            try:
                r = await c.post(
                    f"{COMFY_HTTP}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                )
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise HTTPException(502, f"Engine rejected: {e.response.text[:300]}")
            data = r.json()

        state.active_gen = GenState(
            prompt_id=data["prompt_id"],
            gen_id=gen_id,
            params=params.model_dump(),
            started_at=time.time(),
            kind="image",
        )
        state.last_error = None

    return {
        "prompt_id": data["prompt_id"],
        "client_id": client_id,
        "gen_id": gen_id,
        "queue_number": data.get("number"),
    }


@app.post("/api/image_upscale")
async def image_upscale(params: UpscaleParams):
    """Upscale an image 2× or 4× using a Real-ESRGAN-style model."""
    if state.state_lock is None:
        raise HTTPException(503, "backend not ready")
    if params.factor not in (2, 4):
        raise HTTPException(400, "factor must be 2 or 4")
    await llm.unload()
    async with state.state_lock:
        if state.active_gen is not None:
            raise HTTPException(409, "Someone is already making something. Wait a moment.")

        workflow = build_upscale_workflow(params)
        gen_id = uuid.uuid4().hex[:8]
        client_id = MONITOR_CLIENT_ID
        async with httpx.AsyncClient(timeout=30) as c:
            try:
                r = await c.post(
                    f"{COMFY_HTTP}/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                )
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise HTTPException(502, f"Engine rejected: {e.response.text[:300]}")
            data = r.json()

        params_dict = params.model_dump()
        params_dict["image_mode"] = "upscale"
        state.active_gen = GenState(
            prompt_id=data["prompt_id"],
            gen_id=gen_id,
            params=params_dict,
            started_at=time.time(),
            kind="upscale",
        )
        state.last_error = None

    return {
        "prompt_id": data["prompt_id"],
        "client_id": client_id,
        "gen_id": gen_id,
        "queue_number": data.get("number"),
    }


@app.get("/api/image/{filename}")
async def get_image(filename: str):
    """Serve an image file from the output dir."""
    path = COMFY_OUTPUT / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "not found")
    ext = path.suffix.lower()
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=media)



@app.post("/api/storybook")
async def create_storybook(p: StorybookParams):
    if state.state_lock is None:
        raise HTTPException(503, "backend not ready")
    if not p.story.strip():
        raise HTTPException(400, "story is required")
    if p.n_pages < 2 or p.n_pages > 15:
        raise HTTPException(400, "n_pages must be 2..15")
    if not await llm.is_available():
        raise HTTPException(503, "Local LLM (Ollama llama3.2:3b) is not available. Start ollama and pull the model.")
    async with state.state_lock:
        if state.active_gen is not None:
            raise HTTPException(409, "Someone is already making something.")
        gen_id = uuid.uuid4().hex[:8]
        prompt_id = f"storybook-{gen_id}"
        state.active_gen = GenState(
            prompt_id=prompt_id,
            gen_id=gen_id,
            params=p.model_dump(),
            started_at=time.time(),
            kind="storybook",
        )
        state.last_error = None

    asyncio.create_task(storybook._run_storybook(p, prompt_id, gen_id))
    return {"prompt_id": prompt_id, "gen_id": gen_id, "kind": "storybook"}



@app.post("/api/storybook/approve")
async def storybook_approve():
    """Tell the orchestrator that the user is happy with the keyframes — proceed to Wan."""
    if state.active_gen is None or state.active_gen.kind != "storybook":
        raise HTTPException(409, "no storybook awaiting approval")
    if state.active_gen.node != "awaiting-approval":
        raise HTTPException(409, f"storybook is in '{state.active_gen.node}' state, not waiting for approval")
    state.active_gen.approval_cancelled = False
    state.active_gen.approval_event.set()
    return {"ok": True}


@app.post("/api/storybook/cancel_approval")
async def storybook_cancel_approval():
    """User rejected the keyframes; abort the storybook cleanly without running Wan."""
    if state.active_gen is None or state.active_gen.kind != "storybook":
        raise HTTPException(409, "no storybook awaiting approval")
    if state.active_gen.node != "awaiting-approval":
        raise HTTPException(409, f"storybook is in '{state.active_gen.node}' state, not waiting for approval")
    state.active_gen.approval_cancelled = True
    state.active_gen.approval_event.set()
    return {"ok": True}



@app.post("/api/storybook/regenerate_keyframe")
async def storybook_regenerate_keyframe(p: RegenerateKeyframeParams):
    """While the storybook is paused at the approval gate, re-run a single Flux frame
    (start or end of scene N) with a fresh seed. Updates the keyframe cache in place
    so the preview strip shows the new image. Wan has not started yet, so this is cheap."""
    if state.active_gen is None or state.active_gen.kind != "storybook":
        raise HTTPException(409, "no storybook awaiting approval")
    if state.active_gen.node != "awaiting-approval":
        raise HTTPException(409, "regen only allowed during keyframe approval")
    keyframes = state.active_gen.keyframes
    if not (0 <= p.scene_index < len(keyframes)):
        raise HTTPException(400, "scene_index out of range")
    if p.frame == "start" and p.scene_index > 0:
        raise HTTPException(400, "page 2+ start frames are byte-perfect copies of the previous end; regen that end frame instead")

    kf = keyframes[p.scene_index]
    params = state.active_gen.params or {}
    width = STORY_ASPECT_DIMS.get(params.get("aspect", "landscape"), STORY_ASPECT_DIMS["landscape"])[0]
    height = STORY_ASPECT_DIMS.get(params.get("aspect", "landscape"), STORY_ASPECT_DIMS["landscape"])[1]
    # Use the same composite reference this scene was originally generated with so the
    # cast stays visually consistent on regen.
    composite_ref = kf.get("composite_ref") or ""
    kontext_available = (
        composite_ref
        and (Path("/media/yunus/More Data/comfyui-models/diffusion_models") / FLUX_KONTEXT_MODEL).exists()
        and (COMFY_INPUT / composite_ref).exists()
    )

    new_seed = int(time.time() * 1000) % (2**31)
    await _comfy_free()
    if p.frame == "start":
        prompt_text = kf.get("start_prompt") or ""
        if not prompt_text:
            raise HTTPException(409, "this start frame is from a saved character and cannot be regenerated")
        wf = (
            build_flux_kontext_workflow(
                prompt=prompt_text, width=width, height=height,
                seed=new_seed, reference_image=composite_ref, steps=20,
            )
            if kontext_available
            else build_flux_image_workflow(ImageGenerateParams(
                image_mode="create", prompt=prompt_text,
                width=width, height=height, seed=new_seed, model="flux",
            ))
        )
        out = await _submit_comfy_and_wait(wf, timeout_s=300)
        shutil.copyfile(str(COMFY_OUTPUT / out), str(COMFY_INPUT / kf["start_input_name"]))
        kf["start_image"] = out
        if state.active_gen.preview_images:
            state.active_gen.preview_images[0] = out
        await storybook._rescore_drift_for_active()
        return {"ok": True, "filename": out}
    else:
        prompt_text = kf["end_prompt"]
        if kontext_available:
            wf = build_flux_kontext_workflow(
                prompt=prompt_text, width=width, height=height,
                seed=new_seed, reference_image=composite_ref, steps=20,
            )
        else:
            wf = build_flux_image_workflow(ImageGenerateParams(
                image_mode="create", prompt=prompt_text,
                width=width, height=height, seed=new_seed, model="flux",
            ))
        out = await _submit_comfy_and_wait(wf, timeout_s=300)
        shutil.copyfile(str(COMFY_OUTPUT / out), str(COMFY_INPUT / kf["end_input_name"]))
        kf["end_image"] = out
        # If a later scene took this scene's end as its start, propagate the change so
        # FLF2V chaining stays byte-perfect.
        if p.scene_index + 1 < len(keyframes):
            next_kf = keyframes[p.scene_index + 1]
            shutil.copyfile(str(COMFY_OUTPUT / out), str(COMFY_INPUT / next_kf["start_input_name"]))
            next_kf["start_image"] = out
            if state.active_gen.preview_images and (p.scene_index + 1) < len(state.active_gen.preview_images):
                state.active_gen.preview_images[p.scene_index + 1] = out
        await storybook._rescore_drift_for_active()
        return {"ok": True, "filename": out}



@app.post("/api/storybook/regenerate_scene")
async def storybook_regenerate_scene(p: RegenerateSceneParams):
    """Re-animate a single scene's Wan clip using the cached keyframes from the original
    run, then re-stitch the final video. The other scenes are untouched."""
    if state.state_lock is None:
        raise HTTPException(503, "backend not ready")
    items = load_json(HISTORY_FILE, [])
    entry = next((it for it in items if it.get("id") == p.gen_id and it.get("kind") == "storybook"), None)
    if not entry:
        raise HTTPException(404, "no storybook with that gen_id in history")
    params = entry.get("params") or {}
    scenes_meta = params.get("scenes_meta") or []
    target = next((s for s in scenes_meta if int(s.get("scene_index", -1)) == p.scene_index), None)
    if not target:
        raise HTTPException(404, "scene_index not found in this storybook's metadata")
    # Sanity: input frames must still be on disk. Prefer the chained start frame (the one
    # actually used in the original run) so a regen keeps the seam continuity; fall back to
    # the Kontext start keyframe if the chained frame was cleaned up.
    chain_start = target.get("chain_start_input_name") or target["start_input_name"]
    if not (COMFY_INPUT / chain_start).exists():
        chain_start = target["start_input_name"]
    target["_resolved_start_input"] = chain_start
    for fname in (chain_start, target["end_input_name"]):
        if not (COMFY_INPUT / fname).exists():
            raise HTTPException(409, f"input frame '{fname}' is no longer on disk — can't regen")

    async with state.state_lock:
        if state.active_gen is not None:
            raise HTTPException(409, "another generation is in progress")
        synthetic_prompt_id = f"regen-{p.gen_id}-{p.scene_index}-{int(time.time())}"
        state.active_gen = GenState(
            prompt_id=synthetic_prompt_id,
            gen_id=p.gen_id,
            params={**params, "regenerating_scene": p.scene_index},
            started_at=time.time(),
            kind="storybook",
        )
        state.active_gen.node = f"page-{p.scene_index+1}-animate"
        state.active_gen.step = 1
        state.active_gen.total_steps = 2
        state.last_error = None

    asyncio.create_task(storybook._do_regenerate_scene(p.gen_id, p.scene_index, target, params))
    return {"ok": True, "prompt_id": synthetic_prompt_id}



# ---------- Character library (re-use a protagonist across multiple stories) ----------

@app.get("/api/characters")
async def list_characters():
    return {"items": load_json(CHARACTER_LIBRARY_FILE, [])}


@app.post("/api/characters")
async def save_character(p: SaveCharacterParams):
    """Persist the character from a completed storybook gen into the library. Pulls the
    protagonist's canonical reference image and canon dict from the gen's history entry."""
    items = load_json(HISTORY_FILE, [])
    entry = next((it for it in items if it.get("id") == p.gen_id and it.get("kind") == "storybook"), None)
    if not entry:
        raise HTTPException(404, "no storybook with that gen_id in history")

    params = entry.get("params") or {}
    plan = params.get("plan") or {}
    chars = llm.coerce_characters(plan)
    proto = llm.protagonist_of(chars) or {}
    canon = {k: v for k, v in proto.items() if k != "role"}
    character_prose = plan.get("character") or ""

    proto_ref_filename = params.get("protagonist_ref_filename") or ""
    src_ref = COMFY_INPUT / proto_ref_filename if proto_ref_filename else None
    if not src_ref or not src_ref.exists():
        raise HTTPException(404, "character reference image no longer on disk for that gen")

    CHARACTER_LIBRARY_DIR.mkdir(exist_ok=True)
    char_id = uuid.uuid4().hex[:10]
    ref_filename = f"char_{char_id}.png"
    shutil.copyfile(str(src_ref), str(CHARACTER_LIBRARY_DIR / ref_filename))

    saved = {
        "id": char_id,
        "name": p.name.strip() or (canon.get("name") or "Character").strip(),
        "canon": canon,
        "character": character_prose,
        "ref_filename": ref_filename,
        "created_at": time.time(),
        "source_gen_id": p.gen_id,
    }
    library = load_json(CHARACTER_LIBRARY_FILE, [])
    library.insert(0, saved)
    save_json(CHARACTER_LIBRARY_FILE, library[:100])
    return saved


@app.delete("/api/characters/{char_id}")
async def delete_character(char_id: str):
    library = load_json(CHARACTER_LIBRARY_FILE, [])
    target = next((c for c in library if c.get("id") == char_id), None)
    library = [c for c in library if c.get("id") != char_id]
    save_json(CHARACTER_LIBRARY_FILE, library)
    if target:
        ref = CHARACTER_LIBRARY_DIR / (target.get("ref_filename") or "")
        if ref.exists():
            try: ref.unlink()
            except Exception: pass
    return {"ok": True}


@app.get("/api/characters/{char_id}/image")
async def get_character_image(char_id: str):
    library = load_json(CHARACTER_LIBRARY_FILE, [])
    target = next((c for c in library if c.get("id") == char_id), None)
    if not target:
        raise HTTPException(404, "character not found")
    ref_path = CHARACTER_LIBRARY_DIR / (target.get("ref_filename") or "")
    if not ref_path.exists():
        raise HTTPException(404, "reference image missing")
    return FileResponse(ref_path, media_type="image/png")


@app.post("/api/llm/improve")
async def llm_improve(p: ImprovePromptParams):
    if not await llm.is_available():
        raise HTTPException(503, "Local LLM not available")
    # Free ComfyUI's cached models so the (~23 GB) qwen3.6 can actually load on GPU.
    await _comfy_free()
    try:
        out = await llm.improve_prompt(p.prompt, style=p.style or None)
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {e}")
    return {"prompt": out}


@app.get("/api/state")
async def get_state():
    """Current backend state — what gen is active, last error, etc."""
    return {
        "active": state.active_gen.to_dict() if state.active_gen else None,
        "last_error": state.last_error,
        "monitor_client_id": MONITOR_CLIENT_ID,
    }


@app.post("/api/interrupt")
async def interrupt():
    """Stop the running gen AND free model VRAM so the next request can start clean."""
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            await c.post(f"{COMFY_HTTP}/interrupt")
        except Exception as e:
            print(f"[interrupt] failed to call /interrupt: {e}")
        try:
            await c.post(
                f"{COMFY_HTTP}/free",
                json={"unload_models": True, "free_memory": True},
            )
        except Exception as e:
            print(f"[interrupt] failed to call /free: {e}")
        # Also clear ComfyUI's queue so any pending jobs don't auto-start
        try:
            await c.post(f"{COMFY_HTTP}/queue", json={"clear": True})
        except Exception:
            pass
    state.active_gen = None
    return {"ok": True}


@app.websocket("/api/ws/{client_id}")
async def ws_relay(socket: WebSocket, client_id: str):
    """Forward ComfyUI WS events for one prompt_id back to the browser."""
    await socket.accept()
    upstream_url = f"{COMFY_WS}?clientId={client_id}"
    try:
        async with websockets.connect(upstream_url, max_size=2**24) as upstream:
            while True:
                try:
                    msg = await asyncio.wait_for(upstream.recv(), timeout=600)
                except asyncio.TimeoutError:
                    await socket.send_json({"type": "timeout"})
                    break
                if isinstance(msg, (bytes, bytearray)):
                    continue
                try:
                    await socket.send_text(msg)
                except WebSocketDisconnect:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await socket.send_json({"type": "ws_error", "error": str(e)})
        except Exception:
            pass


@app.get("/api/result/{prompt_id}")
async def result(prompt_id: str):
    """Once execution is finished, this returns the output filename + saves it to history."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{COMFY_HTTP}/history/{prompt_id}")
    h = r.json().get(prompt_id, {})
    if not h:
        raise HTTPException(404, "prompt not in history yet")

    outputs = h.get("outputs", {})
    filename: Optional[str] = None
    for _, out in outputs.items():
        for item in (out.get("videos") or out.get("gifs") or []):
            if item.get("filename"):
                filename = item["filename"]
                break
        if filename:
            break

    if not filename:
        raise HTTPException(404, "no video output yet")

    prompt_data = (h.get("prompt") or [None, None, {}])[2] or {}
    return {"filename": filename, "prompt_data": prompt_data}


@app.post("/api/history")
async def add_history(entry: HistoryEntry):
    items = load_json(HISTORY_FILE, [])
    items.insert(0, entry.model_dump())
    items = items[:200]
    save_json(HISTORY_FILE, items)
    return {"ok": True}


@app.get("/api/history")
async def list_history():
    items = load_json(HISTORY_FILE, [])
    return {"items": [it for it in items if (COMFY_OUTPUT / it["filename"]).exists()]}


@app.delete("/api/history/{entry_id}")
async def delete_history(entry_id: str, hard: bool = False):
    items = load_json(HISTORY_FILE, [])
    new_items = []
    deleted = None
    for it in items:
        if it.get("id") == entry_id:
            deleted = it
            continue
        new_items.append(it)
    save_json(HISTORY_FILE, new_items)
    if deleted and hard:
        p = COMFY_OUTPUT / deleted["filename"]
        if p.exists():
            try: p.unlink()
            except Exception: pass
        tp = THUMB_DIR / (deleted["filename"] + ".jpg")
        if tp.exists():
            try: tp.unlink()
            except Exception: pass
    return {"ok": True, "deleted": bool(deleted)}


@app.get("/api/video/{filename}")
async def get_video(filename: str):
    path = COMFY_OUTPUT / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/thumb/{filename}")
async def get_thumb(filename: str):
    video_path = COMFY_OUTPUT / filename
    if not video_path.exists():
        raise HTTPException(404, "video not found")
    thumb_path = THUMB_DIR / (filename + ".jpg")
    if not thumb_path.exists():
        if not _generate_thumb(video_path, thumb_path):
            raise HTTPException(500, "thumbnail extraction failed")
    return FileResponse(thumb_path, media_type="image/jpeg")


# Serve the built frontend (vite `dist/`) from this same process, so there is no
# separate node/vite dev server to run (or for external process managers to reap).
# Mounted LAST so every /api/* route above takes precedence; html=True serves
# index.html at "/". The app is single-page (view state in localStorage), so a
# plain static mount is sufficient — same-origin means the frontend's "/api" calls
# hit these routes directly with no proxy.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
else:
    print(f"[main] frontend build not found at {_FRONTEND_DIST} — run `npm run build` in studio/frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
