"""Shared application state: the single in-flight generation plus the ComfyUI monitor.

The mutable singletons (active_gen, last_error, state_lock, monitor_task, poll_task) live
here and are accessed *qualified* from other modules (e.g. `state.active_gen`) so that
reassignment is visible everywhere. The monitor/poll loops keep `active_gen` in sync with
ComfyUI and persist finished gens to history."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Optional

import httpx
import websockets

from config import COMFY_HTTP, COMFY_WS, HISTORY_FILE, MONITOR_CLIENT_ID
from store import load_json, save_json


# ---------- Mutable singletons (access qualified: state.active_gen) ----------
state_lock: Optional[asyncio.Lock] = None
active_gen: "Optional[GenState]" = None
last_error: Optional[str] = None
monitor_task: Optional[asyncio.Task] = None
poll_task: Optional[asyncio.Task] = None


class GenState:
    def __init__(self, prompt_id: str, gen_id: str, params: dict, started_at: float, kind: str = "video"):
        self.prompt_id = prompt_id
        self.gen_id = gen_id
        self.params = params
        self.started_at = started_at
        self.kind = kind  # "video" | "image" | "upscale" | "storybook"
        self.node: Optional[str] = None
        self.step: int = 0
        self.total_steps: int = 0
        self.preview_images: list[str] = []  # storybook: per-page Flux outputs
        self.scene_descriptions: list[str] = []  # storybook: per-page LLM scene descriptions
        self.character: str = ""  # storybook: LLM-generated character description (protagonist, prose)
        # Storybook cast for the UI — list of {name, role, species, ref_filename}.
        # The orchestrator populates this after the casting phase finishes.
        self.cast: list[dict] = []

        # ----- Storybook keyframe preview gate (#2) -----
        # When the orchestrator finishes generating all Flux start+end pairs, it sets
        # node="awaiting-approval" and waits on `approval_event`. The frontend reads
        # `keyframes` to show the strip, then POSTs /api/storybook/approve or /cancel.
        self.approval_event: asyncio.Event = asyncio.Event()
        self.approval_cancelled: bool = False
        # Per-scene context, populated during the Flux phase. Each entry:
        # {scene_index, start_image, end_image, description, motion_intensity, seed,
        #  start_prompt, end_prompt}
        # Mutable so the regenerate endpoint can update individual scenes in place.
        self.keyframes: list[dict] = []

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "gen_id": self.gen_id,
            "params": self.params,
            "started_at": self.started_at,
            "kind": self.kind,
            "node": self.node,
            "step": self.step,
            "total_steps": self.total_steps,
            "preview_images": list(self.preview_images),
            "scene_descriptions": list(self.scene_descriptions),
            "character": self.character,
            "cast": list(self.cast),
            # Lightweight view of the keyframe context for the frontend approval UI —
            # only the filenames + per-scene metadata it actually needs to render.
            "keyframes": [
                {
                    "scene_index": k.get("scene_index"),
                    "start_image": k.get("start_image"),
                    "end_image": k.get("end_image"),
                    "description": k.get("description"),
                    "motion_intensity": k.get("motion_intensity"),
                    "drift": k.get("drift"),
                    "drift_flagged": k.get("drift_flagged"),
                }
                for k in self.keyframes
            ],
            "elapsed_s": time.time() - self.started_at,
        }


async def _save_completion_to_history(gen: GenState):
    """Fetch result from ComfyUI history and append to our history.json."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{COMFY_HTTP}/history/{gen.prompt_id}")
        h = r.json().get(gen.prompt_id, {})
        outputs = h.get("outputs", {})
        filename = None
        for _, out in outputs.items():
            for key in ("videos", "gifs", "images"):
                for item in (out.get(key) or []):
                    if item.get("filename"):
                        filename = item["filename"]
                        break
                if filename:
                    break
            if filename:
                break
        if not filename:
            return
        items = load_json(HISTORY_FILE, [])
        items = [it for it in items if it.get("prompt_id") != gen.prompt_id]
        # Mode label for display
        mode = (
            gen.params.get("image_mode")  # create / modify / upscale
            or gen.params.get("mode")     # "storybook"
            or gen.kind
        )
        items.insert(0, {
            "id": gen.gen_id,
            "prompt_id": gen.prompt_id,
            "filename": filename,
            "kind": gen.kind,
            "mode": mode,
            "prompt": gen.params.get("prompt", ""),
            "params": gen.params,
            "created_by_name": gen.params.get("user_name", ""),
            "created_by_emoji": gen.params.get("user_emoji", ""),
            "created_at": gen.started_at,
            "duration_s": time.time() - gen.started_at,
        })
        items = items[:200]
        save_json(HISTORY_FILE, items)
    except Exception as e:
        print(f"[monitor] history save failed: {e}")


async def _handle_monitor_event(evt: dict):
    global active_gen, last_error
    if active_gen is None:
        return
    ty = evt.get("type")
    data = evt.get("data") or {}
    pid = data.get("prompt_id")
    if pid and pid != active_gen.prompt_id:
        return

    if ty == "executing":
        node = data.get("node")
        if node is None:
            # pipeline completed for this prompt
            done = active_gen
            active_gen = None
            await _save_completion_to_history(done)
        else:
            active_gen.node = node
    elif ty == "progress":
        v = int(data.get("value", 0) or 0)
        m = int(data.get("max", 1) or 1)
        node = data.get("node") or active_gen.node
        active_gen.node = node
        if node and node.lower() == "sampler" and m <= 100:
            active_gen.step = v
            active_gen.total_steps = m
    elif ty == "execution_error":
        last_error = f"{data.get('node_type','?')}: {data.get('exception_message','?')}"
        active_gen = None


async def _monitor_loop():
    """Persistent WebSocket to ComfyUI. All gens submit with MONITOR_CLIENT_ID,
    so this single socket sees every event for those gens."""
    backoff = 1.0
    while True:
        try:
            url = f"{COMFY_WS}?clientId={MONITOR_CLIENT_ID}"
            async with websockets.connect(url, max_size=2**24, ping_interval=20) as ws:
                backoff = 1.0
                async for msg in ws:
                    if isinstance(msg, (bytes, bytearray)):
                        continue
                    try:
                        evt = json.loads(msg)
                    except Exception:
                        continue
                    await _handle_monitor_event(evt)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[monitor] WS dropped: {e} — reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


async def _queue_poll_loop():
    """Fallback to the WS monitor: every 5s check ComfyUI's queue + history.
    Catches gens submitted before backend start, gens whose WS events we missed,
    and confirms completion even if the WS dropped."""
    global active_gen, last_error
    while True:
        try:
            await asyncio.sleep(5)
            if active_gen is None:
                continue
            # Storybook is orchestrated entirely by _run_storybook; its prompt_id is a
            # synthetic "storybook-..." string and isn't tracked by ComfyUI's queue.
            # The orchestrator clears active_gen itself when done.
            if active_gen.kind == "storybook":
                continue
            pid = active_gen.prompt_id
            async with httpx.AsyncClient(timeout=5) as c:
                qr = await c.get(f"{COMFY_HTTP}/queue")
            q = qr.json()
            running_pids = {item[1] for item in (q.get("queue_running") or [])}
            pending_pids = {item[1] for item in (q.get("queue_pending") or [])}
            if pid in running_pids or pid in pending_pids:
                continue  # still in flight, nothing to do
            # No longer in queue → check history
            async with httpx.AsyncClient(timeout=5) as c:
                hr = await c.get(f"{COMFY_HTTP}/history/{pid}")
            h = hr.json().get(pid)
            if not h:
                # Vanished without history entry — treat as cancelled
                active_gen = None
                continue
            status = (h.get("status") or {}).get("status_str", "")
            if status == "error":
                msgs = (h.get("status") or {}).get("messages") or []
                err_msg = "Generation failed"
                for m in msgs:
                    if isinstance(m, list) and m and m[0] == "execution_error":
                        err_msg = (m[1] or {}).get("exception_message") or err_msg
                        break
                last_error = err_msg
                active_gen = None
            else:
                # success — save to history, clear active
                done = active_gen
                active_gen = None
                await _save_completion_to_history(done)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[poll] error: {e}")


async def _recover_active_gen():
    """At startup, see if ComfyUI is already running a job we should track."""
    global active_gen
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{COMFY_HTTP}/queue")
        running = r.json().get("queue_running") or []
        if not running:
            return
        # Each item: [number, prompt_id, workflow_dict, extra_data?, ...]
        first = running[0]
        prompt_id = first[1]
        workflow = first[2] if len(first) > 2 else {}
        # Heuristically infer mode / prompt from the workflow
        mode = "i2v" if "i2v_encode" in workflow else "t2v"
        prompt_text = (
            workflow.get("text_encode", {}).get("inputs", {}).get("positive_prompt", "")
        )
        active_gen = GenState(
            prompt_id=prompt_id,
            gen_id=f"recovered-{uuid.uuid4().hex[:6]}",
            params={"mode": mode, "prompt": prompt_text, "_recovered": True},
            started_at=time.time(),  # we lost the real start; close enough
        )
        print(f"[monitor] recovered active gen prompt_id={prompt_id}")
    except Exception as e:
        print(f"[monitor] state recovery failed: {e}")
