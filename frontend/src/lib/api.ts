import type {
  HistoryEntry,
  ImageGenParams, UpscaleParams, StorybookParams,
} from "../types";

const BASE = "/api";

async function jsonOrThrow<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  async health() {
    const r = await fetch(`${BASE}/health`);
    return jsonOrThrow<{ ok: boolean; comfy?: string; error?: string }>(r);
  },

  async state() {
    const r = await fetch(`${BASE}/state`);
    return jsonOrThrow<{
      active: null | {
        prompt_id: string;
        gen_id: string;
        params: Record<string, any>;
        started_at: number;
        kind?: "video" | "image" | "upscale" | "storybook";
        node: string | null;
        step: number;
        total_steps: number;
        preview_images?: string[];
        scene_descriptions?: string[];
        character?: string;
        elapsed_s: number;
      };
      last_error: string | null;
      monitor_client_id: string;
    }>(r);
  },

  async generateImage(params: ImageGenParams) {
    const r = await fetch(`${BASE}/image_generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    return jsonOrThrow<{
      prompt_id: string;
      client_id: string;
      gen_id: string;
    }>(r);
  },

  async upscale(params: UpscaleParams) {
    const r = await fetch(`${BASE}/image_upscale`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    return jsonOrThrow<{
      prompt_id: string;
      client_id: string;
      gen_id: string;
    }>(r);
  },

  async storybook(params: StorybookParams) {
    const r = await fetch(`${BASE}/storybook`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    return jsonOrThrow<{ prompt_id: string; gen_id: string; kind: string }>(r);
  },

  async improvePrompt(prompt: string) {
    const r = await fetch(`${BASE}/llm/improve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    return jsonOrThrow<{ prompt: string }>(r);
  },

  async interrupt() {
    return fetch(`${BASE}/interrupt`, { method: "POST" });
  },

  async upload(file: File) {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
    return jsonOrThrow<{ filename: string }>(r);
  },

  async result(prompt_id: string) {
    const r = await fetch(`${BASE}/result/${prompt_id}`);
    return jsonOrThrow<{ filename: string }>(r);
  },

  async history() {
    const r = await fetch(`${BASE}/history`);
    return jsonOrThrow<{ items: HistoryEntry[] }>(r);
  },
  async addHistory(entry: HistoryEntry) {
    return fetch(`${BASE}/history`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
    });
  },
  async deleteHistory(id: string, hard = false) {
    return fetch(`${BASE}/history/${id}?hard=${hard}`, { method: "DELETE" });
  },

  videoUrl(filename: string) {
    return `${BASE}/video/${encodeURIComponent(filename)}`;
  },
  imageUrl(filename: string) {
    return `${BASE}/image/${encodeURIComponent(filename)}`;
  },
  fileUrl(filename: string) {
    return /\.(mp4|webm|gif)$/i.test(filename)
      ? this.videoUrl(filename)
      : this.imageUrl(filename);
  },
  thumbUrl(filename: string) {
    return `${BASE}/thumb/${encodeURIComponent(filename)}`;
  },

  wsUrl(client_id: string) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${location.host}${BASE}/ws/${client_id}`;
  },
};
