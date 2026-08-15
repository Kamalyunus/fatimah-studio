import type {
  HistoryEntry, SavedCharacter,
  ImageGenParams, StorybookParams,
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
        cast?: Array<{ name: string; role: string; species: string; ref_filename: string }>;
        keyframes?: Array<{
          scene_index: number;
          start_image: string;
          end_image: string;
          description?: string;
          motion_intensity?: string;
          drift?: number | null;
          drift_flagged?: boolean | null;
        }>;
        animatic?: string;
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

  async storybook(params: StorybookParams) {
    const r = await fetch(`${BASE}/storybook`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    return jsonOrThrow<{ prompt_id: string; gen_id: string; kind: string }>(r);
  },

  async storybookRegenScene(gen_id: string, scene_index: number) {
    const r = await fetch(`${BASE}/storybook/regenerate_scene`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gen_id, scene_index }),
    });
    return jsonOrThrow<{ ok: boolean; prompt_id: string }>(r);
  },

  // Character library (re-use protagonist across stories)
  async listCharacters() {
    const r = await fetch(`${BASE}/characters`);
    return jsonOrThrow<{ items: SavedCharacter[] }>(r);
  },
  async saveCharacter(name: string, gen_id: string) {
    const r = await fetch(`${BASE}/characters`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, gen_id }),
    });
    return jsonOrThrow<SavedCharacter>(r);
  },
  async deleteCharacter(char_id: string) {
    return fetch(`${BASE}/characters/${char_id}`, { method: "DELETE" });
  },
  characterImageUrl(char_id: string) {
    return `${BASE}/characters/${encodeURIComponent(char_id)}/image`;
  },

  async improvePrompt(prompt: string, style?: string) {
    const r = await fetch(`${BASE}/llm/improve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, style: style ?? "" }),
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

  async useAsInput(filename: string) {
    const r = await fetch(`${BASE}/use_as_input`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename }),
    });
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
