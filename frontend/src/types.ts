export interface HistoryEntry {
  id: string;
  prompt_id: string;
  filename: string;
  kind?: "video" | "image" | "upscale" | "storybook";
  mode: string; // create/modify/upscale for image; "storybook" for storybook
  prompt: string;
  params: Record<string, any>;
  created_by_name?: string;
  created_by_emoji?: string;
  created_at: number;
  duration_s?: number;
}

export interface ImageGenParams {
  image_mode: "create" | "modify";
  prompt: string;
  negative?: string;
  width: number;
  height: number;
  seed: number;
  model: "flux" | "sdxl";
  image: string;     // for modify
  strength: number;  // for modify: 0.3 / 0.6 / 0.85
}

export interface StorybookParams {
  story: string;
  n_pages: number;
  style: "pixar" | "watercolor" | "anime" | "cartoon";
  aspect: "landscape" | "square" | "portrait";
  character_id?: string;   // id of a saved character to re-use; empty = generate fresh
}

export interface SavedCharacter {
  id: string;
  name: string;
  canon: Record<string, string>;
  character: string;
  ref_filename: string;
  created_at: number;
  source_gen_id?: string;
}

export interface KeyframePreview {
  scene_index: number;
  start_image: string;
  end_image: string;
  description?: string;
  motion_intensity?: string;
  drift?: number | null;            // cosine similarity to canonical char ref, 0..1
  drift_flagged?: boolean | null;   // true → character may have drifted from ref
}

export interface CastMember {
  name: string;
  role: string;          // 'protagonist' or 'supporting'
  species: string;
  ref_filename: string;  // PNG in ComfyUI's input/ dir
}

export type GenStatus =
  | { phase: "idle" }
  | { phase: "queued" }
  | {
      phase: "running";
      node?: string;
      step?: number;
      totalSteps?: number;
      startedAt: number;
      kind?: "video" | "image" | "upscale" | "storybook";
      previewImages?: string[];
      sceneDescriptions?: string[];
      character?: string;
      cast?: CastMember[];
      keyframes?: KeyframePreview[];
    }
  | { phase: "done"; filename: string; durationS: number }
  | { phase: "error"; message: string }
  | { phase: "cancelled" };
