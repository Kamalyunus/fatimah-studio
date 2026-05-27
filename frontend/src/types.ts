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

export interface UpscaleParams {
  image: string;
  factor: 2 | 4;
}

export interface StorybookParams {
  story: string;
  n_pages: number;
  style: "pixar" | "watercolor" | "anime" | "cartoon";
  aspect: "landscape" | "square" | "portrait";
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
    }
  | { phase: "done"; filename: string; durationS: number }
  | { phase: "error"; message: string }
  | { phase: "cancelled" };
