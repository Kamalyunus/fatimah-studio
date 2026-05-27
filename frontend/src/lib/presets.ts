// ---------- Aspect ratios ----------
export type Aspect = "landscape" | "square" | "portrait";

export const ASPECTS: Array<{
  key: Aspect;
  label: string;
  hint: string;
  icon: string; // emoji
}> = [
  { key: "landscape", label: "Wide",   hint: "16:9 · YouTube / TV",      icon: "📺" },
  { key: "square",    label: "Square", hint: "1:1 · Instagram",          icon: "⬜" },
  { key: "portrait",  label: "Tall",   hint: "9:16 · TikTok / Reels",    icon: "📱" },
];

// ---------- Image presets ----------
export const IMAGE_DIMS: Record<Aspect, { width: number; height: number }> = {
  landscape: { width: 1280, height: 768 },
  square:    { width: 1024, height: 1024 },
  portrait:  { width: 768,  height: 1280 },
};

// Style chips. The `key` is sent to the backend LLM; the label is what the user sees.
export const IMAGE_STYLE_CHIPS = [
  { key: "cinematic",     label: "Cinematic" },
  { key: "photorealistic", label: "Photorealistic" },
  { key: "anime",         label: "Anime" },
  { key: "painting",      label: "Painting" },
  { key: "pencil sketch", label: "Pencil sketch" },
];

// ---------- Sample prompts ----------
export interface SamplePrompt {
  emoji: string;
  label: string;
  prompt: string;
  aspect?: Aspect;
}

export const IMAGE_SAMPLES: SamplePrompt[] = [
  { emoji: "🌸", label: "Cherry blossoms", prompt: "A breathtaking field of cherry blossoms in full bloom at dawn, soft morning mist, pink petals floating in the air, photorealistic", aspect: "landscape" },
  { emoji: "🍕", label: "Perfect pizza", prompt: "A perfect Margherita pizza fresh from a wood-fired oven, on a rustic wooden board, top-down view, melted mozzarella, basil leaves", aspect: "square" },
  { emoji: "🤖", label: "Painter robot", prompt: "An adorable round robot with a paintbrush, smiling, standing in front of a colorful canvas, soft studio lighting, friendly atmosphere", aspect: "portrait" },
  { emoji: "🏔️", label: "Mountain peaks", prompt: "Snow-capped mountain peaks at golden hour, dramatic clouds, alpenglow on the snow, photorealistic landscape photography", aspect: "landscape" },
  { emoji: "🦊", label: "Forest fox", prompt: "A red fox sitting alert in an autumn forest, sunbeams through the trees, soft bokeh background, wildlife photography", aspect: "square" },
  { emoji: "🍃", label: "Dewy leaf", prompt: "Macro photograph of a single green leaf covered in dewdrops, early morning, soft natural light, extreme detail", aspect: "square" },
  { emoji: "🌃", label: "City night", prompt: "A rain-soaked Tokyo street at night, neon signs reflecting in puddles, cinematic, moody atmosphere, cyberpunk vibes", aspect: "portrait" },
  { emoji: "🐢", label: "Sea turtle", prompt: "A baby sea turtle on a beach, golden sunset light, gently making its way toward the ocean, heartwarming, photorealistic", aspect: "landscape" },
];
