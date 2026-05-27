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

export const IMAGE_STRENGTHS = [
  { key: "subtle",   label: "Subtle",   hint: "Small tweaks",      value: 0.35 },
  { key: "moderate", label: "Moderate", hint: "Noticeable change", value: 0.6 },
  { key: "bold",     label: "Bold",     hint: "Reimagine it",      value: 0.85 },
];

export const IMAGE_STYLE_CHIPS = [
  { label: "Cinematic",     snippet: "cinematic lighting, dramatic, film still" },
  { label: "Photorealistic", snippet: "photorealistic, 8k, sharp, professional photography" },
  { label: "Anime",         snippet: "anime style, vibrant colors, Studio Ghibli-inspired" },
  { label: "Painting",      snippet: "oil painting style, painterly brushstrokes" },
  { label: "Pencil sketch", snippet: "detailed pencil sketch, graphite shading" },
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
