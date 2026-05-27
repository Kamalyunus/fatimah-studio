export interface UserProfile {
  name: string;
  emoji: string;
}

const KEY = "fatimah-studio-user";

export const AVATAR_EMOJIS = [
  "🦊", "🐯", "🐻", "🐼", "🦁", "🐸", "🐙", "🦄",
  "🌸", "🌻", "🌈", "🎨", "📸", "🎬", "🚀", "⭐",
];

export const DEFAULT_PROFILE: UserProfile = { name: "Guest", emoji: "👤" };

export function loadProfile(): UserProfile | null {
  try {
    const s = localStorage.getItem(KEY);
    if (!s) return null;
    const p = JSON.parse(s);
    if (p && typeof p.name === "string" && typeof p.emoji === "string") return p;
    return null;
  } catch {
    return null;
  }
}

export function saveProfile(profile: UserProfile) {
  try {
    localStorage.setItem(KEY, JSON.stringify(profile));
  } catch {}
}

export function clearProfile() {
  try {
    localStorage.removeItem(KEY);
  } catch {}
}
