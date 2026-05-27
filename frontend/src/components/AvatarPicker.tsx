import { useState } from "react";
import { Sparkles } from "lucide-react";
import { AVATAR_EMOJIS, saveProfile, type UserProfile } from "../lib/user";
import { cn } from "../lib/utils";

interface Props {
  initial?: UserProfile | null;
  onDone: (profile: UserProfile) => void;
  onCancel?: () => void;
}

export function AvatarPicker({ initial, onDone, onCancel }: Props) {
  const [name, setName] = useState(initial?.name ?? "");
  const [emoji, setEmoji] = useState(initial?.emoji ?? "🌸");

  const save = () => {
    const trimmed = name.trim();
    if (!trimmed) return;
    const p = { name: trimmed, emoji };
    saveProfile(p);
    onDone(p);
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 backdrop-blur-sm p-4 animate-fade-in">
      <div className="panel p-8 max-w-md w-full animate-bounce-in">
        <div className="text-center mb-6">
          <div className="text-5xl mb-3">{emoji}</div>
          <h2 className="text-2xl font-bold text-fg-base mb-1 flex items-center justify-center gap-2">
            <Sparkles className="h-5 w-5 text-brand" />
            Welcome to Fatimah Studio
          </h2>
          <p className="text-sm text-fg-muted">
            {initial ? "Update your profile" : "Tell us your name so we know who made what"}
          </p>
        </div>

        <div className="space-y-5">
          <div>
            <label className="label block mb-2">Your name</label>
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && save()}
              placeholder="e.g. Fatimah"
              className="input text-base text-center"
            />
          </div>

          <div>
            <label className="label block mb-2">Pick an avatar</label>
            <div className="grid grid-cols-8 gap-2">
              {AVATAR_EMOJIS.map((e) => (
                <button
                  key={e}
                  onClick={() => setEmoji(e)}
                  className={cn(
                    "h-10 w-10 rounded-xl text-xl transition-all flex items-center justify-center",
                    emoji === e
                      ? "bg-brand/15 ring-2 ring-brand scale-110"
                      : "bg-bg-subtle hover:bg-bg-muted hover:scale-105",
                  )}
                >
                  {e}
                </button>
              ))}
            </div>
          </div>

          <div className="flex gap-2 pt-2">
            <button
              onClick={save}
              disabled={!name.trim()}
              className={cn(
                "flex-1 h-12 rounded-xl font-semibold text-base transition shadow-md flex items-center justify-center gap-2",
                "bg-gradient-to-r from-brand to-orange-400 text-white hover:opacity-95 active:scale-[0.99]",
                "disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none",
              )}
            >
              Let's go!
            </button>
            {onCancel && (
              <button
                onClick={onCancel}
                className="px-4 h-12 rounded-xl text-fg-muted hover:bg-bg-muted"
              >
                Cancel
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
