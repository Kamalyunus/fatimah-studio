import { useEffect, useState } from "react";
import { BookOpen, Square as SquareIcon, Trash2 } from "lucide-react";
import { useStudio } from "../lib/store";
import { Section, Textarea } from "./ui";
import { api } from "../lib/api";
import { ASPECTS, type Aspect } from "../lib/presets";
import { cn } from "../lib/utils";
import type { SavedCharacter } from "../types";

type Style = "pixar" | "watercolor" | "anime" | "cartoon";

const STYLES: Array<{ key: Style; label: string; emoji: string; hint: string }> = [
  { key: "pixar",      label: "Pixar 3D",   emoji: "🎬", hint: "Soft, cinematic" },
  { key: "watercolor", label: "Watercolor", emoji: "🎨", hint: "Soft pastels" },
  { key: "anime",      label: "Anime",      emoji: "✨", hint: "Studio Ghibli vibe" },
  { key: "cartoon",    label: "Cartoon",    emoji: "🖍️", hint: "Bright, bold" },
];

// Each scene is a fixed 81-frame Wan clip at 16 fps ≈ 5 s. We let the user pick a
// target duration; n_pages is derived by dividing by 5.
const SECONDS_PER_SCENE = 5;
const DURATION_OPTIONS = [
  { seconds: 30, label: "Short",  hint: "~55 min to make" },
  { seconds: 45, label: "Medium", hint: "~80 min to make" },
  { seconds: 60, label: "Long",   hint: "~105 min to make" },
];

const STORY_SAMPLES = [
  { emoji: "🤖", label: "Robot baker", story: "A friendly little robot named Bolt who really wants to learn how to bake cookies." },
  { emoji: "🐱", label: "Cat astronaut", story: "A curious cat named Mochi who builds a tiny rocket and visits the moon." },
  { emoji: "🌳", label: "Tree friend", story: "A lonely tree in the forest who befriends a tiny bird and learns the meaning of friendship." },
  { emoji: "🐉", label: "Dragon chef", story: "A small green dragon who runs a noodle restaurant deep in the mountains." },
];

export function StorybookPanel() {
  const { status, busyByOther, generateStorybook, cancel } = useStudio();
  const [story, setStory] = useState("");
  const [durationSec, setDurationSec] = useState<number>(45);
  const [style, setStyle] = useState<Style>("pixar");
  const [aspect, setAspect] = useState<Aspect>("landscape");
  const [characters, setCharacters] = useState<SavedCharacter[]>([]);
  const [characterId, setCharacterId] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    api.listCharacters()
      .then((r) => { if (!cancelled) setCharacters(r.items); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [status.phase]);   // refresh after each generation so newly-saved characters appear

  const nPages = Math.max(1, Math.round(durationSec / SECONDS_PER_SCENE));
  const running = status.phase === "running" || status.phase === "queued";
  const canSubmit = !running && !!story.trim();

  const onGenerate = () => {
    generateStorybook({ story, n_pages: nPages, style, aspect, character_id: characterId });
  };

  const onDeleteCharacter = async (id: string) => {
    if (!confirm("Forget this character?")) return;
    await api.deleteCharacter(id);
    if (characterId === id) setCharacterId("");
    setCharacters((cs) => cs.filter((c) => c.id !== id));
  };

  // ~12s Flux image + ~8.5 min Wan per scene at 1024×576 with 81 frames
  // (sageattn, 20 steps, TeaCache 0.20). Plus ~1 min for outline + critique
  // planning passes and final stitching.
  const minutes = Math.round(nPages * 8.5 + 4);

  return (
    <div className="flex flex-col gap-6">
      <div className="bg-gradient-to-br from-brand/10 to-orange-300/10 border border-brand/20 rounded-2xl p-4">
        <div className="flex items-center gap-2 mb-1">
          <BookOpen className="h-5 w-5 text-brand" />
          <h2 className="font-bold text-fg-base">Storybook Movie Maker</h2>
        </div>
        <p className="text-xs text-fg-muted">
          Tell us a story. We'll illustrate {nPages} scenes and animate them into a ~{durationSec}-second video.
        </p>
      </div>

      <Section title="💡 Need an idea?" hint="Tap one to fill the story">
        <div className="flex flex-wrap gap-1.5">
          {STORY_SAMPLES.map((s) => (
            <button
              key={s.label}
              onClick={() => setStory(s.story)}
              className="text-xs px-2.5 py-1.5 rounded-lg border border-border bg-bg-subtle hover:bg-brand/5 hover:border-brand transition flex items-center gap-1"
              title={s.story}
            >
              <span>{s.emoji}</span>
              <span className="text-fg-base">{s.label}</span>
            </button>
          ))}
        </div>
      </Section>

      <Section title="Your story">
        <Textarea
          value={story}
          onChange={(e) => setStory(e.target.value)}
          placeholder="Once upon a time, there was a little robot who..."
          rows={6}
          className="text-base leading-relaxed min-h-[28vh] w-full"
        />
      </Section>

      {characters.length > 0 && (
        <Section title="Main character" hint="Reuse one you made before, or start fresh">
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setCharacterId("")}
              className={cn(
                "rounded-xl border p-2 transition-all text-left flex items-center gap-2",
                characterId === ""
                  ? "border-brand bg-brand/5 shadow-sm"
                  : "border-border bg-bg-subtle hover:border-border-strong",
              )}
            >
              <div className="h-12 w-12 rounded-md bg-bg-muted flex items-center justify-center text-xl">✨</div>
              <div className="pr-1">
                <div className="text-sm font-semibold">New character</div>
                <div className="text-[10px] text-fg-subtle">Generated from the story</div>
              </div>
            </button>
            {characters.map((c) => {
              const active = c.id === characterId;
              return (
                <div key={c.id} className="relative group">
                  <button
                    onClick={() => setCharacterId(c.id)}
                    className={cn(
                      "rounded-xl border p-2 transition-all text-left flex items-center gap-2",
                      active
                        ? "border-brand bg-brand/5 shadow-sm"
                        : "border-border bg-bg-subtle hover:border-border-strong",
                    )}
                    title={c.character}
                  >
                    <img src={api.characterImageUrl(c.id)} alt={c.name}
                         className="h-12 w-12 rounded-md object-cover bg-bg-muted" loading="lazy" />
                    <div className="pr-1 min-w-0">
                      <div className="text-sm font-semibold truncate max-w-[140px]">{c.name}</div>
                      <div className="text-[10px] text-fg-subtle truncate max-w-[140px]">
                        {(c.canon?.species as string) || ""}
                      </div>
                    </div>
                  </button>
                  <button
                    onClick={() => onDeleteCharacter(c.id)}
                    className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-bg-inset border border-border opacity-0 group-hover:opacity-100 transition flex items-center justify-center text-danger hover:bg-danger hover:text-white"
                    title="Forget this character"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              );
            })}
          </div>
        </Section>
      )}

      <Section title="How long?" hint="Total video length">
        <div className="grid grid-cols-3 gap-2">
          {DURATION_OPTIONS.map((o) => {
            const active = o.seconds === durationSec;
            return (
              <button
                key={o.seconds}
                onClick={() => setDurationSec(o.seconds)}
                className={cn(
                  "rounded-xl border p-3 text-center transition-all",
                  active
                    ? "border-brand bg-brand/5 shadow-sm"
                    : "border-border bg-bg-subtle hover:border-border-strong",
                )}
              >
                <div className={cn("text-2xl font-bold", active && "text-brand")}>
                  {o.seconds}s
                </div>
                <div className="text-xs text-fg-base font-semibold mt-0.5">
                  {o.label}
                </div>
                <div className="text-[10px] text-fg-subtle mt-0.5">
                  {o.hint}
                </div>
              </button>
            );
          })}
        </div>
      </Section>

      <Section title="Style">
        <div className="grid grid-cols-2 gap-2">
          {STYLES.map((s) => {
            const active = s.key === style;
            return (
              <button
                key={s.key}
                onClick={() => setStyle(s.key)}
                className={cn(
                  "rounded-xl border p-3 text-left transition-all",
                  active
                    ? "border-brand bg-brand/5 shadow-sm"
                    : "border-border bg-bg-subtle hover:border-border-strong",
                )}
              >
                <div className="text-2xl mb-1 leading-none">{s.emoji}</div>
                <div className={cn("text-sm font-semibold", active && "text-brand")}>
                  {s.label}
                </div>
                <div className="text-[10px] text-fg-subtle mt-0.5">{s.hint}</div>
              </button>
            );
          })}
        </div>
      </Section>

      <Section title="Shape">
        <div className="grid grid-cols-3 gap-2">
          {ASPECTS.map((a) => {
            const active = aspect === a.key;
            return (
              <button
                key={a.key}
                onClick={() => setAspect(a.key)}
                className={cn(
                  "rounded-xl border p-3 transition-all text-left",
                  active
                    ? "border-brand bg-brand/5 shadow-sm"
                    : "border-border bg-bg-subtle hover:border-border-strong",
                )}
              >
                <div className="text-2xl mb-1 leading-none">{a.icon}</div>
                <div className={cn("text-sm font-semibold", active && "text-brand")}>
                  {a.label}
                </div>
                <div className="text-[10px] text-fg-subtle mt-0.5">{a.hint}</div>
              </button>
            );
          })}
        </div>
      </Section>

      <div className="pt-2 sticky bottom-0 -mx-4 px-4 pb-4 pt-2 bg-gradient-to-t from-bg-base via-bg-base to-transparent">
        {running ? (
          <>
            {busyByOther && (
              <div className="text-center text-xs text-fg-muted pb-1">
                Someone is making something — stop it to take over.
              </div>
            )}
            <button
              onClick={cancel}
              className="w-full h-14 rounded-xl bg-danger text-white font-semibold text-base hover:opacity-90 transition flex items-center justify-center gap-2"
            >
              <SquareIcon className="h-4 w-4" fill="currentColor" /> Stop & free GPU
            </button>
          </>
        ) : (
          <button
            onClick={onGenerate}
            disabled={!canSubmit}
            className={cn(
              "w-full h-14 rounded-xl font-semibold text-base transition shadow-md flex items-center justify-center gap-2",
              "bg-gradient-to-r from-brand to-indigo-500 text-white hover:opacity-95 active:scale-[0.99]",
              "disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none",
            )}
          >
            <BookOpen className="h-5 w-5" />
            Make my storybook
          </button>
        )}
        <div className="text-center text-[10px] text-fg-subtle mt-2">
          Estimated time: ~{minutes} min
        </div>
      </div>
    </div>
  );
}
