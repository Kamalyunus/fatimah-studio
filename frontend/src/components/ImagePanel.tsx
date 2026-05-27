import { useEffect, useRef, useState } from "react";
import { Sparkles, Wand2, X, Square as SquareIcon } from "lucide-react";
import { useStudio } from "../lib/store";
import { Section, Textarea } from "./ui";
import { ImageUpload } from "./ImageUpload";
import { api } from "../lib/api";
import {
  ASPECTS, IMAGE_DIMS, IMAGE_STYLE_CHIPS, IMAGE_SAMPLES,
  type Aspect,
} from "../lib/presets";

// Fixed modify strength — "moderate" tier. Used to be selectable; removed because
// users rarely changed it and the choice added noise. 0.6 = noticeable change, keeps
// the subject recognisable.
const MODIFY_STRENGTH = 0.6;
import { cn } from "../lib/utils";

// Sub-mode is implicit: image slot empty → create; image slot has a file → modify.
// initialMode is now informational only — kept for backward-compat with HomeScreen wiring.
type ImageMode = "create" | "modify";

export function ImagePanel({ initialMode = "create" }: { initialMode?: ImageMode } = {}) {
  void initialMode;
  const { status, result, busyByOther, generateImage, cancel, loadedEntry } = useStudio();
  const [aspect, setAspect] = useState<Aspect>("landscape");
  const [prompt, setPrompt] = useState("");
  const [image, setImage] = useState("");
  // Tracks which result filename we've already auto-loaded so we don't re-loop on every
  // status refresh after a generation finishes.
  const lastAutoLoadedRef = useRef<string>("");

  // Hydrate from a loaded history entry
  useEffect(() => {
    if (!loadedEntry) return;
    if (loadedEntry.kind !== "image") return;
    const p = loadedEntry.params || {};
    setPrompt(loadedEntry.prompt || "");
    if (loadedEntry.mode === "modify") {
      if (typeof p.image === "string") setImage(p.image);
    } else {
      // For Create entries, also offer the result as a starting point for iteration
      if (loadedEntry.filename) {
        api.useAsInput(loadedEntry.filename)
          .then((r) => { if (r.filename) setImage(r.filename); })
          .catch(() => {});
      }
    }
    const w = p.width ?? 1024, h = p.height ?? 1024;
    if (w > h * 1.2) setAspect("landscape");
    else if (h > w * 1.2) setAspect("portrait");
    else setAspect("square");
  }, [loadedEntry]);

  // After a Create finishes, auto-populate the image slot with the result so the next
  // Generate iterates on it. Only fires for image results; storybook/video results are
  // ignored. Skip if user already loaded a different image during the gen.
  useEffect(() => {
    if (status.phase !== "done") return;
    if (result?.kind !== "image") return;
    if (!result.filename) return;
    if (lastAutoLoadedRef.current === result.filename) return;
    lastAutoLoadedRef.current = result.filename;
    api.useAsInput(result.filename)
      .then((r) => { if (r.filename) setImage(r.filename); })
      .catch(() => {});
  }, [status.phase, result]);

  const isModify = !!image;
  const running = status.phase === "running" || status.phase === "queued";
  const canSubmit = !running && !!prompt.trim();

  const onGenerate = () => {
    const dims = IMAGE_DIMS[aspect];
    generateImage({
      image_mode: isModify ? "modify" : "create",
      prompt,
      negative: "",
      width: dims.width,
      height: dims.height,
      seed: 0,
      model: "flux",
      image,
      strength: isModify ? MODIFY_STRENGTH : 1.0,
    });
  };

  const [improving, setImproving] = useState(false);
  // null = nothing pending; otherwise the style key currently being applied.
  // We track which chip (or "improve") is in flight so the UI can show a per-button spinner.
  const [pendingStyle, setPendingStyle] = useState<string | null>(null);

  const improvePrompt = async () => {
    if (improving || pendingStyle || !prompt.trim()) return;
    setImproving(true);
    try {
      const r = await api.improvePrompt(prompt);
      if (r.prompt) setPrompt(r.prompt);
    } catch {/* silent */}
    finally { setImproving(false); }
  };

  const applyStyle = async (styleKey: string) => {
    if (improving || pendingStyle || !prompt.trim()) return;
    setPendingStyle(styleKey);
    try {
      const r = await api.improvePrompt(prompt, styleKey);
      if (r.prompt) setPrompt(r.prompt);
    } catch {/* silent */}
    finally { setPendingStyle(null); }
  };

  const clearImage = () => setImage("");

  return (
    <div className="flex flex-col gap-6">
      {/* Photo slot — present always; mode flips based on whether it has a file */}
      <Section
        title={isModify ? "Editing this picture" : "Start from a photo (optional)"}
        hint={isModify ? "Describe a change below, or remove to start fresh" : "Or skip and describe a new picture"}
      >
        <div className="relative">
          <ImageUpload filename={image} onChange={setImage} />
          {isModify && (
            <button
              onClick={clearImage}
              className="absolute top-2 right-2 h-8 w-8 rounded-full bg-black/60 hover:bg-danger text-white flex items-center justify-center transition shadow-md"
              title="Remove photo (back to Create mode)"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </Section>

      {/* Idea chips — only when starting fresh */}
      {!isModify && (
        <Section title="💡 Need an idea?" hint="Tap one to fill the prompt">
          <div className="flex flex-wrap gap-1.5">
            {IMAGE_SAMPLES.map((s) => (
              <button
                key={s.label}
                onClick={() => {
                  setPrompt(s.prompt);
                  if (s.aspect) setAspect(s.aspect);
                }}
                className="text-xs px-2.5 py-1.5 rounded-lg border border-border bg-bg-subtle hover:bg-brand/5 hover:border-brand transition flex items-center gap-1"
                title={s.prompt}
              >
                <span>{s.emoji}</span>
                <span className="text-fg-base">{s.label}</span>
              </button>
            ))}
          </div>
        </Section>
      )}

      {/* Prompt + Refine/style chips — same component in both modes */}
      <Section title={isModify ? "What should change?" : "Describe your picture"}>
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={
            isModify
              ? "Make it look like an oil painting / change the season to autumn / add a rainbow…"
              : "A cozy cabin on a snowy mountain at sunset, smoke rising from the chimney…"
          }
          rows={isModify ? 6 : 8}
          className={cn(
            "text-base leading-relaxed w-full",
            isModify ? "min-h-[28vh]" : "min-h-[35vh]"
          )}
        />
        <div className="flex flex-wrap gap-1.5 pt-1 items-center">
          <button
            onClick={improvePrompt}
            disabled={improving || pendingStyle !== null || !prompt.trim()}
            className="text-xs px-2.5 py-1 rounded-md border border-brand/40 bg-brand/5 text-brand hover:bg-brand/10 transition flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Add concrete visual detail to your prompt — model picks the style"
          >
            <Wand2 className={cn("h-3 w-3", improving && "animate-spin")} />
            {improving ? "Thinking…" : "Refine"}
          </button>
          {IMAGE_STYLE_CHIPS.map((c) => {
            const busy = pendingStyle === c.key;
            const disabled = !prompt.trim() || improving || (pendingStyle !== null && !busy);
            return (
              <button
                key={c.key}
                onClick={() => applyStyle(c.key)}
                disabled={disabled}
                className="text-xs px-2.5 py-1 rounded-md border border-border bg-bg-subtle text-fg-muted hover:text-brand hover:border-brand transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
                title={busy ? `Rewriting in ${c.label.toLowerCase()} style…` : `Rewrite prompt with ${c.label.toLowerCase()} style`}
              >
                {busy ? <Wand2 className="h-3 w-3 animate-spin" /> : <span>+</span>}
                {c.label}
              </button>
            );
          })}
        </div>
      </Section>

      {/* Shape selector — only when starting a fresh picture (Create mode) */}
      {!isModify && (
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
      )}

      {/* Big action button */}
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
            <Sparkles className="h-5 w-5" />
            {isModify ? "Apply the change" : "Make my picture"}
          </button>
        )}
      </div>
    </div>
  );
}
