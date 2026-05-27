import { useEffect, useState } from "react";
import { Sparkles, Wand2, Maximize, Square as SquareIcon } from "lucide-react";
import { useStudio } from "../lib/store";
import { Section, Textarea } from "./ui";
import { ImageUpload } from "./ImageUpload";
import { api } from "../lib/api";
import {
  ASPECTS, IMAGE_DIMS, IMAGE_STRENGTHS, IMAGE_STYLE_CHIPS, IMAGE_SAMPLES,
  type Aspect,
} from "../lib/presets";
import { cn } from "../lib/utils";

type ImageMode = "create" | "modify" | "enhance";

export function ImagePanel({ initialMode = "create" }: { initialMode?: ImageMode } = {}) {
  const { status, busyByOther, generateImage, upscale, cancel, loadedEntry } = useStudio();
  const [imageMode, setImageMode] = useState<ImageMode>(initialMode);
  const [aspect, setAspect] = useState<Aspect>("landscape");
  const [strength, setStrength] = useState<typeof IMAGE_STRENGTHS[0]>(IMAGE_STRENGTHS[1]);
  const [factor, setFactor] = useState<2 | 4>(2);
  const [prompt, setPrompt] = useState("");
  const [image, setImage] = useState("");

  useEffect(() => { setImageMode(initialMode); }, [initialMode]);

  // Hydrate from a loaded history entry (when user clicks an image/upscale entry)
  useEffect(() => {
    if (!loadedEntry) return;
    if (loadedEntry.kind !== "image" && loadedEntry.kind !== "upscale") return;
    const p = loadedEntry.params || {};
    setPrompt(loadedEntry.prompt || "");
    // Derive sub-mode from history entry
    if (loadedEntry.kind === "upscale") {
      setImageMode("enhance");
      if (typeof p.image === "string") setImage(p.image);
      if (p.factor === 2 || p.factor === 4) setFactor(p.factor);
    } else if (loadedEntry.mode === "modify") {
      setImageMode("modify");
      if (typeof p.image === "string") setImage(p.image);
      const matched = IMAGE_STRENGTHS.find((s) => Math.abs(s.value - (p.strength ?? 0.6)) < 0.05);
      if (matched) setStrength(matched);
    } else {
      setImageMode("create");
    }
    // Derive aspect from width/height
    const w = p.width ?? 1024, h = p.height ?? 1024;
    if (w > h * 1.2) setAspect("landscape");
    else if (h > w * 1.2) setAspect("portrait");
    else setAspect("square");
  }, [loadedEntry]);

  const running = status.phase === "running" || status.phase === "queued";
  const canSubmit = (() => {
    if (running) return false;
    if (imageMode === "create") return !!prompt.trim();
    if (imageMode === "modify") return !!prompt.trim() && !!image;
    if (imageMode === "enhance") return !!image;
    return false;
  })();

  const onGenerate = () => {
    if (imageMode === "enhance") {
      upscale({ image, factor });
    } else {
      const dims = IMAGE_DIMS[aspect];
      generateImage({
        image_mode: imageMode,
        prompt,
        negative: "",
        width: dims.width,
        height: dims.height,
        seed: 0,
        model: "flux",
        image,
        strength: imageMode === "modify" ? strength.value : 1.0,
      });
    }
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

  return (
    <div className="flex flex-col gap-6">
      {/* Sub-mode tabs */}
      <div className="grid grid-cols-3 bg-bg-muted p-1 rounded-xl gap-1">
        {([
          { key: "create",  label: "Create",  icon: Sparkles },
          { key: "modify",  label: "Modify",  icon: Wand2 },
          { key: "enhance", label: "Enhance", icon: Maximize },
        ] as const).map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setImageMode(key)}
            className={cn(
              "rounded-lg py-2 text-sm font-semibold transition-all flex items-center justify-center gap-1.5",
              imageMode === key
                ? "bg-bg-inset text-fg-base shadow-sm"
                : "text-fg-muted hover:text-fg-base",
            )}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </div>

      {imageMode === "create" && (
        <>
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

          <Section title="Describe your picture">
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="A cozy cabin on a snowy mountain at sunset, smoke rising from the chimney…"
              rows={4}
              className="text-base leading-relaxed"
            />
            <div className="flex flex-wrap gap-1.5 pt-1 items-center">
              <button
                onClick={improvePrompt}
                disabled={improving || pendingStyle !== null || !prompt.trim()}
                className="text-xs px-2.5 py-1 rounded-md border border-brand/40 bg-brand/5 text-brand hover:bg-brand/10 transition flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
                title="Use AI to enrich your prompt"
              >
                <Wand2 className={cn("h-3 w-3", improving && "animate-spin")} />
                {improving ? "Thinking…" : "Improve"}
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
                    {busy ? (
                      <Wand2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <span>+</span>
                    )}
                    {c.label}
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
        </>
      )}

      {imageMode === "modify" && (
        <>
          <Section title="Your photo" hint="We'll change it based on your description">
            <ImageUpload filename={image} onChange={setImage} />
          </Section>

          <Section title="What should change?">
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Make it look like an oil painting / change the season to autumn / add a rainbow…"
              rows={3}
            />
            <div className="flex flex-wrap gap-1.5 pt-1 items-center">
              <button
                onClick={improvePrompt}
                disabled={improving || pendingStyle !== null || !prompt.trim()}
                className="text-xs px-2.5 py-1 rounded-md border border-brand/40 bg-brand/5 text-brand hover:bg-brand/10 transition flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
                title="Use AI to enrich your prompt"
              >
                <Wand2 className={cn("h-3 w-3", improving && "animate-spin")} />
                {improving ? "Thinking…" : "Improve"}
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
                    {busy ? (
                      <Wand2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <span>+</span>
                    )}
                    {c.label}
                  </button>
                );
              })}
            </div>
          </Section>

          <Section title="How much change?">
            <div className="grid grid-cols-3 gap-2">
              {IMAGE_STRENGTHS.map((s) => {
                const active = strength.key === s.key;
                return (
                  <button
                    key={s.key}
                    onClick={() => setStrength(s)}
                    className={cn(
                      "rounded-xl border p-3 text-center transition-all",
                      active
                        ? "border-brand bg-brand/5 shadow-sm"
                        : "border-border bg-bg-subtle hover:border-border-strong",
                    )}
                  >
                    <div className={cn("text-sm font-semibold", active && "text-brand")}>
                      {s.label}
                    </div>
                    <div className="text-[10px] text-fg-subtle mt-0.5">{s.hint}</div>
                  </button>
                );
              })}
            </div>
          </Section>
        </>
      )}

      {imageMode === "enhance" && (
        <>
          <Section title="Your photo" hint="We'll make it bigger and sharper">
            <ImageUpload filename={image} onChange={setImage} />
          </Section>

          <Section title="Make it bigger by">
            <div className="grid grid-cols-2 gap-2">
              {[2, 4].map((f) => {
                const active = factor === f;
                return (
                  <button
                    key={f}
                    onClick={() => setFactor(f as 2 | 4)}
                    className={cn(
                      "rounded-xl border p-4 text-center transition-all",
                      active
                        ? "border-brand bg-brand/5 shadow-sm"
                        : "border-border bg-bg-subtle hover:border-border-strong",
                    )}
                  >
                    <div className={cn("text-2xl font-bold", active && "text-brand")}>
                      {f}×
                    </div>
                    <div className="text-[10px] text-fg-subtle mt-0.5">
                      {f === 2 ? "Bigger, faster" : "Maximum size"}
                    </div>
                  </button>
                );
              })}
            </div>
          </Section>
        </>
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
            {imageMode === "create" ? "Make my picture" :
             imageMode === "modify" ? "Apply the change" :
             "Make it bigger"}
          </button>
        )}
      </div>
    </div>
  );
}
