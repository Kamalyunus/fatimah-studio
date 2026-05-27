import { useEffect, useState } from "react";
import { Download, Copy } from "lucide-react";
import { useStudio } from "../lib/store";
import { Button } from "./ui";
import { ProgressDisplay } from "./ProgressDisplay";
import { Confetti } from "./Confetti";
import { api } from "../lib/api";
import { cn, fmtSeconds } from "../lib/utils";

function isVideoFilename(fn: string | null | undefined): boolean {
  if (!fn) return false;
  return /\.(mp4|webm|mov|gif)$/i.test(fn);
}

export function OutputPanel() {
  const { status, result, cancel } = useStudio();
  const [celebrate, setCelebrate] = useState(0);
  const filename =
    result?.filename || (status.phase === "done" ? status.filename : null);
  const showOutput = !!filename;
  const idleOrShown = status.phase === "idle" || status.phase === "done";
  const isVideo = isVideoFilename(filename);

  // Storybook live preview: thumbnails of pages already illustrated, while animation continues
  const storybookPreview =
    status.phase === "running" &&
    status.kind === "storybook" &&
    (status.previewImages?.length ?? 0) > 0
      ? status.previewImages
      : null;
  const sceneDescs =
    status.phase === "running" ? status.sceneDescriptions ?? [] : [];
  const character =
    status.phase === "running" ? status.character : undefined;
  const totalPages =
    status.phase === "running" && status.totalSteps
      ? Math.ceil(status.totalSteps / 2)
      : 0;

  // Fire confetti on phase transition to "done"
  useEffect(() => {
    if (status.phase === "done") setCelebrate((n) => n + 1);
  }, [status.phase]);

  const copyPrompt = async () => {
    if (result) await navigator.clipboard.writeText(result.prompt);
  };

  return (
    <div className="space-y-4">
      <Confetti trigger={celebrate} />
      <div className="panel overflow-hidden">
        {showOutput && idleOrShown ? (
          <div className="relative bg-black flex items-center justify-center" style={{ minHeight: 320 }}>
            {isVideo ? (
              <video
                key={filename}
                src={api.videoUrl(filename!)}
                autoPlay
                loop
                muted
                playsInline
                controls
                className="w-full max-h-[78vh] object-contain"
              />
            ) : (
              <img
                key={filename}
                src={api.imageUrl(filename!)}
                alt={result?.prompt ?? "output"}
                className="w-full max-h-[78vh] object-contain"
              />
            )}
          </div>
        ) : (
          <div className="aspect-video flex items-center justify-center bg-gradient-to-br from-bg-subtle to-bg-muted">
            <ProgressDisplay status={status} onCancel={cancel} />
          </div>
        )}
      </div>

      {storybookPreview && (
        <div className="panel p-4 animate-fade-in space-y-3">
          <div className="flex items-baseline justify-between">
            <div className="text-xs font-medium text-fg-muted">
              ✨ Pages illustrated ({storybookPreview.length}/{totalPages || "?"})
            </div>
          </div>
          {character && (
            <div className="text-xs text-fg-muted italic border-l-2 border-brand pl-2 py-0.5">
              <span className="font-semibold not-italic text-fg-base">Main character:</span> {character}
            </div>
          )}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
            {Array.from({ length: totalPages || storybookPreview.length }).map((_, i) => {
              const fn = storybookPreview[i];
              const done = !!fn;
              const active = i === storybookPreview.length;
              const desc = sceneDescs[i] || "";
              return (
                <div
                  key={i}
                  className={cn(
                    "rounded-xl overflow-hidden border bg-bg-subtle",
                    done ? "border-border" : "border-dashed border-border",
                    active && "ring-2 ring-brand ring-offset-1 ring-offset-bg-inset animate-soft-pulse",
                  )}
                >
                  <div className="relative aspect-video bg-bg-muted">
                    {done ? (
                      <img
                        src={api.imageUrl(fn)}
                        alt={`page ${i + 1}`}
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-xs text-fg-subtle">
                        {active ? "illustrating…" : `page ${i + 1}`}
                      </div>
                    )}
                    <div className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded-md bg-black/60 text-white text-[10px] font-semibold">
                      Page {i + 1}
                    </div>
                  </div>
                  {desc && (
                    <div className="px-2.5 py-2 text-[11px] leading-snug text-fg-muted line-clamp-3" title={desc}>
                      {desc}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {(showOutput || status.phase === "done") && (
        <div className="flex items-center gap-2 flex-wrap">
          {filename && (
            <a
              href={isVideo ? api.videoUrl(filename) : api.imageUrl(filename)}
              download
              className="inline-flex items-center gap-1.5 px-3 h-9 rounded-lg bg-brand-subtle text-brand text-sm font-medium hover:opacity-80 transition"
            >
              <Download className="h-3.5 w-3.5" />
              Save to computer
            </a>
          )}
          {result?.prompt && (
            <Button variant="ghost" size="sm" onClick={copyPrompt}>
              <Copy className="h-3.5 w-3.5" />
              Copy prompt
            </Button>
          )}
          {result?.duration_s != null && status.phase === "done" && (
            <div className="ml-auto text-xs text-fg-subtle">
              Ready in {fmtSeconds(result.duration_s)}
            </div>
          )}
        </div>
      )}

      {(status.phase === "error" || status.phase === "cancelled") && (
        <ProgressDisplay status={status} onCancel={cancel} />
      )}
    </div>
  );
}
