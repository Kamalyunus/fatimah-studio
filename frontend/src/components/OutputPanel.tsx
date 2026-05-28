import { useEffect, useState } from "react";
import { Download, Copy, Check, X, RefreshCw, UserPlus, AlertTriangle } from "lucide-react";
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

  // Keyframe approval gate: the storybook paused after generating all Flux frames,
  // waiting for the user to approve or regenerate before kicking off the heavy Wan pass.
  const awaitingApproval =
    status.phase === "running"
    && status.kind === "storybook"
    && status.node === "awaiting-approval"
    && (status.keyframes?.length ?? 0) > 0;

  // Storybook live preview: thumbnails of pages already illustrated, while animation continues
  const storybookPreview =
    status.phase === "running" &&
    status.kind === "storybook" &&
    !awaitingApproval &&
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

      {awaitingApproval && (
        <ApprovalGate
          keyframes={status.phase === "running" ? status.keyframes ?? [] : []}
          character={character}
        />
      )}

      {status.phase === "done" && result?.kind === "storybook" && (
        <>
          <SaveCharacterPanel gen_id={result.id} default_name={character ?? ""} />
          <ScenesStrip entry={result} />
        </>
      )}

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


/** Keyframe approval gate (#2). Shows every scene's start+end pair after the Flux
 * phase, with regen buttons + Approve / Cancel. While this is mounted the storybook
 * orchestrator is blocked on an asyncio.Event waiting for one of the buttons. */
function ApprovalGate({
  keyframes, character,
}: {
  keyframes: Array<{
    scene_index: number;
    start_image: string;
    end_image: string;
    description?: string;
    motion_intensity?: string;
    drift?: number | null;
    drift_flagged?: boolean | null;
  }>;
  character?: string;
}) {
  const flaggedCount = keyframes.filter((k) => k.drift_flagged).length;
  const [busy, setBusy] = useState<string | null>(null); // e.g. "regen-3-end" / "approve"
  const onRegen = async (scene_index: number, frame: "start" | "end") => {
    const key = `regen-${scene_index}-${frame}`;
    if (busy) return;
    setBusy(key);
    try { await api.storybookRegenKeyframe(scene_index, frame); }
    catch (e) { alert((e as Error).message); }
    finally { setBusy(null); }
  };
  const onApprove = async () => {
    if (busy) return;
    setBusy("approve");
    try { await api.storybookApprove(); }
    catch (e) { alert((e as Error).message); }
    finally { setBusy(null); }
  };
  const onCancel = async () => {
    if (busy) return;
    if (!confirm("Cancel the storybook? Generated keyframes will be discarded.")) return;
    setBusy("cancel");
    try { await api.storybookCancelApproval(); }
    catch (e) { alert((e as Error).message); }
    finally { setBusy(null); }
  };

  return (
    <div className="panel p-4 animate-fade-in space-y-3 border-brand/40">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <div className="text-sm font-semibold text-fg-base">
            👋 Pick your keyframes
          </div>
          <div className="text-xs text-fg-muted mt-0.5">
            Approve to animate ({keyframes.length} scenes), or regenerate any that look off.
            Pages 2+ start with the previous page's end image, so regenerate the end frame to fix continuity.
            {flaggedCount > 0 && (
              <span className="ml-1 text-amber-500 font-medium">
                · {flaggedCount} scene{flaggedCount === 1 ? "" : "s"} may have character drift — flagged with ⚠.
              </span>
            )}
          </div>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={onCancel} disabled={!!busy}>
            <X className="h-3.5 w-3.5" /> Cancel
          </Button>
          <button
            onClick={onApprove}
            disabled={!!busy}
            className="inline-flex items-center gap-1.5 px-4 h-9 rounded-lg bg-gradient-to-r from-brand to-indigo-500 text-white text-sm font-semibold shadow-sm hover:opacity-95 disabled:opacity-50"
          >
            <Check className="h-4 w-4" />
            {busy === "approve" ? "Starting…" : "Approve & animate"}
          </button>
        </div>
      </div>
      {character && (
        <div className="text-xs text-fg-muted italic border-l-2 border-brand pl-2 py-0.5">
          <span className="font-semibold not-italic text-fg-base">Main character:</span> {character}
        </div>
      )}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {keyframes.map((kf) => {
          const startBusy = busy === `regen-${kf.scene_index}-start`;
          const endBusy = busy === `regen-${kf.scene_index}-end`;
          const startRegenAllowed = kf.scene_index === 0;
          return (
            <div
              key={kf.scene_index}
              className={cn(
                "rounded-xl border bg-bg-subtle overflow-hidden",
                kf.drift_flagged ? "border-amber-500/60" : "border-border",
              )}
            >
              <div className="px-2.5 pt-2 pb-1 flex items-center justify-between gap-2">
                <div className="text-xs font-semibold">Page {kf.scene_index + 1}</div>
                <div className="flex items-center gap-2">
                  {kf.drift_flagged && (
                    <div
                      className="flex items-center gap-1 text-amber-500 text-[10px] font-semibold"
                      title={
                        typeof kf.drift === "number"
                          ? `Character similarity to reference: ${(kf.drift * 100).toFixed(0)}% — below threshold. Try regenerating the start frame.`
                          : "Character may have drifted from reference."
                      }
                    >
                      <AlertTriangle className="h-3 w-3" />
                      drift
                    </div>
                  )}
                  {kf.motion_intensity && (
                    <div className="text-[10px] text-fg-subtle uppercase tracking-wider">
                      {kf.motion_intensity}
                    </div>
                  )}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-1 p-1">
                <div className="relative">
                  <img src={api.imageUrl(kf.start_image)} alt={`start ${kf.scene_index + 1}`}
                       className="w-full aspect-video object-cover rounded-md" loading="lazy" />
                  <div className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-black/60 text-white text-[9px] font-bold">START</div>
                  {startRegenAllowed && (
                    <button
                      onClick={() => onRegen(kf.scene_index, "start")}
                      disabled={!!busy}
                      title="Regenerate this start frame"
                      className="absolute bottom-1 right-1 h-7 w-7 rounded-md bg-black/60 hover:bg-brand text-white flex items-center justify-center transition disabled:opacity-50"
                    >
                      <RefreshCw className={cn("h-3.5 w-3.5", startBusy && "animate-spin")} />
                    </button>
                  )}
                </div>
                <div className="relative">
                  <img src={api.imageUrl(kf.end_image)} alt={`end ${kf.scene_index + 1}`}
                       className="w-full aspect-video object-cover rounded-md" loading="lazy" />
                  <div className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-black/60 text-white text-[9px] font-bold">END</div>
                  <button
                    onClick={() => onRegen(kf.scene_index, "end")}
                    disabled={!!busy}
                    title="Regenerate this end frame"
                    className="absolute bottom-1 right-1 h-7 w-7 rounded-md bg-black/60 hover:bg-brand text-white flex items-center justify-center transition disabled:opacity-50"
                  >
                    <RefreshCw className={cn("h-3.5 w-3.5", endBusy && "animate-spin")} />
                  </button>
                </div>
              </div>
              {kf.description && (
                <div className="px-2.5 py-2 text-[11px] leading-snug text-fg-muted line-clamp-2" title={kf.description}>
                  {kf.description}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}


/** Per-scene strip after a storybook completes. Each tile previews the scene's first
 * frame + has a regen button to re-animate just that scene (#3). */
function ScenesStrip({ entry }: { entry: any }) {
  const scenes: Array<{
    scene_index: number;
    start_image?: string;
    description?: string;
    motion_intensity?: string;
  }> = entry?.params?.scenes_meta ?? [];
  const [busyScene, setBusyScene] = useState<number | null>(null);
  if (!scenes.length) return null;
  const onRegen = async (i: number) => {
    if (busyScene !== null) return;
    if (!confirm(`Regenerate scene ${i + 1}? Takes ~6 min.`)) return;
    setBusyScene(i);
    try {
      await api.storybookRegenScene(entry.id, i);
      // The store's poller will pick up the active gen and we'll re-enter the running phase.
    } catch (e) {
      alert((e as Error).message);
    } finally {
      // Don't clear busyScene; we want the button to stay disabled until phase transitions.
      setTimeout(() => setBusyScene(null), 2000);
    }
  };
  return (
    <div className="panel p-4 animate-fade-in space-y-3">
      <div className="text-xs font-medium text-fg-muted">🎬 Scenes</div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
        {scenes.map((s) => {
          const busy = busyScene === s.scene_index;
          return (
            <div key={s.scene_index} className="rounded-xl border border-border bg-bg-subtle overflow-hidden">
              <div className="relative aspect-video bg-bg-muted">
                {s.start_image && (
                  <img src={api.imageUrl(s.start_image)} alt={`scene ${s.scene_index + 1}`}
                       className="w-full h-full object-cover" loading="lazy" />
                )}
                <div className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-black/60 text-white text-[10px] font-semibold">
                  {s.scene_index + 1}
                </div>
                <button
                  onClick={() => onRegen(s.scene_index)}
                  disabled={busyScene !== null}
                  title="Re-animate this scene"
                  className="absolute bottom-1 right-1 h-7 w-7 rounded-md bg-black/60 hover:bg-brand text-white flex items-center justify-center transition disabled:opacity-50"
                >
                  <RefreshCw className={cn("h-3.5 w-3.5", busy && "animate-spin")} />
                </button>
              </div>
              {s.description && (
                <div className="px-2 py-1.5 text-[10px] leading-snug text-fg-muted line-clamp-2" title={s.description}>
                  {s.description}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}


/** Shown after a storybook completes — lets the user save the protagonist for re-use. */
function SaveCharacterPanel({ gen_id, default_name }: { gen_id: string; default_name: string }) {
  const [name, setName] = useState(() => default_name.split(",")[0].trim().split(" ")[0] || "");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const onSave = async () => {
    if (!name.trim() || saving) return;
    setSaving(true);
    try {
      await api.saveCharacter(name.trim(), gen_id);
      setSaved(true);
    } catch (e) {
      alert((e as Error).message);
    } finally {
      setSaving(false);
    }
  };
  if (saved) {
    return (
      <div className="panel p-3 text-xs text-fg-muted flex items-center gap-2">
        <Check className="h-3.5 w-3.5 text-success" />
        Character saved. Pick it next time in the Storybook tab.
      </div>
    );
  }
  return (
    <div className="panel p-3 flex items-center gap-2 flex-wrap">
      <UserPlus className="h-4 w-4 text-brand" />
      <div className="text-sm font-medium">Save this character?</div>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Character name"
        className="input h-8 px-2 text-sm min-w-[140px]"
      />
      <Button variant="ghost" size="sm" onClick={onSave} disabled={!name.trim() || saving}>
        {saving ? "Saving…" : "Save for later"}
      </Button>
    </div>
  );
}
