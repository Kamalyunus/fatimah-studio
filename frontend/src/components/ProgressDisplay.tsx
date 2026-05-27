import { useEffect, useState } from "react";
import { X, AlertCircle, CheckCircle2 } from "lucide-react";
import type { GenStatus } from "../types";
import { Button } from "./ui";
import { cn, fmtSeconds } from "../lib/utils";

interface Props {
  status: GenStatus;
  onCancel: () => void;
}

function Ring({ progress }: { progress: number }) {
  const R = 36;
  const C = 2 * Math.PI * R;
  return (
    <svg viewBox="0 0 88 88" className="h-20 w-20">
      <circle cx="44" cy="44" r={R} className="fill-none stroke-bg-muted" strokeWidth="6" />
      <circle
        cx="44"
        cy="44"
        r={R}
        className="fill-none stroke-brand transition-all duration-500"
        strokeWidth="6"
        strokeLinecap="round"
        strokeDasharray={C}
        strokeDashoffset={C * (1 - progress)}
        transform="rotate(-90 44 44)"
      />
    </svg>
  );
}

export function ProgressDisplay({ status, onCancel }: Props) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (status.phase !== "running") return;
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, [status.phase]);
  void tick;

  if (status.phase === "idle") {
    return (
      <div className="flex flex-col items-center justify-center py-10 px-6 text-center">
        <div className="h-20 w-20 rounded-full bg-bg-muted flex items-center justify-center mb-3">
          <svg className="h-8 w-8 text-fg-subtle" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        </div>
        <div className="text-sm font-medium text-fg-base">No generation yet</div>
        <div className="text-xs text-fg-subtle mt-1">Hit Generate to see your video here</div>
      </div>
    );
  }

  if (status.phase === "queued") {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-center animate-fade-in">
        <Ring progress={0.05} />
        <div className="mt-3 text-sm font-medium">Queued…</div>
      </div>
    );
  }

  if (status.phase === "running") {
    const node = status.node ?? "loading";
    const step = status.step ?? 0;
    const total = status.totalSteps ?? 0;
    const samplerActive = node === "sampler" && total > 0 && total <= 100;

    // Storybook macro-progress nodes
    const storybookMatch = node?.match(/^page-(\d+)-(image|animate)$/);
    const transitionMatch = node?.match(/^transition-(\d+)$/);
    const narrationMatch = node?.match(/^narration-(\d+)$/);
    const isStorybookNode =
      node === "planning" || node === "stitching" || node === "narrating"
      || !!storybookMatch || !!transitionMatch || !!narrationMatch;

    let ratio: number;
    if (samplerActive) {
      ratio = step / total;
    } else if (isStorybookNode && total > 0) {
      // step / total goes 0..1 across the whole storybook
      ratio = Math.min(0.99, step / total);
    } else if (node === "save") {
      ratio = 0.95;
    } else {
      ratio = 0.15;
    }

    const elapsed = Date.now() / 1000 - status.startedAt;
    const eta =
      samplerActive && step > 0 ? (elapsed / step) * (total - step) :
      (isStorybookNode && total > 0 && step > 0) ? (elapsed / step) * (total - step) :
      null;

    // Cycle through encouraging messages based on phase + progress
    const samplerMessages = [
      "Sketching the scene…",
      "Adding shapes and colors…",
      "Painting in the details…",
      "Bringing it to life…",
      "Polishing the final touches…",
      "Almost there!",
    ];
    let friendly: string;
    let subtitle = "";
    if (samplerActive) {
      const idx = Math.min(
        samplerMessages.length - 1,
        Math.floor((step / total) * samplerMessages.length),
      );
      friendly = samplerMessages[idx];
    } else if (node === "planning") {
      friendly = "Writing the story…";
    } else if (node === "stitching") {
      friendly = "Putting it all together…";
    } else if (storybookMatch) {
      const pageNum = parseInt(storybookMatch[1], 10);
      const kind = storybookMatch[2];
      const totalPages = total > 0 ? Math.ceil(total / 2) : 0;
      friendly = kind === "image"
        ? `Illustrating page ${pageNum}…`
        : `Animating page ${pageNum}…`;
      if (totalPages) subtitle = `Page ${pageNum} of ${totalPages}`;
    } else if (transitionMatch) {
      const idx = parseInt(transitionMatch[1], 10);
      friendly = "Blending pages together…";
      subtitle = `Transition ${idx}`;
    } else if (narrationMatch) {
      const idx = parseInt(narrationMatch[1], 10);
      friendly = "Recording the storyteller's voice…";
      subtitle = `Page ${idx}`;
    } else if (node === "narrating") {
      friendly = "Preparing the storyteller…";
    } else {
      const labelMap: Record<string, string> = {
        model_loader: "Warming up the model…",
        t5: "Warming up…",
        vae: "Warming up…",
        clip_vision: "Warming up…",
        checkpoint: "Loading the model…",
        unet: "Loading the model…",
        clip: "Reading your words…",
        lora: "Tuning the model…",
        load_image: "Looking at your photo…",
        encode: "Understanding your photo…",
        text_encode: "Reading your prompt…",
        positive: "Reading your prompt…",
        negative: "Setting things up…",
        guidance: "Setting things up…",
        clip_vision_encode: "Studying your photo…",
        empty_embeds: "Getting ready…",
        empty_latent: "Getting ready…",
        i2v_encode: "Preparing the canvas…",
        block_swap: "Spinning up both GPUs…",
        upscale_model: "Loading the enhancer…",
        upscale: "Making it bigger and sharper…",
        scale_down: "Tidying up the size…",
        sampler: "Sketching the scene…",
        decode: "Rendering the final picture…",
        interpolate: "Adding smooth motion…",
        save: "Saving your creation…",
      };
      friendly = labelMap[node] ?? node;
    }

    return (
      <div className="flex flex-col items-center justify-center py-8 px-6 text-center animate-fade-in">
        <div className="relative">
          <Ring progress={ratio} />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-base font-semibold tabular-nums">
              {samplerActive || (isStorybookNode && total > 0)
                ? `${Math.round(ratio * 100)}%`
                : "•••"}
            </span>
          </div>
        </div>
        <div className="mt-4 text-base font-semibold text-fg-base">{friendly}</div>
        <div className="mt-1 text-xs text-fg-muted tabular-nums">
          {samplerActive ? `frame ${step} of ${total} · ` : ""}
          {subtitle ? subtitle + " · " : ""}
          {fmtSeconds(elapsed)} elapsed
          {eta != null && eta > 0 ? ` · ~${fmtSeconds(eta)} left` : ""}
        </div>
        <Button variant="ghost" size="sm" className="mt-4" onClick={onCancel}>
          <X className="h-3.5 w-3.5" /> Cancel
        </Button>
      </div>
    );
  }

  if (status.phase === "error") {
    const msg = status.message ?? "";
    const friendly =
      /out of memory|OOM/i.test(msg)
        ? "Ran out of GPU memory. Try a smaller Quality or simpler prompt."
        : /please.*prompt/i.test(msg)
        ? "Please write a prompt first."
        : /upload.*image|start image/i.test(msg)
        ? "Please upload a photo first."
        : /ComfyUI rejected/i.test(msg) || /502/.test(msg)
        ? "The video engine couldn't handle that. Try again or pick a smaller Quality."
        : /websocket|ws_error|connect/i.test(msg)
        ? "Lost connection to the engine. Try again in a moment."
        : msg;
    return (
      <div className={cn("p-4 rounded-xl border bg-danger/5 border-danger/30 animate-fade-in max-w-md")}>
        <div className="flex items-start gap-3">
          <AlertCircle className="h-5 w-5 text-danger shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold text-danger">Something went wrong</div>
            <div className="text-sm text-fg-base mt-1 break-words">{friendly}</div>
          </div>
        </div>
      </div>
    );
  }

  if (status.phase === "cancelled") {
    return (
      <div className="p-4 rounded-xl border border-border bg-bg-muted/50 text-sm text-fg-muted animate-fade-in">
        Stopped.
      </div>
    );
  }

  if (status.phase === "done") {
    return (
      <div className="p-3 rounded-xl border border-success/30 bg-success/5 flex items-center gap-2 animate-fade-in">
        <CheckCircle2 className="h-4 w-4 text-success" />
        <div className="text-sm">
          <span className="font-semibold">Done</span>
          <span className="text-fg-muted"> · {fmtSeconds(status.durationS)}</span>
        </div>
      </div>
    );
  }
  return null;
}
