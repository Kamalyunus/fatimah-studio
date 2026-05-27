import { useEffect } from "react";
import { Trash2, X } from "lucide-react";
import { useStudio } from "../lib/store";
import { api } from "../lib/api";
import { cn } from "../lib/utils";

interface Props {
  open: boolean;
  onClose: () => void;
}

export function HistoryDrawer({ open, onClose }: Props) {
  const { history, deleteHistoryEntry, loadFromHistory, refreshHistory } = useStudio();

  useEffect(() => {
    if (open) refreshHistory();
  }, [open, refreshHistory]);

  return (
    <>
      <div
        className={cn(
          "fixed inset-0 z-40 bg-black/40 backdrop-blur-sm transition-opacity",
          open ? "opacity-100" : "opacity-0 pointer-events-none"
        )}
        onClick={onClose}
      />
      <div
        className={cn(
          "fixed inset-y-0 right-0 z-50 w-full sm:w-[440px] bg-bg-base border-l border-border shadow-2xl flex flex-col transition-transform duration-300",
          open ? "translate-x-0" : "translate-x-full"
        )}
      >
        <div className="h-14 border-b border-border flex items-center justify-between px-5 shrink-0">
          <div>
            <div className="font-semibold text-fg-base">History</div>
            <div className="text-xs text-fg-subtle">{history.length} generations</div>
          </div>
          <button
            onClick={onClose}
            className="h-8 w-8 rounded-lg hover:bg-bg-muted flex items-center justify-center text-fg-muted"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 overflow-auto p-3">
          {history.length === 0 ? (
            <div className="p-8 text-center text-sm text-fg-subtle">
              No generations yet. Run one to see it here.
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {history.map((h) => (
                <div
                  key={h.id}
                  className="group relative rounded-xl overflow-hidden border border-border hover:border-brand transition cursor-pointer bg-bg-subtle"
                  onClick={() => {
                    loadFromHistory(h);
                    onClose();
                  }}
                >
                  <div className="aspect-video bg-black relative">
                    <img
                      src={api.thumbUrl(h.filename)}
                      alt={h.prompt}
                      className="w-full h-full object-cover"
                      loading="lazy"
                    />
                    {/.(mp4|webm|gif)$/i.test(h.filename) && (
                      <video
                        src={api.videoUrl(h.filename)}
                        muted
                        loop
                        playsInline
                        preload="none"
                        className="absolute inset-0 w-full h-full object-cover opacity-0 group-hover:opacity-100 transition-opacity"
                        onMouseEnter={(e) => (e.currentTarget as HTMLVideoElement).play().catch(() => {})}
                        onMouseLeave={(e) => {
                          const v = e.currentTarget as HTMLVideoElement;
                          v.pause();
                          v.currentTime = 0;
                        }}
                      />
                    )}
                    {h.kind && h.kind !== "video" && (
                      <div className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded-md bg-black/60 text-white text-[10px] font-medium uppercase tracking-wider">
                        {h.kind}
                      </div>
                    )}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm("Delete this?")) {
                          deleteHistoryEntry(h.id, true);
                        }
                      }}
                      className="absolute top-1.5 right-1.5 h-7 w-7 rounded-md bg-black/60 hover:bg-danger text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition"
                      title="Delete"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                  <div className="p-2.5">
                    <div className="text-xs text-fg-base line-clamp-2 leading-snug">
                      {h.prompt || "(no prompt)"}
                    </div>
                    <div className="mt-1.5 flex items-center justify-between gap-2 text-[10px] text-fg-subtle">
                      {h.created_by_name ? (
                        <div className="flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-bg-muted text-fg-base">
                          <span className="leading-none">{h.created_by_emoji || "👤"}</span>
                          <span className="font-medium">{h.created_by_name}</span>
                        </div>
                      ) : <span />}
                      <span>{new Date(h.created_at * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
