import { useEffect, useState } from "react";

const COLORS = ["#ea6046", "#f59e0b", "#fbbf24", "#34d399", "#60a5fa", "#a78bfa", "#f472b6"];

/** Fires a one-shot confetti burst whenever `trigger` flips truthy/changes. */
export function Confetti({ trigger }: { trigger: unknown }) {
  const [pieces, setPieces] = useState<Array<{ id: number; left: string; bg: string; dx: string; dur: string; delay: string }>>([]);

  useEffect(() => {
    if (!trigger) return;
    const n = 80;
    const next = Array.from({ length: n }).map((_, i) => {
      const startLeft = Math.random() * 100;
      const drift = (Math.random() - 0.5) * 600; // horizontal drift in px
      const dur = (1.6 + Math.random() * 1.4).toFixed(2) + "s";
      const delay = (Math.random() * 0.4).toFixed(2) + "s";
      return {
        id: i,
        left: `${startLeft}vw`,
        bg: COLORS[Math.floor(Math.random() * COLORS.length)],
        dx: `${drift}px`,
        dur,
        delay,
      };
    });
    setPieces(next);
    const t = setTimeout(() => setPieces([]), 3500);
    return () => clearTimeout(t);
  }, [trigger]);

  return (
    <>
      {pieces.map((p) => (
        <span
          key={p.id}
          className="confetti-piece"
          style={
            {
              left: p.left,
              background: p.bg,
              ["--dx" as any]: p.dx,
              ["--dur" as any]: p.dur,
              animationDelay: p.delay,
            } as React.CSSProperties
          }
        />
      ))}
    </>
  );
}
