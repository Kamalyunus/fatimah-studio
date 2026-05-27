import { useCallback, useRef, useState } from "react";
import { Upload, X } from "lucide-react";
import { api } from "../lib/api";
import { cn } from "../lib/utils";

interface Props {
  filename: string;
  onChange: (filename: string) => void;
}

export function ImageUpload({ filename, onChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      setBusy(true);
      try {
        const url = URL.createObjectURL(file);
        setPreviewUrl(url);
        const { filename } = await api.upload(file);
        onChange(filename);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [onChange]
  );

  return (
    <div
      className={cn(
        "relative rounded-2xl border-2 border-dashed transition-all overflow-hidden",
        dragging ? "border-brand bg-brand-subtle" : "border-border bg-bg-subtle/40",
        previewUrl ? "border-solid" : ""
      )}
      onDragEnter={(e) => { e.preventDefault(); setDragging(true); }}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault(); setDragging(false);
        const f = e.dataTransfer?.files?.[0];
        if (f) handleFile(f);
      }}
    >
      {previewUrl ? (
        <div className="relative">
          <img src={previewUrl} alt="start" className="w-full h-56 object-cover" />
          <button
            onClick={() => {
              setPreviewUrl(null);
              onChange("");
            }}
            className="absolute top-2 right-2 h-8 w-8 rounded-full bg-black/60 text-white flex items-center justify-center hover:bg-black/80"
            title="Remove"
          >
            <X className="h-4 w-4" />
          </button>
          {filename && (
            <div className="absolute bottom-2 left-2 px-2 py-1 rounded-md bg-black/60 text-white/90 text-[10px] font-mono">
              {filename}
            </div>
          )}
        </div>
      ) : (
        <button
          onClick={() => inputRef.current?.click()}
          className="w-full h-56 flex flex-col items-center justify-center text-fg-muted hover:text-fg-base transition"
        >
          <Upload className="h-8 w-8 mb-2" />
          <div className="text-sm font-medium">
            {busy ? "Uploading…" : "Drop image or click to upload"}
          </div>
          <div className="text-xs text-fg-subtle mt-1">PNG, JPG, WebP</div>
        </button>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
        }}
      />
      {error && <div className="px-3 py-2 text-xs text-danger">{error}</div>}
    </div>
  );
}
