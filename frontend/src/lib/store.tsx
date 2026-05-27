import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type {
  GenStatus, HistoryEntry,
  ImageGenParams, UpscaleParams, StorybookParams,
} from "../types";
import { api } from "./api";
import { loadProfile } from "./user";

function withProfile<T extends object>(p: T): T & { user_name: string; user_emoji: string } {
  const prof = loadProfile();
  return {
    ...p,
    user_name: prof?.name ?? "",
    user_emoji: prof?.emoji ?? "",
  } as T & { user_name: string; user_emoji: string };
}

interface Ctx {
  status: GenStatus;
  result: HistoryEntry | null;

  /** True when an active gen is running but was NOT submitted by this browser session. */
  busyByOther: boolean;

  /** Last entry loaded from history (or null). Components watch this to hydrate their local state. */
  loadedEntry: HistoryEntry | null;

  history: HistoryEntry[];
  refreshHistory: () => Promise<HistoryEntry[]>;
  deleteHistoryEntry: (id: string, hard?: boolean) => Promise<void>;
  loadFromHistory: (entry: HistoryEntry) => void;

  theme: "light" | "dark";
  toggleTheme: () => void;

  generateImage: (params: ImageGenParams) => Promise<void>;
  upscale: (params: UpscaleParams) => Promise<void>;
  generateStorybook: (params: StorybookParams) => Promise<void>;
  cancel: () => Promise<void>;
}

const StudioContext = createContext<Ctx | null>(null);

export function useStudio() {
  const ctx = useContext(StudioContext);
  if (!ctx) throw new Error("useStudio outside StudioProvider");
  return ctx;
}

type ServerState = Awaited<ReturnType<typeof api.state>>;

const MY_PROMPT_IDS_KEY = "wan-studio-my-prompt-ids";

function loadMyPromptIds(): string[] {
  try {
    return JSON.parse(localStorage.getItem(MY_PROMPT_IDS_KEY) || "[]");
  } catch {
    return [];
  }
}
function saveMyPromptIds(ids: string[]) {
  try {
    localStorage.setItem(MY_PROMPT_IDS_KEY, JSON.stringify(ids.slice(-20)));
  } catch {}
}
function addMyPromptId(id: string) {
  const ids = loadMyPromptIds();
  if (!ids.includes(id)) {
    ids.push(id);
    saveMyPromptIds(ids);
  }
}

export function StudioProvider({ children }: { children: ReactNode }) {
  const [serverState, setServerState] = useState<ServerState | null>(null);
  const [result, setResult] = useState<HistoryEntry | null>(null);
  const [loadedEntry, setLoadedEntry] = useState<HistoryEntry | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [localError, setLocalError] = useState<string | null>(null);
  const [theme, setTheme] = useState<"light" | "dark">(
    () => (document.documentElement.classList.contains("dark") ? "dark" : "light")
  );
  const lastActivePromptIdRef = useRef<string | null>(null);

  const refreshHistory = useCallback(async () => {
    try {
      const { items } = await api.history();
      setHistory(items);
      return items;
    } catch {
      return [];
    }
  }, []);

  // Poll server state — source of truth for the active gen.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async () => {
      try {
        const s = await api.state();
        if (cancelled) return;
        setServerState(s);

        // Completion transition: was active, now isn't, no error → save & show result
        const wasActiveId = lastActivePromptIdRef.current;
        if (wasActiveId && !s.active && !s.last_error) {
          lastActivePromptIdRef.current = null;
          const items = await refreshHistory();
          const completed = items.find((it) => it.prompt_id === wasActiveId);
          if (completed) setResult(completed);
        }
        if (s.active) {
          lastActivePromptIdRef.current = s.active.prompt_id;
        }
      } catch {
        // ignore transient errors
      } finally {
        if (!cancelled) {
          const delay = serverState?.active ? 1500 : 4000;
          timer = setTimeout(tick, delay);
        }
      }
    };

    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  const toggleTheme = useCallback(() => {
    setTheme((t) => {
      const next = t === "light" ? "dark" : "light";
      document.documentElement.classList.toggle("dark", next === "dark");
      try { localStorage.setItem("wan-studio-theme", next); } catch {}
      return next;
    });
  }, []);

  const loadFromHistory = useCallback((entry: HistoryEntry) => {
    // ImagePanel and StorybookPanel watch loadedEntry to hydrate their own state.
    setResult(entry);
    setLoadedEntry(entry);
  }, []);

  const cancel = useCallback(async () => {
    try {
      await api.interrupt();
    } catch {}
    setServerState((s) => s ? { ...s, active: null } : s);
  }, []);

  const generateImage = useCallback(async (p: ImageGenParams) => {
    setLocalError(null);
    if (!p.prompt.trim()) {
      setLocalError("Please write a prompt first.");
      return;
    }
    if (p.image_mode === "modify" && !p.image) {
      setLocalError("Please upload a photo first.");
      return;
    }
    try {
      const sub = await api.generateImage(withProfile(p));
      addMyPromptId(sub.prompt_id);
      lastActivePromptIdRef.current = sub.prompt_id;
      setResult(null);
    } catch (e) {
      const msg = (e as Error).message;
      setLocalError(/409/.test(msg) ? "Someone is already making something. Wait a moment." : msg);
    }
  }, []);

  const generateStorybook = useCallback(async (p: StorybookParams) => {
    setLocalError(null);
    if (!p.story.trim()) { setLocalError("Please write a story first."); return; }
    try {
      const sub = await api.storybook(withProfile(p));
      addMyPromptId(sub.prompt_id);
      lastActivePromptIdRef.current = sub.prompt_id;
      setResult(null);
    } catch (e) {
      const msg = (e as Error).message;
      if (/503/.test(msg)) setLocalError("The story-writing helper isn't ready yet. Try again in a minute.");
      else if (/409/.test(msg)) setLocalError("Someone is already making something. Wait a moment.");
      else setLocalError(msg);
    }
  }, []);

  const upscale = useCallback(async (p: UpscaleParams) => {
    setLocalError(null);
    if (!p.image) {
      setLocalError("Please upload a photo first.");
      return;
    }
    try {
      const sub = await api.upscale(withProfile(p));
      addMyPromptId(sub.prompt_id);
      lastActivePromptIdRef.current = sub.prompt_id;
      setResult(null);
    } catch (e) {
      const msg = (e as Error).message;
      setLocalError(/409/.test(msg) ? "Someone is already making something. Wait a moment." : msg);
    }
  }, []);

  const deleteHistoryEntry = useCallback(
    async (id: string, hard = false) => {
      await api.deleteHistory(id, hard);
      refreshHistory();
    },
    [refreshHistory],
  );

  const status: GenStatus = useMemo(() => {
    if (localError) return { phase: "error", message: localError };
    if (serverState?.last_error)
      return { phase: "error", message: serverState.last_error };
    if (serverState?.active) {
      const a = serverState.active;
      return {
        phase: "running",
        node: a.node ?? undefined,
        step: a.step || undefined,
        totalSteps: a.total_steps || undefined,
        startedAt: a.started_at,
        kind: a.kind,
        previewImages: a.preview_images,
        sceneDescriptions: a.scene_descriptions,
        character: a.character,
      };
    }
    if (result) {
      return { phase: "done", filename: result.filename, durationS: result.duration_s ?? 0 };
    }
    return { phase: "idle" };
  }, [serverState, result, localError]);

  const busyByOther = useMemo(() => {
    if (!serverState?.active) return false;
    const myIds = loadMyPromptIds();
    return !myIds.includes(serverState.active.prompt_id);
  }, [serverState]);

  return (
    <StudioContext.Provider
      value={{
        status,
        result,
        loadedEntry,
        busyByOther,
        history,
        refreshHistory,
        deleteHistoryEntry,
        loadFromHistory,
        theme,
        toggleTheme,
        generateImage,
        upscale,
        generateStorybook,
        cancel,
      }}
    >
      {children}
    </StudioContext.Provider>
  );
}
