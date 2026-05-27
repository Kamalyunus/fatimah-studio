import { useEffect, useState } from "react";
import { Image as ImageIcon, BookOpen } from "lucide-react";
import { StudioProvider, useStudio } from "./lib/store";
import { Header } from "./components/Header";
import { ImagePanel } from "./components/ImagePanel";
import { StorybookPanel } from "./components/StorybookPanel";
import { OutputPanel } from "./components/OutputPanel";
import { HistoryDrawer } from "./components/HistoryDrawer";
import { HomeScreen } from "./components/HomeScreen";
import { AvatarPicker } from "./components/AvatarPicker";
import { api } from "./lib/api";
import { cn } from "./lib/utils";
import { loadProfile, type UserProfile } from "./lib/user";

type TopMode = "image" | "storybook";
type ImageSubMode = "create" | "modify" | "enhance";
type View = "home" | "studio";

function ModeTabs({ mode, onChange }: { mode: TopMode; onChange: (m: TopMode) => void }) {
  const tabs: Array<{ key: TopMode; label: string; icon: typeof ImageIcon }> = [
    { key: "storybook", label: "Storybook", icon: BookOpen },
    { key: "image", label: "Picture", icon: ImageIcon },
  ];
  return (
    <div className="flex bg-bg-muted p-1 rounded-xl mb-4">
      {tabs.map(({ key, label, icon: Icon }) => (
        <button
          key={key}
          onClick={() => onChange(key)}
          className={cn(
            "flex-1 rounded-lg py-2.5 text-xs sm:text-sm font-semibold transition-all flex items-center justify-center gap-1.5",
            mode === key
              ? "bg-bg-inset text-fg-base shadow-sm"
              : "text-fg-muted hover:text-fg-base",
          )}
        >
          <Icon className="h-4 w-4" /> {label}
        </button>
      ))}
    </div>
  );
}

function Studio() {
  const { loadedEntry } = useStudio();
  const [historyOpen, setHistoryOpen] = useState(false);
  const [comfyOk, setComfyOk] = useState<boolean | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(() => loadProfile());
  const [showAvatarPicker, setShowAvatarPicker] = useState(false);
  const [view, setView] = useState<View>(() => {
    try {
      return (localStorage.getItem("fatimah-studio-view") as View) ?? "home";
    } catch {
      return "home";
    }
  });
  const [topMode, setTopMode] = useState<TopMode>(() => {
    try {
      const saved = localStorage.getItem("fatimah-studio-top-mode") as TopMode | null;
      return saved === "image" || saved === "storybook" ? saved : "storybook";
    } catch {
      return "storybook";
    }
  });
  const [imageSubMode, setImageSubMode] = useState<ImageSubMode>("create");

  // Route a clicked history entry to the right tab
  useEffect(() => {
    if (!loadedEntry) return;
    const kind = loadedEntry.kind;
    const mode = loadedEntry.mode;
    if (kind === "image" || kind === "upscale" || mode === "create" || mode === "modify" || mode === "upscale") {
      setTopMode("image");
      if (kind === "upscale" || mode === "upscale") setImageSubMode("enhance");
      else if (mode === "modify") setImageSubMode("modify");
      else setImageSubMode("create");
    } else {
      // storybook entries (or any other video output) → storybook tab
      setTopMode("storybook");
    }
    setView("studio");
  }, [loadedEntry]);

  useEffect(() => {
    if (!profile) setShowAvatarPicker(true);
  }, [profile]);

  useEffect(() => { try { localStorage.setItem("fatimah-studio-view", view); } catch {} }, [view]);
  useEffect(() => { try { localStorage.setItem("fatimah-studio-top-mode", topMode); } catch {} }, [topMode]);

  useEffect(() => {
    const tick = async () => {
      try {
        const h = await api.health();
        setComfyOk(h.ok);
      } catch {
        setComfyOk(false);
      }
    };
    tick();
    const id = setInterval(tick, 15000);
    return () => clearInterval(id);
  }, []);

  const onPickImage = (sub: ImageSubMode) => {
    setTopMode("image");
    setImageSubMode(sub);
    setView("studio");
  };

  return (
    <div className="min-h-screen flex flex-col bg-bg-base">
      <Header
        onOpenHistory={() => setHistoryOpen(true)}
        comfyOk={comfyOk}
        profile={profile}
        onEditProfile={() => setShowAvatarPicker(true)}
        onGoHome={view === "studio" ? () => setView("home") : undefined}
      />

      {view === "home" ? (
        <main className="flex-1">
          <HomeScreen
            profile={profile ?? { name: "friend", emoji: "🌸" }}
            onPickImageCreate={() => onPickImage("create")}
            onPickImageModify={() => onPickImage("modify")}
            onPickImageEnhance={() => onPickImage("enhance")}
            onPickStorybook={() => { setTopMode("storybook"); setView("studio"); }}
          />
        </main>
      ) : (
        <main className="flex-1 max-w-[1400px] w-full mx-auto px-4 sm:px-6 py-6 grid grid-cols-1 lg:grid-cols-[440px_minmax(0,1fr)] gap-6">
          <section className="lg:max-h-[calc(100vh-104px)] lg:overflow-y-auto lg:pr-2 lg:sticky lg:top-20">
            <ModeTabs mode={topMode} onChange={setTopMode} />
            {topMode === "image" ? (
              <ImagePanel initialMode={imageSubMode} />
            ) : (
              <StorybookPanel />
            )}
          </section>
          <section>
            <OutputPanel />
          </section>
        </main>
      )}

      <HistoryDrawer open={historyOpen} onClose={() => setHistoryOpen(false)} />

      {showAvatarPicker && (
        <AvatarPicker
          initial={profile}
          onDone={(p) => {
            setProfile(p);
            setShowAvatarPicker(false);
          }}
          onCancel={profile ? () => setShowAvatarPicker(false) : undefined}
        />
      )}
    </div>
  );
}

export default function App() {
  return (
    <StudioProvider>
      <Studio />
    </StudioProvider>
  );
}
