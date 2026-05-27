import { Moon, Sun, Sparkles, History, AlertCircle, Home } from "lucide-react";
import { useStudio } from "../lib/store";
import { Button } from "./ui";
import { cn } from "../lib/utils";
import type { UserProfile } from "../lib/user";

interface Props {
  onOpenHistory: () => void;
  comfyOk: boolean | null;
  profile: UserProfile | null;
  onEditProfile: () => void;
  onGoHome?: () => void;
}

export function Header({ onOpenHistory, comfyOk, profile, onEditProfile, onGoHome }: Props) {
  const { theme, toggleTheme, history } = useStudio();

  return (
    <header className="sticky top-0 z-30 backdrop-blur-xl bg-bg-base/70 border-b border-border">
      <div className="max-w-[1600px] mx-auto px-6 h-14 flex items-center justify-between">
        <button
          onClick={onGoHome}
          className="flex items-center gap-2.5 group"
          title="Home"
        >
          <div className="h-8 w-8 rounded-xl bg-gradient-to-br from-brand to-orange-400 flex items-center justify-center shadow-md group-hover:shadow-lg transition-shadow">
            <Sparkles className="h-4 w-4 text-white" />
          </div>
          <div className="font-semibold text-fg-base text-base">Fatimah Studio</div>
        </button>

        <div className="flex items-center gap-1">
          {comfyOk === false && (
            <div
              className="flex items-center gap-1.5 px-2.5 h-8 mr-1 rounded-full text-xs bg-danger/10 text-danger"
              title="The engine isn't running."
            >
              <AlertCircle className="h-3.5 w-3.5" />
              Engine offline
            </div>
          )}

          {profile && (
            <button
              onClick={onEditProfile}
              className="flex items-center gap-1.5 px-2.5 h-8 mr-1 rounded-full text-xs bg-bg-muted hover:bg-bg-subtle transition border border-border"
              title="Change avatar"
            >
              <span className="text-base leading-none">{profile.emoji}</span>
              <span className="font-medium text-fg-base">{profile.name}</span>
            </button>
          )}

          {onGoHome && (
            <Button variant="ghost" size="icon" onClick={onGoHome} title="Home">
              <Home className="h-4 w-4" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={onOpenHistory}
            title={`Your creations (${history.length})`}
          >
            <span className="relative">
              <History className="h-4 w-4" />
              {history.length > 0 && (
                <span
                  className={cn(
                    "absolute -top-1 -right-1 h-3 min-w-[12px] px-1 rounded-full",
                    "bg-brand text-brand-fg text-[9px] font-bold leading-3 flex items-center justify-center",
                  )}
                >
                  {history.length > 99 ? "99+" : history.length}
                </span>
              )}
            </span>
          </Button>
          <Button variant="ghost" size="icon" onClick={toggleTheme} title="Toggle theme">
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>
      </div>
    </header>
  );
}
