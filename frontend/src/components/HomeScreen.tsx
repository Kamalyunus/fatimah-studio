import { Image as ImageIcon, Maximize, Pencil, BookOpen } from "lucide-react";
import { useStudio } from "../lib/store";
import type { UserProfile } from "../lib/user";
import { cn } from "../lib/utils";

interface Props {
  profile: UserProfile;
  onPickImageCreate: () => void;
  onPickImageModify: () => void;
  onPickImageEnhance: () => void;
  onPickStorybook: () => void;
}

export function HomeScreen({
  profile, onPickImageCreate, onPickImageModify, onPickImageEnhance, onPickStorybook,
}: Props) {
  const { history } = useStudio();

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-16 animate-fade-in">
      <div className="text-center mb-12">
        <div className="text-6xl mb-3 animate-bounce-in">{profile.emoji}</div>
        <h1 className="text-3xl sm:text-4xl font-bold text-fg-base mb-2">
          Hi {profile.name}! 👋
        </h1>
        <p className="text-lg text-fg-muted">What would you like to make today?</p>
      </div>

      {/* Featured: Storybook */}
      <button
        onClick={onPickStorybook}
        className={cn(
          "group relative w-full overflow-hidden rounded-3xl p-6 text-left mb-4",
          "bg-gradient-to-br from-brand/15 via-orange-200/10 to-amber-200/15",
          "border border-brand/30 hover:border-brand",
          "shadow-sm hover:shadow-xl transition-all",
          "hover:-translate-y-1 active:translate-y-0",
        )}
      >
        <div className="flex items-center gap-5">
          <div className="text-6xl group-hover:scale-110 transition-transform">📚</div>
          <div className="flex-1">
            <div className="text-xs font-bold uppercase tracking-wider text-brand mb-1">
              ✨ Featured
            </div>
            <div className="font-bold text-xl text-fg-base mb-1">
              Make a storybook
            </div>
            <div className="text-sm text-fg-muted">
              Tell us a story. We'll illustrate and animate it into a movie just for you.
            </div>
          </div>
          <BookOpen className="h-5 w-5 text-brand shrink-0 hidden sm:block" />
        </div>
      </button>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-5xl mx-auto">
        <BigCard
          icon={ImageIcon}
          emoji="🖼️"
          title="Make a picture"
          subtitle="Turn words into a beautiful image"
          accent="from-violet-400 to-fuchsia-400"
          onClick={onPickImageCreate}
        />
        <BigCard
          icon={Pencil}
          emoji="🎨"
          title="Change a photo"
          subtitle="Upload a photo, describe your edit"
          accent="from-sky-400 to-cyan-400"
          onClick={onPickImageModify}
        />
        <BigCard
          icon={Maximize}
          emoji="🔍"
          title="Enhance a photo"
          subtitle="Make small photos bigger and sharper"
          accent="from-emerald-400 to-teal-400"
          onClick={onPickImageEnhance}
        />
      </div>

      {history.length > 0 && (
        <div className="text-center mt-12 text-sm text-fg-muted">
          You've made <span className="font-bold text-fg-base">{history.length}</span>{" "}
          {history.length === 1 ? "creation" : "creations"} so far ✨
        </div>
      )}
    </div>
  );
}

function BigCard({
  icon: Icon, emoji, title, subtitle, accent, onClick,
}: {
  icon: typeof ImageIcon;
  emoji: string;
  title: string;
  subtitle: string;
  accent: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "group relative overflow-hidden rounded-3xl p-6 text-left",
        "bg-bg-inset border border-border hover:border-border-strong",
        "shadow-sm hover:shadow-xl transition-all",
        "hover:-translate-y-1 active:translate-y-0",
        "min-h-[180px]",
      )}
    >
      <div
        className={cn(
          "absolute inset-x-0 top-0 h-1 bg-gradient-to-r opacity-80",
          accent,
        )}
      />
      <div className="text-4xl mb-3 transition-transform group-hover:scale-110 inline-block">
        {emoji}
      </div>
      <div className="font-bold text-lg text-fg-base mb-1">{title}</div>
      <div className="text-sm text-fg-muted leading-relaxed">{subtitle}</div>
      <div className={cn(
        "absolute -bottom-8 -right-8 w-32 h-32 rounded-full bg-gradient-to-br opacity-10 transition-opacity group-hover:opacity-20",
        accent,
      )} />
      <Icon className="absolute bottom-4 right-4 h-5 w-5 text-fg-subtle group-hover:text-brand transition-colors" />
    </button>
  );
}
