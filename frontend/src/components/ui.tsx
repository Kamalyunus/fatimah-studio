import { forwardRef, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type TextareaHTMLAttributes } from "react";
import { cn } from "../lib/utils";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "subtle";
type ButtonSize = "sm" | "md" | "lg" | "icon";

export const Button = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: ButtonVariant;
    size?: ButtonSize;
  }
>(function Button({ className, variant = "secondary", size = "md", ...props }, ref) {
  return (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center gap-2 font-medium rounded-lg transition-all duration-150",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-brand/60 focus-visible:ring-offset-2 focus-visible:ring-offset-bg-base",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        {
          primary: "bg-brand text-brand-fg hover:opacity-90 active:scale-[0.98] shadow-sm",
          secondary: "bg-bg-muted text-fg-base hover:bg-bg-subtle border border-border",
          ghost: "text-fg-muted hover:text-fg-base hover:bg-bg-muted",
          danger: "bg-danger text-white hover:opacity-90",
          subtle: "bg-brand-subtle text-brand hover:opacity-80",
        }[variant],
        {
          sm: "h-8 px-3 text-xs",
          md: "h-10 px-4 text-sm",
          lg: "h-12 px-6 text-sm",
          icon: "h-9 w-9",
        }[size],
        className
      )}
      {...props}
    />
  );
});

export function Card({
  className,
  children,
  ...props
}: { className?: string; children: ReactNode } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("panel p-4", className)} {...props}>
      {children}
    </div>
  );
}

export function Section({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <div className="label">{title}</div>
        {hint && <span className="text-xs text-fg-subtle">{hint}</span>}
      </div>
      {children}
    </div>
  );
}

export const Textarea = forwardRef<
  HTMLTextAreaElement,
  TextareaHTMLAttributes<HTMLTextAreaElement>
>(function Textarea({ className, ...props }, ref) {
  return (
    <textarea
      ref={ref}
      className={cn(
        "input min-h-24 leading-relaxed resize-y",
        className
      )}
      {...props}
    />
  );
});

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  function Input({ className, ...props }, ref) {
    return <input ref={ref} className={cn("input", className)} {...props} />;
  }
);

interface SliderProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  hint?: string;
  format?: (n: number) => string;
}

export function Slider({
  label, value, onChange, min, max, step = 1, hint, format,
}: SliderProps) {
  const displayed = format ? format(value) : value.toString();
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between">
        <label className="text-sm text-fg-muted">{label}</label>
        <span className="text-sm font-mono tabular-nums text-fg-base">{displayed}</span>
      </div>
      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="w-full appearance-none bg-transparent cursor-pointer
            [&::-webkit-slider-runnable-track]:h-1.5 [&::-webkit-slider-runnable-track]:rounded-full [&::-webkit-slider-runnable-track]:bg-bg-muted
            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-brand [&::-webkit-slider-thumb]:mt-[-5px] [&::-webkit-slider-thumb]:shadow-md
            [&::-webkit-slider-thumb]:transition-transform hover:[&::-webkit-slider-thumb]:scale-110
            [&::-moz-range-track]:h-1.5 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-bg-muted
            [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-brand [&::-moz-range-thumb]:border-0
          "
          style={{
            background: `linear-gradient(to right, rgb(var(--brand)) 0%, rgb(var(--brand)) ${pct}%, transparent ${pct}%, transparent 100%)`,
            backgroundSize: "calc(100% - 4px) 6px",
            backgroundPosition: "2px center",
            backgroundRepeat: "no-repeat",
          }}
        />
      </div>
      {hint && <div className="text-xs text-fg-subtle">{hint}</div>}
    </div>
  );
}

export function ChoiceRow({
  options,
  value,
  onChange,
  size = "sm",
}: {
  options: Array<{ label: string; value: string; hint?: string }>;
  value: string;
  onChange: (v: string) => void;
  size?: "sm" | "md";
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={cn(
            "rounded-lg border transition-all",
            size === "sm" ? "px-3 py-1.5 text-xs" : "px-4 py-2 text-sm",
            value === opt.value
              ? "bg-brand text-brand-fg border-brand shadow-sm"
              : "bg-bg-subtle text-fg-muted border-border hover:border-border-strong hover:text-fg-base"
          )}
          title={opt.hint}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
}) {
  return (
    <label className="flex items-start gap-3 cursor-pointer group">
      <div
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn(
          "shrink-0 h-5 w-9 rounded-full p-0.5 transition-colors duration-200 mt-0.5",
          checked ? "bg-brand" : "bg-bg-muted"
        )}
      >
        <div
          className={cn(
            "h-4 w-4 rounded-full bg-white shadow-md transition-transform duration-200",
            checked ? "translate-x-4" : "translate-x-0"
          )}
        />
      </div>
      <div className="flex-1">
        <div className="text-sm font-medium text-fg-base group-hover:text-fg-base">{label}</div>
        {hint && <div className="text-xs text-fg-subtle mt-0.5">{hint}</div>}
      </div>
    </label>
  );
}
