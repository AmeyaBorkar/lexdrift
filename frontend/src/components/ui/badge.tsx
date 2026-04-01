import { cn } from "@/lib/utils";
import type { HTMLAttributes } from "react";

type Severity = "critical" | "high" | "medium" | "low";

const severityStyles: Record<Severity, string> = {
  critical:
    "bg-danger/10 text-danger border-danger/20",
  high:
    "bg-warning/10 text-warning border-warning/20",
  medium:
    "bg-warning/10 text-warning/80 border-warning/15",
  low:
    "bg-muted text-muted-foreground border-border",
};

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  severity?: Severity;
  variant?: "default" | "outline";
}

export function Badge({
  severity = "low",
  variant = "default",
  className,
  ...props
}: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded px-2 py-0.5 text-xs font-medium border transition-colors duration-200",
        variant === "outline"
          ? "bg-transparent border-border text-muted-foreground"
          : severityStyles[severity],
        className
      )}
      {...props}
    />
  );
}
