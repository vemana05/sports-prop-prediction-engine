import { cn } from "@/lib/utils";
import type { TierName } from "@/lib/types";

export const TIER_LABEL: Record<TierName, string> = { safe: "Safe", risky: "Risky", dangerous: "Dangerous" };

export const TIER_CLASSES: Record<TierName, { dot: string; text: string; ring: string; tint: string }> = {
  safe: { dot: "bg-safe", text: "text-safe-ink", ring: "border-safe", tint: "bg-safe/10" },
  risky: { dot: "bg-risky", text: "text-risky-ink", ring: "border-risky", tint: "bg-risky/10" },
  dangerous: { dot: "bg-danger", text: "text-danger-ink", ring: "border-danger", tint: "bg-danger/10" },
};

export function TierBadge({ tier, className }: { tier: TierName; className?: string }) {
  const c = TIER_CLASSES[tier];
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium", c.tint, c.text, className)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", c.dot)} aria-hidden />
      {TIER_LABEL[tier]}
    </span>
  );
}
