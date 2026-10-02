"use client";

import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { TIER_CLASSES, TIER_LABEL } from "./tier-badge";
import { cn } from "@/lib/utils";
import { pct } from "@/lib/format";
import type { TierConfig, TierName } from "@/lib/types";

const TIERS: TierName[] = ["safe", "risky", "dangerous"];

function hitRange(t: TierConfig) {
  if (t.target_parlay_prob_max >= 1) return `≥ ${pct(t.target_parlay_prob_min)} hit`;
  if (t.target_parlay_prob_min <= 0) return `< ${pct(t.target_parlay_prob_max)} hit`;
  return `${pct(t.target_parlay_prob_min)}–${pct(t.target_parlay_prob_max)} hit`;
}

/** Three-step control colored with the tier tokens (spec 10.4 A). */
export function TierSelector({
  value,
  onChange,
  tiers,
}: {
  value: TierName;
  onChange: (t: TierName) => void;
  tiers?: Record<TierName, TierConfig>;
}) {
  return (
    <RadioGroup
      value={value}
      onValueChange={(v) => onChange(v as TierName)}
      aria-label="Risk tier"
      className="grid-cols-3 gap-2 md:grid-cols-1"
    >
      {TIERS.map((t) => {
        const c = TIER_CLASSES[t];
        const cfg = tiers?.[t];
        const selected = value === t;
        return (
          <RadioGroupItem
            key={t}
            value={t}
            className={cn(
              "group flex min-w-0 items-start gap-2 rounded-control border bg-panel p-2 text-left transition-colors sm:gap-2.5 sm:p-2.5 md:p-3",
              selected ? cn(c.ring, c.tint) : "border-border hover:bg-panel-2",
            )}
          >
            <span
              className={cn(
                "mt-1 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border-2",
                selected ? c.ring : "border-muted/60",
              )}
              aria-hidden
            >
              {selected && <span className={cn("h-2 w-2 rounded-full", c.dot)} />}
            </span>
            <span className="min-w-0">
              <span className={cn("block truncate font-display text-base font-semibold leading-tight sm:text-lg", selected ? c.text : "text-ink")}>
                {TIER_LABEL[t]}
              </span>
              {cfg && (
                <span className="block text-xs text-muted">
                  {cfg.legs_min}–{cfg.legs_max} legs
                  <span className="hidden md:inline"> · {hitRange(cfg)}</span>
                </span>
              )}
            </span>
          </RadioGroupItem>
        );
      })}
    </RadioGroup>
  );
}
