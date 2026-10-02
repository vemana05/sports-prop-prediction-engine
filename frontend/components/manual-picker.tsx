"use client";

import { Check, Plus } from "lucide-react";
import { Headshot } from "./headshot";
import { Skeleton } from "@/components/ui/skeleton";
import { american, pct } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Leg, TierConfig } from "@/lib/types";

/**
 * "Build my own": every leg in the filtered pool, best model probability
 * first. Picked legs are scored by the same engine as suggestions. A leg
 * on a player prop already in the slip is disabled (no duplicates or
 * over/under contradictions).
 */
export function ManualPicker({
  legs,
  loading,
  picked,
  onToggle,
  tierCfg,
  tierLabel,
}: {
  legs: Leg[];
  loading: boolean;
  picked: string[];
  onToggle: (leg: Leg) => void;
  tierCfg?: TierConfig;
  tierLabel: string;
}) {
  if (loading)
    return (
      <div className="space-y-2">
        {Array.from({ length: 5 }, (_, i) => <Skeleton key={i} className="h-16 w-full" />)}
      </div>
    );
  if (legs.length === 0)
    return <p className="rounded-control border border-dashed border-border bg-panel px-5 py-8 text-center text-sm text-muted">No legs match your filters.</p>;

  const pickedPlayers = new Set(legs.filter((l) => picked.includes(l.id)).map((l) => `${l.player_id}-${l.market}`));
  return (
    <ul className="divide-y divide-border overflow-hidden rounded-control border border-border bg-panel">
      {legs.map((leg) => {
        const on = picked.includes(leg.id);
        const blocked = !on && pickedPlayers.has(`${leg.player_id}-${leg.market}`);
        const belowTier = tierCfg && leg.shrunk_p < tierCfg.min_leg_prob;
        return (
          <li key={leg.id}>
            <button
              type="button"
              onClick={() => onToggle(leg)}
              disabled={blocked}
              aria-pressed={on}
              className={cn(
                "flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors sm:px-4",
                on ? "bg-accent/10" : "hover:bg-panel-2",
                blocked && "cursor-not-allowed opacity-45",
              )}
            >
              <Headshot src={leg.headshot_url} name={leg.player_name} className="h-9 w-9" />
              <span className="min-w-0 flex-1">
                <span className="block truncate font-medium leading-tight">{leg.player_name}</span>
                <span className="block text-sm">
                  <span className="font-semibold text-accent-ink">{leg.label}</span>
                  <span className="text-muted"> · {american(leg.american_odds)}</span>
                  {blocked && <span className="text-xs text-muted"> · player already in slip</span>}
                  {!blocked && belowTier && <span className="text-xs text-muted"> · below {tierLabel}'s {pct(tierCfg!.min_leg_prob)}</span>}
                </span>
              </span>
              <span className="shrink-0 text-right">
                <span className="num block text-lg font-semibold leading-none">{pct(leg.model_p)}</span>
                <span className="block text-xs text-muted">book {pct(leg.book_p)}</span>
              </span>
              <span
                className={cn(
                  "flex h-7 w-7 shrink-0 items-center justify-center rounded-full border",
                  on ? "border-ink bg-ink text-panel" : "border-border text-muted",
                )}
                aria-hidden
              >
                {on ? <Check className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
