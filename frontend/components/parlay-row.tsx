import { cn } from "@/lib/utils";
import { american, evPerDollar, money, pct } from "@/lib/format";
import type { Parlay } from "@/lib/types";

/** Suggested parlay row (spec 10.4 A). Click loads it into the bet slip. */
export function ParlayRow({ parlay, selected, onSelect }: { parlay: Parlay; selected: boolean; onSelect: () => void }) {
  const players = parlay.legs.map((l) => l.player_name.split(" ").slice(-1)[0]).join(", ");
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={cn(
        "w-full rounded-control border bg-panel p-3 text-left transition-colors sm:p-4",
        selected ? "border-ink ring-1 ring-ink" : "border-border hover:border-muted",
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="font-display text-lg font-semibold leading-tight">
            {parlay.name} <span className="font-sans text-sm font-normal text-muted">· {parlay.legs.length} legs</span>
          </p>
          <p className="truncate text-sm text-muted">{players}</p>
        </div>
        <div className="shrink-0 text-right">
          <p className="num text-xl font-bold leading-none">{pct(parlay.p_hit)}</p>
          <p className="text-xs text-muted">model hit</p>
        </div>
      </div>
      {parlay.american_odds == null ? (
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
          <span className="text-muted">
            Legs {parlay.legs.map((l) => pct(l.model_p)).join(" · ")}
          </span>
          {parlay.fair_american_odds != null && (
            <span>
              <span className="text-muted">Fair </span>
              <span className="num font-semibold">{american(parlay.fair_american_odds)}</span>
            </span>
          )}
          {!!parlay.notes?.length && (
            <span className="rounded-full border border-border px-2 py-0.5 text-xs text-muted">Game script</span>
          )}
          <span className="ml-auto text-xs text-muted">Hit chance only · no odds</span>
        </div>
      ) : (
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        <span>
          <span className="text-muted">Pays </span>
          <span className="num font-semibold">{american(parlay.american_odds)}</span>
        </span>
        <span>
          <span className="text-muted">EV </span>
          {parlay.flagged ? (
            <span className="font-medium text-muted" title="This parlay tripped the data check, so its EV can't be trusted.">
              unreliable
            </span>
          ) : (
            <span className={cn("num font-semibold", (parlay.ev_per_dollar ?? 0) > 0 && "text-safe-ink")}>
              {evPerDollar(parlay.ev_per_dollar ?? 0)}
            </span>
          )}
        </span>
        <span className="ml-auto">
          {parlay.flagged ? (
            <span className="rounded-full border border-risky px-2 py-0.5 text-xs font-medium text-risky-ink">Check data</span>
          ) : (
            <>
              <span className="text-muted">Stake </span>
              <span className="num font-semibold">{money(parlay.stake)}</span>
            </>
          )}
        </span>
      </div>
      )}
    </button>
  );
}

export function ParlayRowSkeleton() {
  return (
    <div className="rounded-control border border-border bg-panel p-4">
      <div className="flex justify-between">
        <div className="space-y-2">
          <div className="h-5 w-32 animate-pulse rounded bg-panel-2" />
          <div className="h-4 w-48 animate-pulse rounded bg-panel-2" />
        </div>
        <div className="h-7 w-12 animate-pulse rounded bg-panel-2" />
      </div>
      <div className="mt-4 h-4 w-3/4 animate-pulse rounded bg-panel-2" />
    </div>
  );
}
