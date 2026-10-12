import { Headshot } from "./headshot";
import { ProbabilityBar } from "./probability-bar";
import { american, clockTime } from "@/lib/format";
import type { League, Leg } from "@/lib/types";

function matchup(leg: Leg) {
  if (!leg.team) return null;
  if (!leg.opponent) return leg.team;
  return leg.is_home ? `${leg.team} vs ${leg.opponent}` : `${leg.team} @ ${leg.opponent}`;
}

/** Spec 10.4 B. Used inside the bet slip and in leg lists. */
export function LegRow({ leg, league }: { leg: Leg; league: League }) {
  const context = [leg.position, matchup(leg), leg.tip_utc && `${clockTime(leg.tip_utc)} ET`].filter(Boolean).join(" · ");
  return (
    <div className="flex gap-3">
      <Headshot src={leg.headshot_url} name={leg.player_name} className="h-10 w-10" />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <p className="truncate font-medium leading-tight">{leg.player_name}</p>
          {leg.american_odds != null ? (
            <span className="num shrink-0 text-base font-semibold">{american(leg.american_odds)}</span>
          ) : (
            <span className="shrink-0 text-xs text-muted">no odds</span>
          )}
        </div>
        <div className="flex items-baseline justify-between gap-2 text-sm">
          <span className="font-semibold text-accent-ink">{leg.label}</span>
          {context && <span className="truncate text-xs text-muted">{context}</span>}
        </div>
        {leg.injury && (
          <span className="mt-1 inline-block rounded-full bg-risky/15 px-2 py-0.5 text-xs text-risky-ink">{leg.injury}</span>
        )}
        <ProbabilityBar model={leg.model_p} book={leg.book_p} league={league} className="mt-2" />
        <p className="mt-1 text-xs text-muted">{leg.reason}</p>
      </div>
    </div>
  );
}
