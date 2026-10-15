"use client";

import { useEarlySeason } from "@/lib/api";

/** NBA only: shown until every team has played 10 games (backend/predict/early_season.py). */
export function EarlySeasonBanner({ className }: { className?: string }) {
  const { data } = useEarlySeason();
  if (!data?.active || data.min_games_played == null) return null;
  const left = data.games_remaining;
  const short = data.teams_below.length;
  return (
    <div role="note" className={`rounded-control border border-risky/50 bg-risky/10 px-3 py-2 text-sm ${className ?? ""}`}>
      <p>
        <span className="font-semibold text-risky-ink">Early season: probabilities are unreliable. </span>
        Fewest games played by any team: <span className="font-semibold tabular-nums">{data.min_games_played}</span> of{" "}
        {data.threshold} ({left} to go; {short} team{short !== 1 ? "s" : ""} still under {data.threshold}).
      </p>
      <ul className="mt-1 list-disc space-y-0.5 pl-5 text-xs text-muted">
        <li>The model leans on last season's stats until teams have about {data.threshold} games of this season.</li>
        <li>Star projections run low early in the season (spec 5.9).</li>
        <li>The data check is stricter until then, and any parlay it flags is staked at $0.</li>
      </ul>
    </div>
  );
}
