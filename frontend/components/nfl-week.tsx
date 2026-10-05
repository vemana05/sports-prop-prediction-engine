"use client";

import { useState } from "react";
import { ChevronDown, ChevronLeft, ChevronRight } from "lucide-react";
import { ActualValue, MissLegend } from "@/components/actual-value";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useNflWeek } from "@/lib/api";
import { clockTime, slateDate } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { NflGame, NflMarket, NflPlayer, NflTeam, NflWeek, NflWeekInfo } from "@/lib/types";

const MARKETS: { key: NflMarket; label: string; title: string }[] = [
  { key: "passing_yards", label: "Pass yds", title: "Passing yards" },
  { key: "rushing_yards", label: "Rush yds", title: "Rushing yards" },
  { key: "receiving_yards", label: "Rec yds", title: "Receiving yards" },
  { key: "receptions", label: "Rec", title: "Receptions" },
];

const toF = (c: number) => Math.round((c * 9) / 5 + 32);
const toMph = (kph: number) => Math.round(kph / 1.609);

const weekRange = (w: NflWeekInfo) =>
  w.first_kickoff_utc && w.last_kickoff_utc
    ? `${slateDate(w.first_kickoff_utc)} – ${slateDate(w.last_kickoff_utc)}`
    : "";

/** NFL view of the Schedule page: a week picker over the nflverse schedule
 * and tentative per-player projections. Projections only, no odds. */
export function NflWeekView() {
  const [week, setWeek] = useState<number | null>(null); // null = server's default (current week)
  const { data, error, isFetching, isPlaceholderData } = useNflWeek(week, true);
  const current = week ?? data?.week ?? null;

  return (
    <>
      <WeekPicker weeks={data?.weeks ?? []} current={current} onChange={setWeek} busy={isFetching} />
      {error ? (
        <p className="mt-6 text-danger-ink">Couldn't load NFL projections: {(error as Error).message}</p>
      ) : !data ? (
        <div className="mt-6 space-y-3">
          <Skeleton className="h-24 w-full" />
          {Array.from({ length: 3 }, (_, i) => <Skeleton key={i} className="h-20 w-full" />)}
        </div>
      ) : (
        <div className={cn("transition-opacity", isPlaceholderData && "opacity-60")}>
          <Notes d={data} />
          {data.games.length === 0 ? (
            <div className="mt-6 rounded-control border border-dashed border-border bg-panel px-5 py-10 text-center">
              <p className="font-display text-lg font-semibold">No games scheduled this week</p>
            </div>
          ) : (
            <ul className="mt-6 space-y-3">
              {data.games.map((g, i) => <GameCard key={g.game_id} game={g} defaultOpen={i === data.games.findIndex((x) => !x.final)} />)}
            </ul>
          )}
        </div>
      )}
    </>
  );
}

function WeekPicker({ weeks, current, onChange, busy }: { weeks: NflWeekInfo[]; current: number | null; onChange: (w: number) => void; busy: boolean }) {
  const idx = weeks.findIndex((w) => w.week === current);
  const info = idx >= 0 ? weeks[idx] : null;
  return (
    <div className="mt-4 flex flex-wrap items-center gap-2">
      <Button variant="outline" size="icon" aria-label="Previous week" disabled={idx <= 0} onClick={() => onChange(weeks[idx - 1].week)}>
        <ChevronLeft className="h-4 w-4" />
      </Button>
      <select
        aria-label="NFL week"
        value={current ?? ""}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-9 rounded-control border border-border bg-panel px-3 text-base"
      >
        {current == null && <option value="">Loading weeks…</option>}
        {weeks.map((w) => (
          <option key={w.week} value={w.week}>
            {w.label} · {weekRange(w)}
          </option>
        ))}
      </select>
      <Button variant="outline" size="icon" aria-label="Next week" disabled={idx < 0 || idx >= weeks.length - 1} onClick={() => onChange(weeks[idx + 1].week)}>
        <ChevronRight className="h-4 w-4" />
      </Button>
      {info && (
        <span className="ml-1 text-sm text-muted">
          <span className="num">{info.n_games}</span> games · <span className="num">{info.n_final}</span> final
        </span>
      )}
      {busy && <span className="sr-only" role="status">Loading</span>}
    </div>
  );
}

function TentativeChip() {
  return <span className="mr-1.5 inline-block rounded-full bg-risky/15 px-2 py-0.5 text-xs font-medium text-risky-ink">Tentative</span>;
}

function Notes({ d }: { d: NflWeek }) {
  return (
    <section aria-label="About these projections" className="mt-4 rounded-control border border-border bg-panel p-4 text-sm">
      <p>
        <TentativeChip /> Computed {slateDate(d.computed_at)}, {clockTime(d.computed_at)} ET. Recomputed daily.{" "}
        <span className="font-medium">Projections only:</span> NFL odds aren't collected, so there are no lines or parlays.
      </p>
      <ul className="mt-2 list-disc space-y-1 pl-5 text-muted">
        <li>Ranges are the middle 50% of outcomes (25th–75th percentile). Half of real games land outside them.</li>
        <li>
          Models were trained on {d.models_trained_through} and earlier seasons only, so every {d.season} number here is out of sample.
          Final games show the pre-kickoff projection next to what actually happened.
        </li>
        <li>
          {d.injuries_posted ? (
            <>This week's injury report is in: players listed Out are left off, and Questionable players are marked Q.</>
          ) : (
            <>
              <span className="text-ink">Injury report not posted yet.</span> Projections assume everyone on the latest depth chart plays.
            </>
          )}
        </li>
      </ul>
      {d.games.some((g) => g.final) && <MissLegend className="mt-3 border-t border-border pt-3" />}
    </section>
  );
}

function GameCard({ game, defaultOpen }: { game: NflGame; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const panelId = `proj-${game.game_id}`;
  const w = game.weather;
  return (
    <li className="rounded-control border border-border bg-panel">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 p-4">
        <div className="w-28 shrink-0">
          {game.kickoff_utc && (
            <>
              <p className="text-xs text-muted">{slateDate(game.kickoff_utc)}</p>
              <p className="num text-lg font-semibold">
                {clockTime(game.kickoff_utc)} <span className="text-xs font-normal text-muted">ET</span>
              </p>
            </>
          )}
        </div>
        <div className="min-w-0 flex-1 space-y-1">
          <TeamLine team={game.away} where="Away" score={game.away_score} rest={game.rest.away} />
          <TeamLine team={game.home} where="Home" score={game.home_score} rest={game.rest.home} />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {game.final ? (
            <span className="rounded-full bg-panel-2 px-2 py-0.5 text-xs font-medium">Final</span>
          ) : (
            <span className="text-sm text-muted">Odds not enabled</span>
          )}
          <span className="rounded-full bg-panel-2 px-2 py-0.5 text-xs text-muted" title={game.stadium ?? undefined}>
            {game.indoor ? "Indoors" : w ? `${toF(w.temperature_c)}°F · wind ${toMph(w.wind_kph)} mph${w.precip_mm > 0 ? ` · ${w.precip_mm} mm rain` : ""}` : "Outdoors · forecast not out yet"}
          </span>
          <Button variant="ghost" size="sm" aria-expanded={open} aria-controls={panelId} onClick={() => setOpen(!open)}>
            {open ? "Hide" : "Show"} projections
            <ChevronDown className={cn("h-4 w-4 transition-transform", open && "rotate-180")} aria-hidden />
          </Button>
        </div>
      </div>
      {open && (
        <div id={panelId} className="grid gap-4 border-t border-border p-4 xl:grid-cols-2">
          <TeamTable t={game.teams.away} final={game.final} />
          <TeamTable t={game.teams.home} final={game.final} />
        </div>
      )}
    </li>
  );
}

function TeamLine({ team, where, score, rest }: { team: string; where: string; score: number | null; rest: number | null }) {
  return (
    <p className="flex flex-wrap items-center gap-2">
      <span className="font-display w-12 text-lg font-semibold">{team}</span>
      <span className="w-10 text-xs text-muted">{where}</span>
      {score != null && <span className="num w-8 text-lg font-semibold">{score}</span>}
      {rest != null && rest !== 7 && (
        <span className={cn("rounded-full px-2 py-0.5 text-xs", rest < 7 ? "bg-risky/15 font-medium text-risky-ink" : "bg-panel-2 text-muted")}>
          {rest} days rest
        </span>
      )}
    </p>
  );
}

function TeamTable({ t, final }: { t: NflTeam; final: boolean }) {
  return (
    <section aria-label={`${t.team} projections`} className="min-w-0">
      <div className="mb-2 flex flex-wrap items-baseline gap-x-2">
        <h3 className="font-display text-lg font-semibold">{t.team}</h3>
        <TentativeChip />
        <span className="text-xs text-muted">vs {t.opponent}</span>
      </div>
      {t.players.length === 0 ? (
        <p className="text-sm text-muted">No depth-chart players found for {t.team}.</p>
      ) : (
        <div className="overflow-x-auto rounded-control border border-border">
          <table className="w-full min-w-[480px] text-left text-sm">
            <thead className="bg-panel-2 text-xs text-muted">
              <tr>
                <th scope="col" className="px-3 py-2 font-medium">Player</th>
                {MARKETS.map((m) => <th key={m.key} scope="col" title={m.title} className="px-2 py-2 text-right font-medium">{m.label}</th>)}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {t.players.map((p) => <PlayerRow key={p.player_id} p={p} final={final} />)}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function PlayerRow({ p, final }: { p: NflPlayer; final: boolean }) {
  return (
    <tr>
      <th scope="row" className="max-w-[200px] px-3 py-1.5 font-medium">
        <span className="block truncate">{p.player_name}</span>
        <span className="text-xs font-normal text-muted">
          {p.position}{p.depth_rank ?? ""}
          {p.questionable && <span className="ml-1.5 rounded bg-risky/15 px-1 font-medium text-risky-ink" title="Questionable on the injury report">Q</span>}
        </span>
      </th>
      {MARKETS.map((m) => {
        const v = p.projections[m.key];
        if (!v) return <td key={m.key} className="px-2 py-1.5 text-right text-muted">—</td>;
        const digits = m.key === "receptions" ? 1 : 0;
        return (
          <td key={m.key} className="px-2 py-1.5 text-right">
            <span className="num block text-base font-semibold leading-tight">{v.mean.toFixed(digits)}</span>
            <span className="num block text-xs text-muted">{v.lo.toFixed(0)}–{v.hi.toFixed(0)}</span>
            {final && v.actual != null && (
              <ActualValue actual={v.actual} lo={v.lo} hi={v.hi} digits={0} prefix="actual " className="block text-xs" />
            )}
          </td>
        );
      })}
    </tr>
  );
}
