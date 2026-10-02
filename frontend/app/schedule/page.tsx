"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ChevronDown, ChevronLeft, ChevronRight } from "lucide-react";
import { LeagueToggle } from "@/components/league-toggle";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useSchedule } from "@/lib/api";
import { clockTime } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { League, ProjectedPlayer, ScheduleDay, ScheduledGame, StatKey, TeamProjection, TeamRest } from "@/lib/types";

const STATS: { key: StatKey; label: string }[] = [
  { key: "minutes", label: "Min" },
  { key: "points", label: "Pts" },
  { key: "rebounds", label: "Reb" },
  { key: "assists", label: "Ast" },
  { key: "fg3m", label: "3PM" },
];

/** Shift a YYYY-MM-DD date by whole days without timezone drift. */
const shiftDay = (iso: string, days: number) => {
  const d = new Date(`${iso}T12:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
};

const longDate = (iso: string) =>
  new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric", timeZone: "UTC" });

const shortDate = (iso: string) =>
  new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" });

const ordinal = (n: number) => `${n}${n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th"}`;

export default function SchedulePage() {
  const [league, setLeague] = useState<League>("nba");
  const [date, setDate] = useState<string | null>(null); // null = server's default (next game day)
  const { data, error, isFetching, isPlaceholderData } = useSchedule(league, league === "nba" ? date : null);

  useEffect(() => {
    const fromDom = document.documentElement.dataset.league;
    if (fromDom === "nfl") setLeague("nfl");
  }, []);
  useEffect(() => {
    document.documentElement.dataset.league = league;
  }, [league]);

  const day = data?.status === "ready" ? data : null;
  const current = date ?? day?.date ?? null;

  return (
    <div className="mx-auto max-w-[1200px] px-4 pb-16 pt-6 lg:px-6">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-border pb-4">
        <LeagueToggle value={league} onChange={setLeague} />
        <h1 className="font-display text-xl font-bold">Schedule &amp; projections</h1>
      </div>

      {league === "nfl" ? (
        <div className="mt-6 rounded-control border border-dashed border-border bg-panel px-5 py-10 text-center">
          <p className="font-display text-lg font-semibold">NFL arrives in Phase 6</p>
          <p className="mx-auto mt-1 max-w-prose text-sm text-muted">
            The NFL version will use a week picker built on nflverse schedules instead of a date picker.
          </p>
          <Button className="mt-4" onClick={() => setLeague("nba")}>Switch to NBA</Button>
        </div>
      ) : (
        <>
          <DateNav current={current} day={day} onChange={setDate} busy={isFetching} />
          {error ? (
            <p className="mt-6 text-danger-ink">Couldn't load the schedule: {(error as Error).message}</p>
          ) : !day ? (
            <div className="mt-6 space-y-3">
              <Skeleton className="h-24 w-full" />
              {Array.from({ length: 3 }, (_, i) => <Skeleton key={i} className="h-20 w-full" />)}
            </div>
          ) : (
            <div className={cn("transition-opacity", isPlaceholderData && "opacity-60")}>
              <Notes day={day} />
              {day.games.length === 0 ? (
                <div className="mt-6 rounded-control border border-dashed border-border bg-panel px-5 py-10 text-center">
                  <p className="font-display text-lg font-semibold">No NBA games on {shortDate(day.date)}</p>
                  {day.next_game_day && (
                    <Button className="mt-4" onClick={() => setDate(day.next_game_day)}>
                      Next game day: {shortDate(day.next_game_day)}
                    </Button>
                  )}
                </div>
              ) : (
                <ul className="mt-6 space-y-3">
                  {day.games.map((g) => <GameCard key={g.game_id} game={g} />)}
                </ul>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function DateNav({ current, day, onChange, busy }: { current: string | null; day: ScheduleDay | null; onChange: (d: string) => void; busy: boolean }) {
  return (
    <div className="mt-4 flex flex-wrap items-center gap-2">
      <Button variant="outline" size="icon" aria-label="Previous day" disabled={!current} onClick={() => current && onChange(shiftDay(current, -1))}>
        <ChevronLeft className="h-4 w-4" />
      </Button>
      <input
        type="date"
        aria-label="Game date"
        value={current ?? ""}
        onChange={(e) => e.target.value && onChange(e.target.value)}
        className="num h-9 rounded-control border border-border bg-panel px-3 text-base"
      />
      <Button variant="outline" size="icon" aria-label="Next day" disabled={!current} onClick={() => current && onChange(shiftDay(current, 1))}>
        <ChevronRight className="h-4 w-4" />
      </Button>
      {current && <span className="ml-1 font-medium">{longDate(current)}</span>}
      <div className="ml-auto flex flex-wrap gap-2">
        {day?.prev_game_day && (
          <Button variant="ghost" size="sm" onClick={() => onChange(day.prev_game_day!)}>← Previous game day</Button>
        )}
        {day?.next_game_day && (
          <Button variant="ghost" size="sm" onClick={() => onChange(day.next_game_day!)}>Next game day →</Button>
        )}
      </div>
      {busy && <span className="sr-only" role="status">Loading</span>}
    </div>
  );
}

function Notes({ day }: { day: ScheduleDay }) {
  const computed = new Date(day.computed_at);
  return (
    <section aria-label="About these projections" className="mt-4 rounded-control border border-border bg-panel p-4 text-sm">
      <p>
        <TentativeChip /> Computed {computed.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "America/New_York" })},{" "}
        {clockTime(day.computed_at)} ET from games played before {shortDate(day.form_as_of)}. Recomputed daily.
      </p>
      <ul className="mt-2 list-disc space-y-1 pl-5 text-muted">
        <li>
          Ranges are the middle 50% of outcomes (25th–75th percentile). Half of real games land outside them.
        </li>
        <li>
          <span className="text-ink">Teammates out: unknown until injury reports post</span> (the day before the game). Projections
          assume nobody is out.
        </li>
        <li>
          <span className="text-ink">Early-season star projections run low.</span> Last-10 averages are dragged down by injury-return and
          rest games at the end of last season, so opening-week numbers for stars are likely too low (spec 5.9).
        </li>
      </ul>
    </section>
  );
}

function TentativeChip() {
  return <span className="mr-1.5 inline-block rounded-full bg-risky/15 px-2 py-0.5 text-xs font-medium text-risky-ink">Tentative</span>;
}

function restChip(r: TeamRest) {
  if (r.season_opener) return { text: "Season opener", tone: "muted" as const };
  if (r.back_to_back) return { text: "Back-to-back", tone: "warn" as const };
  if (r.rest_days == null) return null;
  return { text: `${r.rest_days - 1} day${r.rest_days - 1 === 1 ? "" : "s"} rest`, tone: "muted" as const };
}

function RestChip({ rest }: { rest: TeamRest }) {
  const c = restChip(rest);
  if (!c) return null;
  return (
    <span className={cn("rounded-full px-2 py-0.5 text-xs", c.tone === "warn" ? "bg-risky/15 font-medium text-risky-ink" : "bg-panel-2 text-muted")}>
      {c.text}
    </span>
  );
}

function GameCard({ game }: { game: ScheduledGame }) {
  const [open, setOpen] = useState(false);
  const panelId = `proj-${game.game_id}`;
  return (
    <li className="rounded-control border border-border bg-panel">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 p-4">
        <p className="num w-20 shrink-0 text-lg font-semibold">{clockTime(game.tip_utc)} <span className="text-xs font-normal text-muted">ET</span></p>
        <div className="min-w-0 flex-1 space-y-1">
          <TeamLine team={game.away} where="Away" rest={game.rest.away} />
          <TeamLine team={game.home} where="Home" rest={game.rest.home} />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {game.lines_posted ? (
            <Button asChild variant="primary" size="sm">
              <Link href={game.builder_game_id ? `/?game=${encodeURIComponent(game.builder_game_id)}` : "/"}>Lines posted — build parlays</Link>
            </Button>
          ) : (
            <span className="text-sm text-muted">Lines not posted yet</span>
          )}
          <Button variant="ghost" size="sm" aria-expanded={open} aria-controls={panelId} onClick={() => setOpen(!open)}>
            {open ? "Hide" : "Show"} projections
            <ChevronDown className={cn("h-4 w-4 transition-transform", open && "rotate-180")} aria-hidden />
          </Button>
        </div>
      </div>
      {open && (
        <div id={panelId} className="grid gap-4 border-t border-border p-4 xl:grid-cols-2">
          <TeamTable t={game.teams.away} rest={game.rest.away} />
          <TeamTable t={game.teams.home} rest={game.rest.home} />
        </div>
      )}
    </li>
  );
}

function TeamLine({ team, where, rest }: { team: string; where: string; rest: TeamRest }) {
  return (
    <p className="flex flex-wrap items-center gap-2">
      <span className="font-display w-12 text-lg font-semibold">{team}</span>
      <span className="w-10 text-xs text-muted">{where}</span>
      <RestChip rest={rest} />
    </p>
  );
}

function TeamTable({ t, rest }: { t: TeamProjection; rest: TeamRest }) {
  const context = [
    `vs ${t.opponent}`,
    t.opp_pace != null && t.opp_pace_rank != null ? `opp. pace ${t.opp_pace} (${ordinal(t.opp_pace_rank)} fastest of ${t.n_teams_ranked})` : null,
    restChip(rest)?.text.toLowerCase(),
    "teammates out: unknown",
  ].filter(Boolean);
  return (
    <section aria-label={`${t.team} projections`} className="min-w-0">
      <div className="mb-2 flex flex-wrap items-baseline gap-x-2">
        <h3 className="font-display text-lg font-semibold">{t.team}</h3>
        <TentativeChip />
        <span className="text-xs text-muted">{context.join(" · ")}</span>
      </div>
      {t.players.length === 0 ? (
        <p className="text-sm text-muted">No recent players found for {t.team}.</p>
      ) : (
        <div className="overflow-x-auto rounded-control border border-border">
          <table className="w-full min-w-[520px] text-left text-sm">
            <thead className="bg-panel-2 text-xs text-muted">
              <tr>
                <th scope="col" className="px-3 py-2 font-medium">Player</th>
                {STATS.map((s) => <th key={s.key} scope="col" className="px-2 py-2 text-right font-medium">{s.label}</th>)}
                <th scope="col" className="px-3 py-2 text-right font-medium" title="Points the opponent allows to this player's position, last 10 games">
                  Opp vs pos
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {t.players.map((p) => <PlayerRow key={p.player_id} p={p} />)}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function PlayerRow({ p }: { p: ProjectedPlayer }) {
  return (
    <tr>
      <th scope="row" className="max-w-[180px] truncate px-3 py-1.5 font-medium">{p.player_name}</th>
      {STATS.map((s) => {
        const v = p.projections[s.key];
        return (
          <td key={s.key} className="px-2 py-1.5 text-right">
            <span className="num block text-base font-semibold leading-tight">{v.mean.toFixed(1)}</span>
            <span className="num block text-xs text-muted">{v.lo.toFixed(0)}–{v.hi.toFixed(0)}</span>
          </td>
        );
      })}
      <td className="num px-3 py-1.5 text-right text-muted">{p.opp_def_vs_position_pts?.toFixed(1) ?? "—"}</td>
    </tr>
  );
}
