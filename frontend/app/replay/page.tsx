"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { ActualValue, MissLegend } from "@/components/actual-value";
import { LeagueToggle } from "@/components/league-toggle";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useReplay } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { League, Replay, ReplayRow, ReplaySummary } from "@/lib/types";

type Ready = Extract<Replay, { status: "ready" }>;

const WEEK_CHOICES = [4, 8, 12, 26];
const fmt = (x: number | null | undefined, d = 1) => (x == null ? "—" : x.toFixed(d));
const signed = (x: number, d = 1) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(d)}`;
const pct = (x: number) => `${Math.round(x * 100)}%`;

export default function ReplayPage() {
  const [league, setLeague] = useState<League>("nba");
  const [market, setMarket] = useState<string | null>(null);
  const [period, setPeriod] = useState<string | null>(null); // null = most recent
  const [weeks, setWeeks] = useState(8);
  const { data, error, isFetching, isPlaceholderData } = useReplay(league, market, period, weeks);

  useEffect(() => {
    if (document.documentElement.dataset.league === "nfl") setLeague("nfl");
  }, []);
  useEffect(() => {
    document.documentElement.dataset.league = league;
  }, [league]);

  const switchLeague = (l: League) => {
    setLeague(l);
    setMarket(null);
    setPeriod(null);
  };
  const ready = data?.status === "ready" && data.league === league ? data : null;

  return (
    <div className="mx-auto max-w-[1200px] px-4 pb-16 pt-6 lg:px-6">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-border pb-4">
        <LeagueToggle value={league} onChange={switchLeague} />
        <h1 className="font-display text-xl font-bold">Replay</h1>
        {isFetching && <span className="text-sm text-muted">Loading…</span>}
      </div>

      {error ? (
        <p className="mt-6 text-danger-ink">Couldn't load the replay: {(error as Error).message}</p>
      ) : data?.status === "unavailable" && data.league === league ? (
        <p className="mt-6 rounded-control border border-dashed border-border bg-panel px-5 py-8 text-muted">{data.detail}</p>
      ) : !ready ? (
        <div className="mt-6 space-y-3">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-28 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : (
        <div className={cn("transition-opacity", isPlaceholderData && "opacity-60")}>
          <p className="mt-5 max-w-prose rounded-control border border-risky/40 bg-risky/10 px-4 py-3 text-sm text-ink" role="note">
            <span className="font-semibold text-risky-ink">Not a test of betting edge. </span>
            {ready.note}
          </p>
          <Controls r={ready} onMarket={setMarket} onPeriod={setPeriod} />
          <Summary r={ready} />
          <BiggestMisses rows={ready.biggest_misses} />
          <Rolling r={ready} weeks={weeks} onWeeks={setWeeks} />
          <PlayerTable r={ready} />
          <p className="mt-6 text-xs text-muted">{ready.source}</p>
        </div>
      )}
    </div>
  );
}

function Controls({ r, onMarket, onPeriod }: { r: Ready; onMarket: (m: string) => void; onPeriod: (p: string) => void }) {
  const i = r.periods.findIndex((p) => p.key === r.period.key);
  const older = r.periods[i + 1];
  const newer = r.periods[i - 1];
  const isNba = r.league === "nba";
  return (
    <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-3">
      <ToggleGroup type="single" value={r.market.key} onValueChange={(v) => v && onMarket(v)} aria-label="Market">
        {r.markets.map((m) => (
          <ToggleGroupItem key={m.key} value={m.key} className="data-[state=on]:text-accent-ink">{m.label}</ToggleGroupItem>
        ))}
      </ToggleGroup>
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon" aria-label={isNba ? "Previous game day" : "Previous week"} disabled={!older} onClick={() => older && onPeriod(older.key)}>
          <ChevronLeft className="size-4" />
        </Button>
        {isNba ? (
          <input
            type="date"
            aria-label="Game day"
            className="num h-9 rounded-control border border-border bg-panel px-2 text-sm"
            value={r.period.key}
            min={r.periods[r.periods.length - 1]?.key}
            max={r.periods[0]?.key}
            onChange={(e) => e.target.value && onPeriod(e.target.value)}
          />
        ) : (
          <select
            aria-label="NFL week"
            className="h-9 rounded-control border border-border bg-panel px-2 text-sm"
            value={r.period.key}
            onChange={(e) => onPeriod(e.target.value)}
          >
            {r.periods.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
          </select>
        )}
        <Button variant="ghost" size="icon" aria-label={isNba ? "Next game day" : "Next week"} disabled={!newer} onClick={() => newer && onPeriod(newer.key)}>
          <ChevronRight className="size-4" />
        </Button>
        <span className="text-sm text-muted">
          {r.period.label}
          {r.model_version && <> · model <span className="font-mono text-xs">{r.model_version}</span></>}
        </span>
      </div>
    </div>
  );
}

function Tile({ label, value, sub, tone }: { label: string; value: string; sub: React.ReactNode; tone?: "good" | "bad" }) {
  return (
    <div className="rounded-control border border-border bg-panel p-4">
      <p className="text-sm text-muted">{label}</p>
      <p className={cn("num mt-1 font-display text-xl font-bold", tone === "good" && "text-safe-ink", tone === "bad" && "text-danger-ink")}>{value}</p>
      <p className="mt-1 text-xs text-muted">{sub}</p>
    </div>
  );
}

function Summary({ r }: { r: Ready }) {
  const s = r.summary;
  if (!s) return <p className="mt-6 text-muted">No out-of-sample rows for this {r.league === "nba" ? "day" : "week"}.</p>;
  const model = s.model_mae_same_rows ?? s.model_mae;
  const gain = s.naive_mae ? (s.naive_mae - model) / s.naive_mae : null;
  return (
    <section aria-label="Summary" className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <Tile
        label="Average miss vs. baseline"
        value={`${fmt(model, 2)} vs ${fmt(s.naive_mae, 2)}`}
        tone={gain == null || r.naive_is_model ? undefined : gain > 0 ? "good" : "bad"}
        sub={
          r.naive_is_model
            ? "The app projects this market with the baseline itself, so they match."
            : gain == null
              ? `No ${r.baseline_name.toLowerCase()} available.`
              : `${Math.abs(gain * 100).toFixed(1)}% ${gain > 0 ? "better" : "worse"} than the ${r.baseline_name.toLowerCase()} (${s.n_naive} players with a baseline)`
        }
      />
      <Tile
        label="Actuals inside 25th–75th range"
        value={pct(s.coverage)}
        tone={Math.abs(s.coverage - 0.5) <= 0.1 ? "good" : "bad"}
        sub="Should be about 50%. One day is noisy; check the rolling view."
      />
      <Tile
        label="Average bias"
        value={signed(s.bias, 2)}
        sub={s.bias > 0 ? "Projected too high on average" : s.bias < 0 ? "Projected too low on average" : "No bias"}
      />
      <Tile label="Players" value={String(s.n)} sub={`${r.market.label}, ${r.period.label}`} />
    </section>
  );
}

function BiggestMisses({ rows }: { rows: ReplayRow[] }) {
  if (!rows.length) return null;
  return (
    <section aria-labelledby="misses-heading" className="mt-8">
      <h2 id="misses-heading" className="font-display text-lg font-bold">Five biggest misses</h2>
      <ol className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
        {rows.map((p) => (
          <li key={`${p.player_name}-${p.team}`} className="rounded-control border border-border bg-panel px-3 py-2.5">
            <p className="truncate font-medium">{p.player_name}</p>
            <p className="num mt-0.5 text-sm text-muted">
              Proj {fmt(p.predicted)} → <ActualValue actual={p.actual} lo={p.lo} hi={p.hi} />
            </p>
            <p className="num text-sm font-semibold">Miss {signed(p.miss)}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

function CoverageBar({ value }: { value: number }) {
  return (
    <div className="relative h-2 w-24 rounded-full bg-panel-2" aria-hidden>
      <div className="absolute inset-y-0 left-0 rounded-full bg-accent" style={{ width: `${Math.min(value, 1) * 100}%` }} />
      <div className="absolute -inset-y-1 left-1/2 w-px bg-ink" />
    </div>
  );
}

function Rolling({ r, weeks, onWeeks }: { r: Ready; weeks: number; onWeeks: (n: number) => void }) {
  const total = useMemo(() => {
    const n = r.rolling.reduce((a, w) => a + w.n, 0);
    return n ? r.rolling.reduce((a, w) => a + w.coverage * w.n, 0) / n : null;
  }, [r.rolling]);
  return (
    <section aria-labelledby="rolling-heading" className="mt-10">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 id="rolling-heading" className="font-display text-lg font-bold">Last {weeks} weeks</h2>
        <label className="flex items-center gap-2 text-sm text-muted">
          Weeks
          <select className="h-8 rounded-control border border-border bg-panel px-2 text-sm text-ink" value={weeks} onChange={(e) => onWeeks(Number(e.target.value))}>
            {WEEK_CHOICES.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
      </div>
      <p className="mt-1 max-w-prose text-sm text-muted">
        Ending with the selected {r.league === "nba" ? "day's week" : "week"}.
        {total != null && <> Across these weeks, <span className="num font-semibold text-ink">{pct(total)}</span> of actuals landed inside the range.</>}
        {" "}The tick on each bar marks 50%.
      </p>
      <div className="mt-3 overflow-x-auto rounded-control border border-border bg-panel">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead className="border-b border-border text-muted">
            <tr>
              <th scope="col" className="px-4 py-2 font-medium">{r.league === "nba" ? "Week" : "NFL week"}</th>
              <th scope="col" className="px-4 py-2 text-right font-medium">Players</th>
              <th scope="col" className="px-4 py-2 text-right font-medium">Avg miss</th>
              <th scope="col" className="px-4 py-2 text-right font-medium">Baseline</th>
              <th scope="col" className="px-4 py-2 font-medium">Inside range</th>
              <th scope="col" className="px-4 py-2 text-right font-medium">Bias</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {r.rolling.map((w) => {
              const model = w.model_mae_same_rows ?? w.model_mae;
              const better = w.naive_mae != null && model < w.naive_mae;
              return (
                <tr key={w.label}>
                  <th scope="row" className="px-4 py-2 font-medium">
                    {w.label}
                    {w.model_version && <span className="ml-2 font-mono text-xs font-normal text-muted">{w.model_version}</span>}
                  </th>
                  <td className="num px-4 py-2 text-right text-muted">{w.n}</td>
                  <td className={cn("num px-4 py-2 text-right font-semibold", !r.naive_is_model && w.naive_mae != null && (better ? "text-safe-ink" : "text-danger-ink"))}>{fmt(model, 2)}</td>
                  <td className="num px-4 py-2 text-right text-muted">{fmt(w.naive_mae, 2)}</td>
                  <td className="px-4 py-2">
                    <span className="flex items-center gap-2">
                      <CoverageBar value={w.coverage} />
                      <span className="num">{pct(w.coverage)}</span>
                    </span>
                  </td>
                  <td className="num px-4 py-2 text-right">{signed(w.bias, 2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function PlayerTable({ r }: { r: Ready }) {
  const [q, setQ] = useState("");
  const rows = q ? r.rows.filter((p) => p.player_name.toLowerCase().includes(q.toLowerCase())) : r.rows;
  return (
    <section aria-labelledby="players-heading" className="mt-10">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 id="players-heading" className="font-display text-lg font-bold">Every player, {r.period.label}</h2>
        <input
          type="search"
          placeholder="Filter players"
          aria-label="Filter players"
          className="h-8 w-48 rounded-control border border-border bg-panel px-2 text-sm"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
      </div>
      <MissLegend className="mt-2" />
      <p className="mt-1 text-xs text-muted">Miss = actual − projection.</p>
      <div className="mt-3 overflow-x-auto rounded-control border border-border bg-panel">
        <table className="w-full min-w-[620px] text-left text-sm">
          <thead className="border-b border-border text-muted">
            <tr>
              <th scope="col" className="px-4 py-2 font-medium">Player</th>
              <th scope="col" className="px-4 py-2 font-medium">Game</th>
              <th scope="col" className="px-4 py-2 text-right font-medium">Projection</th>
              <th scope="col" className="px-4 py-2 text-right font-medium">25th–75th</th>
              <th scope="col" className="px-4 py-2 text-right font-medium">Actual</th>
              <th scope="col" className="px-4 py-2 text-right font-medium">Miss</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {rows.map((p, i) => (
              <tr key={`${p.player_name}-${p.team}-${i}`}>
                <th scope="row" className="px-4 py-2 font-medium">{p.player_name}</th>
                <td className="px-4 py-2 text-muted">{p.team}{p.opponent ? ` vs ${p.opponent}` : ""}</td>
                <td className="num px-4 py-2 text-right">{fmt(p.predicted)}</td>
                <td className="num px-4 py-2 text-right text-muted">{fmt(p.lo)}–{fmt(p.hi)}</td>
                <td className="px-4 py-2 text-right"><ActualValue actual={p.actual} lo={p.lo} hi={p.hi} /></td>
                <td className="num px-4 py-2 text-right">{signed(p.miss)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
