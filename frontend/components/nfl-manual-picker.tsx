"use client";

import { useState } from "react";
import { Check, ChevronDown, Plus } from "lucide-react";
import { Headshot } from "./headshot";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { clockTime, pct } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { NflPickPlayer, NflPlayers } from "@/lib/types";

type Side = "over" | "under";

/** Same id the API gives a leg: player-market-side-line (line as Python prints a float). */
export const nflLegId = (playerId: string, market: string, side: Side, line: number) =>
  `${playerId}-${market}-${side}-${Number.isInteger(line) ? line.toFixed(1) : line}`;

/** A typed line: "54.5" stays; a whole number N means N+ (over) or N-1 or fewer (under), i.e. a line of N - 0.5. */
function parseLine(s: string): number | undefined {
  const v = Number(s.trim().replace(/\+$/, ""));
  if (!Number.isFinite(v) || v <= 0) return undefined;
  return Number.isInteger(v) ? v - 0.5 : v;
}

/**
 * NFL "Build my own": any projected player and market from the week's
 * pool, at any ladder value or a typed line, over or under. Picked legs
 * are scored by the same engine as suggestions. Players listed Out are
 * shown but can't be picked.
 */
export function NflManualPicker({
  data,
  loading,
  picked,
  onAdd,
  excludedGames,
}: {
  data?: NflPlayers;
  loading: boolean;
  picked: string[];
  onAdd: (id: string) => void;
  excludedGames: string[];
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<string | null>(null);

  if (loading || !data)
    return (
      <div className="space-y-2">
        {Array.from({ length: 5 }, (_, i) => <Skeleton key={i} className="h-14 w-full" />)}
      </div>
    );
  if (data.status !== "ready")
    return <p className="rounded-control border border-dashed border-border bg-panel px-5 py-8 text-center text-sm text-muted">{data.message ?? "The NFL pool isn't ready."}</p>;

  const q = query.trim().toLowerCase();
  const match = (p: NflPickPlayer) => !q || `${p.player_name} ${p.team} ${p.position}`.toLowerCase().includes(q);
  const players = data.players.filter((p) => !excludedGames.includes(p.game_id ?? "") && match(p));
  const out = data.out.filter(match);
  // player+market already in the slip (one leg per player prop)
  const taken = new Set(picked.map((id) => id.split("-").slice(0, -2).join("-")));

  return (
    <div className="space-y-3">
      <input
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search players or teams"
        aria-label="Search players or teams"
        className="h-9 w-full rounded-control border border-border bg-panel px-3 text-sm outline-none focus:border-ink"
      />
      {players.length === 0 && out.length === 0 ? (
        <p className="rounded-control border border-dashed border-border bg-panel px-5 py-8 text-center text-sm text-muted">No players match.</p>
      ) : (
        <ul className="divide-y divide-border overflow-hidden rounded-control border border-border bg-panel">
          {players.map((p) => (
            <li key={p.player_id}>
              <button
                type="button"
                onClick={() => setOpen(open === p.player_id ? null : p.player_id)}
                aria-expanded={open === p.player_id}
                className="flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors hover:bg-panel-2 sm:px-4"
              >
                <Headshot src={null} name={p.player_name} className="h-9 w-9" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-medium leading-tight">
                    {p.player_name}
                    {p.injury && <span className="ml-2 rounded-full bg-risky/15 px-1.5 py-0.5 text-xs text-risky-ink">Q</span>}
                  </span>
                  <span className="block truncate text-xs text-muted">
                    {p.position} · {p.matchup}
                    {p.tip_utc && ` · ${clockTime(p.tip_utc)} ET`}
                  </span>
                </span>
                <span className="hidden shrink-0 text-xs text-muted sm:block">
                  {Object.entries(p.markets).map(([m, v]) => `${v.mean.toFixed(m === "receptions" ? 1 : 0)} ${data.market_labels[m]}`).join(" · ")}
                </span>
                <ChevronDown className={cn("h-4 w-4 shrink-0 text-muted transition-transform", open === p.player_id && "rotate-180")} aria-hidden />
              </button>
              {open === p.player_id && <LegEditor player={p} labels={data.market_labels} taken={taken} onAdd={onAdd} />}
            </li>
          ))}
          {out.map((p) => (
            <li key={`out-${p.player_id}`} aria-disabled className="flex items-center gap-3 px-3 py-2.5 opacity-45 sm:px-4">
              <Headshot src={null} name={p.player_name} className="h-9 w-9" />
              <span className="min-w-0 flex-1">
                <span className="block truncate font-medium leading-tight">{p.player_name}</span>
                <span className="block text-xs text-muted">{p.position} · {p.team}</span>
              </span>
              <span className="shrink-0 rounded-full bg-danger/15 px-2 py-0.5 text-xs text-danger-ink">Out · can't be picked</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function LegEditor({
  player,
  labels,
  taken,
  onAdd,
}: {
  player: NflPickPlayer;
  labels: Record<string, string>;
  taken: Set<string>;
  onAdd: (id: string) => void;
}) {
  const markets = Object.keys(player.markets);
  const [market, setMarket] = useState(markets[0]);
  const [side, setSide] = useState<Side>("over");
  const [custom, setCustom] = useState("");
  const m = player.markets[market];
  const inSlip = taken.has(`${player.player_id}-${market}`);
  const customLine = parseLine(custom);
  const add = (line: number) => onAdd(nflLegId(player.player_id, market, side, line));

  return (
    <div className="space-y-3 border-t border-border bg-panel-2/50 px-3 py-3 sm:px-4">
      <div className="flex flex-wrap items-center gap-2">
        <ToggleGroup type="single" value={market} onValueChange={(v) => v && setMarket(v)} aria-label="Market" className="flex flex-wrap">
          {markets.map((k) => (
            <ToggleGroupItem key={k} value={k} className="px-2.5 text-sm">{labels[k]}</ToggleGroupItem>
          ))}
        </ToggleGroup>
        <ToggleGroup type="single" value={side} onValueChange={(v) => v && setSide(v as Side)} aria-label="Over or under" className="ml-auto grid grid-cols-2">
          <ToggleGroupItem value="over" className="px-3 text-sm">Over</ToggleGroupItem>
          <ToggleGroupItem value="under" className="px-3 text-sm">Under</ToggleGroupItem>
        </ToggleGroup>
      </div>
      <p className="text-xs text-muted">
        Projects {m.mean.toFixed(1)} {labels[market]}.{" "}
        {inSlip ? "Already in the slip: remove that leg first to pick another line." : "Model hit chance at each ladder value:"}
      </p>
      <div className="grid grid-cols-3 gap-1.5 sm:grid-cols-6">
        {m.ladder.map((r) => {
          const p = side === "over" ? r.p_over : 1 - r.p_over;
          return (
            <button
              key={r.rung}
              type="button"
              disabled={inSlip}
              onClick={() => add(r.line)}
              aria-label={`Add ${player.player_name} ${side === "over" ? `${r.rung}+` : `under ${r.line}`} ${labels[market]}`}
              className="rounded-control border border-border bg-panel px-2 py-1.5 text-left hover:border-ink disabled:cursor-not-allowed disabled:opacity-45"
            >
              <span className="block text-xs text-muted">{side === "over" ? `${r.rung}+` : `Under ${r.line}`}</span>
              <span className="num flex items-center justify-between text-base font-semibold">
                {pct(p)}
                <Plus className="h-3.5 w-3.5 text-muted" aria-hidden />
              </span>
            </button>
          );
        })}
      </div>
      <form
        className="flex items-end gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (customLine !== undefined && !inSlip) add(customLine);
        }}
      >
        <label className="flex-1 text-xs text-muted">
          Or type a line
          <input
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
            placeholder={side === "over" ? "e.g. 54.5 or 55+" : "e.g. 54.5"}
            inputMode="decimal"
            aria-invalid={custom.trim() !== "" && customLine === undefined}
            className="num mt-0.5 h-8 w-full rounded-control border border-border bg-panel px-2 text-sm text-ink outline-none focus:border-ink aria-[invalid=true]:border-danger"
          />
        </label>
        <Button type="submit" size="sm" disabled={customLine === undefined || inSlip}>
          <Check className="h-4 w-4" aria-hidden /> Add {side} {customLine ?? ""}
        </Button>
      </form>
      <p className="text-[11px] text-muted">A whole number N means N+ for an over, N−1 or fewer for an under (a line of N − 0.5). The hit chance shows once it's in the slip.</p>
    </div>
  );
}
