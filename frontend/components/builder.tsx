"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { BankrollInput } from "./bankroll-input";
import { BetSlip, type SlipNotice } from "./bet-slip";
import { ManualPicker } from "./manual-picker";
import { SideSelector } from "./side-selector";
import { CourtDivider } from "./court-divider";
import { LeagueToggle } from "./league-toggle";
import { ParlayRow, ParlayRowSkeleton } from "./parlay-row";
import { TIER_LABEL } from "./tier-badge";
import { TierSelector } from "./tier-selector";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { replaceLeg, scoreParlay, useLegs, useParlays, useSlate, useTiers, type PoolFilters } from "@/lib/api";
import { clockTime, money, pct, slateDate } from "@/lib/format";
import { useStored } from "@/lib/storage";
import { cn } from "@/lib/utils";
import type { GameInfo, League, Leg, Parlay, SearchReport, Sides, SlateInfo, TierConfig, TierName } from "@/lib/types";

const DEFAULT_BANKROLL = 500;

function usePersisted<T>(key: string, initial: T, read: (raw: string) => T | undefined) {
  const [value, setValue] = useState<T>(initial);
  useEffect(() => {
    try {
      const raw = localStorage.getItem(key);
      const parsed = raw == null ? undefined : read(raw);
      if (parsed !== undefined) setValue(parsed);
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const set = (v: T) => {
    setValue(v);
    try {
      localStorage.setItem(key, String(v));
    } catch {}
  };
  return [value, set] as const;
}

interface Veto {
  id: string;
  label: string;
}

const legLabel = (l: Leg) => `${l.player_name} ${l.label}`;

export function Builder() {
  const [league, setLeague] = usePersisted<League>("league", "nba", (r) => (r === "nfl" || r === "nba" ? r : undefined));
  const [bankroll, setBankroll] = usePersisted<number>("bankroll", DEFAULT_BANKROLL, (r) => (Number(r) > 0 ? Number(r) : undefined));
  const [sides, setSides] = usePersisted<Sides>("sides", "both", (r) => (r === "over" || r === "under" || r === "both" ? r : undefined));
  const [tier, setTier] = useState<TierName>("risky");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [excludedGames, setExcludedGames] = useState<string[]>([]);
  // Vetoed legs last for the browser session, so they don't come back on refresh.
  const [vetoes, setVetoes] = useStored<Veto[]>("vetoed-legs", [], { session: true });

  // A suggested parlay after one or more vetoes; `history` lets each veto be undone.
  const [edited, setEdited] = useState<Parlay | null>(null);
  const [history, setHistory] = useState<{ vetoId: string; prev: Parlay | null }[]>([]);
  // Undo is bound at render time from `vetoId`, so it always sees current state.
  const [notice, setNotice] = useState<(Omit<SlipNotice, "onUndo"> & { vetoId?: string }) | null>(null);
  const [busyLegId, setBusyLegId] = useState<string | null>(null);

  const [mode, setMode] = useState<"suggested" | "manual">("suggested");
  const [picked, setPicked] = useState<string[]>([]);

  // `?league=` (set by the layout's init script) wins over the saved league on first load.
  useEffect(() => {
    const fromDom = document.documentElement.dataset.league;
    if (fromDom === "nfl" || fromDom === "nba") setLeague(fromDom);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    document.documentElement.dataset.league = league;
  }, [league]);

  const filters: PoolFilters = { sides, vetoed: vetoes.map((v) => v.id), excludedGames };
  const tiers = useTiers();
  const slate = useSlate(league);
  const parlaysQuery = useParlays(league, tier, bankroll, filters);
  const legsQuery = useLegs(league, filters, mode === "manual");
  const data = parlaysQuery.data;
  const loading = !data || data.status === "warming" || data.status === "idle";

  const manualQuery = useQuery({
    queryKey: ["score", picked, tier, bankroll],
    queryFn: () => scoreParlay({ leg_ids: picked, tier, bankroll }).then((r) => r.parlay),
    enabled: mode === "manual" && picked.length > 0,
    placeholderData: (prev) => prev,
  });

  const parlays = data?.parlays ?? [];
  const suggested = parlays.find((p) => p.id === selectedId) ?? parlays[0] ?? null;
  const slipParlay =
    mode === "manual" ? (picked.length ? manualQuery.data ?? null : null) : history.length ? edited : suggested;

  const resetEdits = () => {
    setEdited(null);
    setHistory([]);
    setNotice(null);
  };

  const undoVeto = (id: string) => {
    setVetoes(vetoes.filter((v) => v.id !== id));
    const last = history[history.length - 1];
    if (last?.vetoId === id) {
      setHistory(history.slice(0, -1));
      setEdited(last.prev);
    }
    setNotice(null);
  };

  const veto = async (leg: Leg) => {
    if (!slipParlay) return;
    setBusyLegId(leg.id);
    const vetoed = [...filters.vetoed, leg.id];
    try {
      const res = await replaceLeg({
        leg_ids: slipParlay.legs.map((l) => l.id),
        veto_id: leg.id,
        tier,
        sides,
        vetoed,
        excluded_games: excludedGames,
        bankroll,
        name: slipParlay.name,
      });
      setVetoes([...vetoes, { id: leg.id, label: legLabel(leg) }]);
      setHistory([...history, { vetoId: leg.id, prev: slipParlay }]);
      setEdited(res.parlay);
      setNotice(
        res.added && res.parlay
          ? {
              tone: "info",
              text: `Vetoed ${legLabel(leg)}. Swapped in ${legLabel(res.added)} — hit chance ${pct(slipParlay.p_hit)} → ${pct(res.parlay.p_hit)}.`,
              vetoId: leg.id,
            }
          : { tone: "warn", text: `Vetoed ${legLabel(leg)}. ${res.message ?? ""}`, vetoId: leg.id },
      );
    } catch (e) {
      setNotice({ tone: "warn", text: `Couldn't veto that leg: ${(e as Error).message}` });
    } finally {
      setBusyLegId(null);
    }
  };

  const switchTier = (t: TierName) => {
    setTier(t);
    setSelectedId(null);
    resetEdits();
  };

  const slipProps = {
    parlay: slipParlay,
    tier,
    league,
    notice:
      mode === "manual"
        ? manualQuery.error
          ? { tone: "warn" as const, text: (manualQuery.error as Error).message }
          : null
        : notice && { tone: notice.tone, text: notice.text, onUndo: notice.vetoId ? () => undoVeto(notice.vetoId!) : undefined },
    legAction: mode === "manual" ? ("remove" as const) : ("veto" as const),
    onLegAction: mode === "manual" ? (l: Leg) => setPicked(picked.filter((id) => id !== l.id)) : veto,
    busyLegId,
    emptyText: mode === "manual" ? "Add legs from the list to score them." : "Pick a suggested parlay to load it here.",
  };

  return (
    <div className="mx-auto max-w-[1600px] px-4 pb-28 pt-4 lg:px-6 lg:pb-10">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-border pb-4">
        <LeagueToggle value={league} onChange={(l) => { setLeague(l); setSelectedId(null); resetEdits(); }} />
        <SlateSummary league={league} slate={slate.data} />
        <div className="ml-auto">
          <BankrollInput value={bankroll} onChange={setBankroll} />
        </div>
      </div>

      <div className="mt-6 grid gap-6 md:grid-cols-[220px_minmax(0,1fr)] lg:grid-cols-[220px_minmax(0,1fr)_minmax(340px,380px)] xl:gap-8">
        <aside className="space-y-6">
          <section>
            <h2 className="mb-2 text-sm font-medium text-muted">Tier</h2>
            <TierSelector value={tier} onChange={switchTier} tiers={tiers.data} />
          </section>
          <section>
            <h2 className="mb-2 text-sm font-medium text-muted">Over / under</h2>
            <SideSelector value={sides} onChange={(s) => { setSides(s); setSelectedId(null); resetEdits(); }} />
          </section>
          <Filters
            games={slate.data?.games ?? []}
            excluded={excludedGames}
            onToggle={(id) => {
              setExcludedGames(excludedGames.includes(id) ? excludedGames.filter((g) => g !== id) : [...excludedGames, id]);
              setSelectedId(null);
              resetEdits();
            }}
          />
          <VetoList vetoes={vetoes} onUndo={undoVeto} onClear={() => { setVetoes([]); resetEdits(); }} />
        </aside>

        <section aria-labelledby="suggested-heading" className="min-w-0">
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-2">
            <h1 id="suggested-heading" className="font-display text-xl font-bold">
              {mode === "manual" ? "Build my own" : "Suggested parlays"}
            </h1>
            {mode === "manual" ? (
              <Button variant="ghost" size="sm" onClick={() => setMode("suggested")}>Back to suggestions</Button>
            ) : (
              !loading && data?.status === "ready" && (
                <span className="text-sm text-muted">
                  {parlays.length} {TIER_LABEL[tier].toLowerCase()} {parlays.length === 1 ? "parlay" : "parlays"}
                </span>
              )
            )}
          </div>
          <CourtDivider league={league} />

          {mode === "manual" ? (
            <>
              <p className="mb-3 text-sm text-muted">
                Pick any legs. They're scored by the same engine as the suggestions, with the stake capped at {TIER_LABEL[tier]}'s
                maximum. Model probabilities are shown as they are, even when they're low.
              </p>
              <ManualPicker
                legs={legsQuery.data?.legs ?? []}
                loading={!legsQuery.data || legsQuery.data.status !== "ready"}
                picked={picked}
                onToggle={(l) => setPicked(picked.includes(l.id) ? picked.filter((id) => id !== l.id) : [...picked, l.id])}
                tierCfg={tiers.data?.[tier]}
                tierLabel={TIER_LABEL[tier]}
              />
            </>
          ) : (
            <>
              <div className={cn("space-y-3 transition-opacity", parlaysQuery.isPlaceholderData && "opacity-60")}>
                {data?.status === "unavailable" ? (
                  <EmptyState
                    title="No NFL parlays yet"
                    body={data.message ?? "NFL models aren't built yet."}
                    action={<Button onClick={() => setLeague("nba")}>Switch to NBA</Button>}
                  />
                ) : data?.status === "error" ? (
                  <EmptyState
                    title="The slate didn't build"
                    body={`The backend hit an error building today's leg pool: ${data.error}. Check the API logs, then retry.`}
                    action={<Button onClick={() => fetch("/api/slate/refresh", { method: "POST" }).then(() => parlaysQuery.refetch())}>Retry</Button>}
                  />
                ) : loading ? (
                  Array.from({ length: 3 }, (_, i) => <ParlayRowSkeleton key={i} />)
                ) : parlays.length === 0 ? (
                  <NoParlays tier={tier} cfg={tiers.data?.[tier]} report={data?.report} sides={sides} />
                ) : (
                  <>
                    {parlays.every((p) => p.flagged) && (
                      <p className="rounded-control border border-risky/60 bg-risky/10 px-3 py-2 text-sm">
                        <span className="font-medium text-risky-ink">Every parlay here tripped the data check, so stakes are $0.</span>{" "}
                        <span className="text-muted">The model and the book disagree by more than 15 points on at least one leg.</span>
                      </p>
                    )}
                    {parlays.map((p) => (
                      <ParlayRow
                        key={p.id}
                        parlay={p}
                        selected={!history.length && p.id === suggested?.id}
                        onSelect={() => { setSelectedId(p.id); resetEdits(); }}
                      />
                    ))}
                  </>
                )}
              </div>

              <div className="mt-6 flex flex-wrap items-center gap-3">
                <Button onClick={() => setMode("manual")} disabled={data?.status !== "ready"}>Build my own</Button>
                <span className="text-sm text-muted">Pick your own legs and score them with the same engine.</span>
              </div>
            </>
          )}
        </section>

        <div className="hidden lg:block">
          <div className="sticky top-4">
            {loading && data?.status !== "unavailable" ? <SlipSkeleton /> : <BetSlip {...slipProps} />}
          </div>
        </div>
      </div>

      <MobileSlip {...slipProps} />
    </div>
  );
}

function SlateSummary({ league, slate }: { league: League; slate?: SlateInfo }) {
  if (!slate || slate.status === "warming" || slate.status === "idle")
    return <Skeleton className="h-5 w-56" aria-label="Loading slate" />;
  if (slate.status === "unavailable") return <span className="text-sm text-muted">NFL arrives in Phase 6</span>;
  if (slate.status === "error") return <span className="text-sm text-danger-ink">Slate failed to build</span>;
  if (!slate.n_games)
    return <span className="text-sm text-muted">No {league.toUpperCase()} games with odds yet</span>;

  const tip = slate.first_tip_utc ? new Date(slate.first_tip_utc) : null;
  const tonight = tip && tip.toDateString() === new Date().toDateString();
  return (
    <div className="flex flex-wrap items-baseline gap-x-3">
      <span className="font-medium">
        {tonight ? "Tonight" : `Next slate: ${slateDate(slate.first_tip_utc!)}`} ·{" "}
        <span className="num text-lg font-semibold">{slate.n_games}</span> {slate.n_games === 1 ? "game" : "games"}
      </span>
      {slate.odds_pulled_at && (
        <span className="text-xs text-muted">
          Odds from {slateDate(slate.odds_pulled_at + "Z")}, {clockTime(slate.odds_pulled_at + "Z")} ET
        </span>
      )}
    </div>
  );
}

function Filters({ games, excluded, onToggle }: { games: GameInfo[]; excluded: string[]; onToggle: (id: string) => void }) {
  return (
    <section className="hidden md:block">
      <h2 className="mb-2 text-sm font-medium text-muted">Games</h2>
      {games.length === 0 ? (
        <p className="text-sm text-muted">No games loaded.</p>
      ) : (
        <ul className="space-y-1.5">
          {games.map((g) => (
            <li key={g.game_id}>
              <label className="flex cursor-pointer items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={!excluded.includes(g.game_id)}
                  onChange={() => onToggle(g.game_id)}
                  className="h-4 w-4 rounded accent-[var(--accent)]"
                />
                <span className="font-medium">{g.teams.join(" · ") || "Unmatched game"}</span>
                {g.tip_utc && <span className="ml-auto text-xs text-muted">{clockTime(g.tip_utc)}</span>}
              </label>
            </li>
          ))}
        </ul>
      )}
      <h2 className="mb-1 mt-5 text-sm font-medium text-muted">Markets</h2>
      <p className="text-sm">Points</p>
      <p className="text-xs text-muted">Rebounds, assists and 3PM unlock once their odds are collected.</p>
    </section>
  );
}

function VetoList({ vetoes, onUndo, onClear }: { vetoes: Veto[]; onUndo: (id: string) => void; onClear: () => void }) {
  if (!vetoes.length) return null;
  return (
    <section>
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-sm font-medium text-muted">Vetoed this session</h2>
        <button type="button" onClick={onClear} className="text-xs text-muted hover:text-ink">Clear all</button>
      </div>
      <ul className="space-y-1.5">
        {vetoes.map((v) => (
          <li key={v.id} className="flex items-baseline gap-2 text-sm">
            <span className="min-w-0 flex-1 truncate text-muted line-through decoration-muted/60">{v.label}</span>
            <button type="button" onClick={() => onUndo(v.id)} className="shrink-0 text-xs font-medium text-accent-ink hover:underline">
              Undo
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** Never a blank tier: say which constraint emptied it, with the numbers. */
function NoParlays({ tier, cfg, report, sides }: { tier: TierName; cfg?: TierConfig; report?: SearchReport; sides: Sides }) {
  const facts: string[] = [];
  if (report) {
    facts.push(`${report.n_pool} ${sides === "both" ? "" : `${sides} `}legs in the pool`);
    if (cfg && report.n_pool) facts.push(`${report.n_eligible} clear ${pct(cfg.min_leg_prob)}`);
    if (report.best_leg_p != null) facts.push(`best leg ${pct(report.best_leg_p)}`);
    if (report.n_combinations)
      facts.push(`${report.n_combinations} combination${report.n_combinations === 1 ? "" : "s"} simulated`);
  }
  const hint = sides === "both" ? "Try another tier, or turn games back on." : "Try another tier, or switch Over / under to Both.";
  return (
    <div className="rounded-control border border-dashed border-border bg-panel px-5 py-8 text-center">
      <p className="font-display text-lg font-semibold">No {TIER_LABEL[tier].toLowerCase()} parlays in this pool</p>
      <p className="mx-auto mt-1 max-w-prose text-sm text-ink">{report?.reason ?? "Nothing in tonight's leg pool fits this tier."}</p>
      {facts.length > 0 && <p className="num mx-auto mt-2 max-w-prose text-xs text-muted">{facts.join(" · ")}</p>}
      <p className="mx-auto mt-3 max-w-prose text-xs text-muted">{hint}</p>
    </div>
  );
}

function EmptyState({ title, body, action }: { title: string; body: string; action?: React.ReactNode }) {
  return (
    <div className="rounded-control border border-dashed border-border bg-panel px-5 py-8 text-center">
      <p className="font-display text-lg font-semibold">{title}</p>
      <p className="mx-auto mt-1 max-w-prose text-sm text-muted">{body}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

function SlipSkeleton() {
  return (
    <div className="rounded-slip bg-panel p-5 shadow-slip">
      <Skeleton className="h-6 w-24" />
      {Array.from({ length: 3 }, (_, i) => (
        <div key={i} className="mt-5 flex gap-3">
          <Skeleton className="h-10 w-10 rounded-full" />
          <div className="flex-1 space-y-2">
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-3 w-full" />
          </div>
        </div>
      ))}
      <Skeleton className="mt-8 h-10 w-1/2" />
    </div>
  );
}

function MobileSlip(props: React.ComponentProps<typeof BetSlip>) {
  const { parlay } = props;
  if (!parlay) return null;
  return (
    <Sheet>
      <SheetTrigger asChild>
        <button
          type="button"
          className="fixed inset-x-3 bottom-3 z-30 flex items-center justify-between rounded-slip bg-ink px-5 py-3.5 text-panel shadow-slip lg:hidden"
        >
          <span className="font-medium">Bet slip</span>
          <span className="num text-lg font-semibold">
            {parlay.legs.length} legs · {pct(parlay.p_hit)} · {money(parlay.stake)}
          </span>
        </button>
      </SheetTrigger>
      <SheetContent>
        <SheetTitle className="sr-only">Bet slip</SheetTitle>
        <SheetDescription className="sr-only">Legs and totals for the selected parlay</SheetDescription>
        <div className="pt-6">
          <BetSlip {...props} />
        </div>
      </SheetContent>
    </Sheet>
  );
}
