"use client";

import { useMemo } from "react";
import Link from "next/link";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { LimitBars } from "@/components/loss-limits";
import { TierBadge } from "@/components/tier-badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useBetMutations, useBets } from "@/lib/api";
import { american, money, pct, slateDate } from "@/lib/format";
import { usePaperMode, useStored } from "@/lib/storage";
import { cn } from "@/lib/utils";
import type { Bet, BetResult, TierName } from "@/lib/types";

const TIERS: TierName[] = ["safe", "risky", "dangerous"];
const RESULT_LABEL: Record<BetResult, string> = { pending: "Pending", won: "Won", lost: "Lost", push: "Push" };

export default function TrackerPage() {
  const [paper, setPaper] = usePaperMode();
  const [startBankroll, setStartBankroll] = useStored<number>("bankroll", 500, {
    parse: (r) => (Number(r) > 0 ? Number(r) : undefined),
  });
  const { data, error } = useBets();
  const bets = useMemo(() => (data?.bets ?? []).filter((b) => b.paper === paper), [data, paper]);

  return (
    <div className="mx-auto max-w-[1100px] px-4 pb-16 pt-8 lg:px-6">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <h1 className="font-display text-2xl font-bold">Bet tracker</h1>
        <label className="ml-auto flex cursor-pointer items-center gap-2.5 text-sm">
          <span className="text-muted">Paper trading</span>
          <button
            type="button"
            role="switch"
            aria-checked={paper}
            onClick={() => setPaper(!paper)}
            className={cn(
              "relative h-6 w-11 rounded-full border transition-colors",
              paper ? "border-accent bg-accent" : "border-border bg-panel-2",
            )}
          >
            <span className={cn("absolute top-0.5 h-[18px] w-[18px] rounded-full bg-panel shadow transition-all", paper ? "left-[22px]" : "left-0.5")} />
          </button>
        </label>
      </div>
      <p className="mt-1 text-sm text-muted">
        {paper
          ? "Showing paper bets only — tracked without real money. Bets you log from the slip while this is on are paper bets."
          : "Showing real-money bets. Turn on paper trading to track picks without staking anything."}
      </p>
      {!paper && <LimitBars />}

      {error ? (
        <p className="mt-6 text-danger-ink">Couldn't load bets: {(error as Error).message}</p>
      ) : !data ? (
        <div className="mt-6 space-y-3">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : bets.length === 0 ? (
        <div className="mt-6 rounded-control border border-dashed border-border bg-panel px-5 py-10 text-center">
          <p className="font-display text-lg font-semibold">No {paper ? "paper " : ""}bets logged yet</p>
          <p className="mx-auto mt-1 max-w-prose text-sm text-muted">Load a parlay into the bet slip and press “Log {paper ? "paper " : ""}bet”.</p>
          <Button asChild className="mt-4"><Link href="/">Build parlays</Link></Button>
        </div>
      ) : (
        <Tracker bets={bets} start={startBankroll} onStart={setStartBankroll} />
      )}
    </div>
  );
}

function Tracker({ bets, start, onStart }: { bets: Bet[]; start: number; onStart: (n: number) => void }) {
  const settled = bets.filter((b) => b.result !== "pending");
  const decided = settled.filter((b) => b.result !== "push");
  const wins = decided.filter((b) => b.result === "won").length;
  const net = settled.reduce((s, b) => s + b.profit, 0);
  const staked = settled.reduce((s, b) => s + b.stake, 0);
  const expectedWins = decided.reduce((s, b) => s + b.model_p, 0);

  // Bankroll after each settlement, in settlement order.
  const curve = useMemo(() => {
    let bal = start;
    const pts = [{ i: 0, label: "Start", balance: start }];
    [...settled]
      .sort((a, b) => (a.settled_at ?? "").localeCompare(b.settled_at ?? ""))
      .forEach((b, idx) => {
        bal += b.profit;
        pts.push({ i: idx + 1, label: `${b.name} · ${RESULT_LABEL[b.result]}`, balance: Math.round(bal * 100) / 100 });
      });
    return pts;
  }, [settled, start]);

  return (
    <>
      <dl className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Bankroll now" value={money(start + net)} sub={`from ${money(start)}`} />
        <Stat label="Net" value={`${net >= 0 ? "+" : "−"}${money(Math.abs(net))}`} sub={`${money(staked)} settled`} tone={net > 0 ? "pos" : undefined} />
        <Stat label="ROI" value={staked ? `${((net / staked) * 100).toFixed(1)}%` : "—"} sub="net ÷ amount staked" />
        <Stat
          label="Record"
          value={`${wins}–${decided.length - wins}`}
          sub={decided.length ? `model expected ${expectedWins.toFixed(1)} wins` : "nothing settled yet"}
        />
      </dl>

      <section aria-labelledby="curve-heading" className="mt-8 rounded-control border border-border bg-panel p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 id="curve-heading" className="font-display text-lg font-semibold">Bankroll</h2>
          <StartInput value={start} onChange={onStart} />
        </div>
        {curve.length < 2 ? (
          <p className="py-10 text-center text-sm text-muted">The curve starts once a bet is settled.</p>
        ) : (
          <div className="mt-2 h-64" role="img" aria-label={`Bankroll over ${curve.length - 1} settled bets, from ${money(start)} to ${money(curve[curve.length - 1].balance)}`}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curve} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
                <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="0" />
                <XAxis dataKey="i" tickLine={false} axisLine={{ stroke: "var(--border)" }} tick={{ fill: "var(--muted)", fontSize: 12 }} allowDecimals={false} />
                <YAxis
                  tickLine={false}
                  axisLine={false}
                  width={64}
                  tick={{ fill: "var(--muted)", fontSize: 12 }}
                  tickFormatter={(v) => money(v)}
                  domain={["auto", "auto"]}
                />
                <ReferenceLine y={start} stroke="var(--muted)" strokeDasharray="4 4" />
                <Tooltip
                  cursor={{ stroke: "var(--muted)", strokeWidth: 1 }}
                  contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 8, color: "var(--ink)" }}
                  labelFormatter={(_, p) => (p?.[0]?.payload as { label: string } | undefined)?.label ?? ""}
                  formatter={(v) => [money(Number(v)), "Bankroll"]}
                />
                <Line type="linear" dataKey="balance" stroke="var(--accent)" strokeWidth={2} dot={{ r: 4, fill: "var(--accent)", stroke: "var(--panel)", strokeWidth: 2 }} activeDot={{ r: 5 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      <section aria-labelledby="tier-heading" className="mt-8">
        <h2 id="tier-heading" className="font-display text-lg font-semibold">Hit rate by tier</h2>
        <div className="mt-3 overflow-x-auto rounded-control border border-border bg-panel">
          <table className="w-full min-w-[520px] text-left text-sm">
            <thead className="border-b border-border text-muted">
              <tr>
                <th scope="col" className="px-4 py-2.5 font-medium">Tier</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">Settled</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">Hit rate</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">Model expected</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">Net</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {TIERS.map((t) => {
                const d = decided.filter((b) => b.tier === t);
                const w = d.filter((b) => b.result === "won").length;
                const n = settled.filter((b) => b.tier === t).reduce((s, b) => s + b.profit, 0);
                return (
                  <tr key={t}>
                    <th scope="row" className="px-4 py-2.5 font-medium"><TierBadge tier={t} /></th>
                    <td className="num px-4 py-2.5 text-right text-base">{d.length}</td>
                    <td className="num px-4 py-2.5 text-right text-base font-semibold">{d.length ? `${pct(w / d.length)} (${w}/${d.length})` : "—"}</td>
                    <td className="num px-4 py-2.5 text-right text-base text-muted">{d.length ? pct(d.reduce((s, b) => s + b.model_p, 0) / d.length) : "—"}</td>
                    <td className="num px-4 py-2.5 text-right text-base">{d.length ? `${n >= 0 ? "+" : "−"}${money(Math.abs(n))}` : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-muted">Pushes are left out of hit rate. A handful of bets says almost nothing about the model.</p>
      </section>

      <BetList bets={bets} />
    </>
  );
}

function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: "pos" }) {
  return (
    <div className="rounded-control border border-border bg-panel p-3">
      <dt className="text-xs text-muted">{label}</dt>
      <dd className={cn("num text-xl font-bold leading-tight", tone === "pos" && "text-safe-ink")}>{value}</dd>
      {sub && <dd className="text-xs text-muted">{sub}</dd>}
    </div>
  );
}

function StartInput({ value, onChange }: { value: number; onChange: (n: number) => void }) {
  return (
    <label className="flex items-center gap-2 text-xs text-muted">
      Starting bankroll
      <span className="flex items-center rounded-control border border-border bg-surface">
        <span className="pl-2">$</span>
        <input
          inputMode="decimal"
          defaultValue={value}
          key={value}
          onBlur={(e) => {
            const n = Number(e.target.value.replace(/[$,\s]/g, ""));
            if (n > 0) onChange(n);
          }}
          className="num h-7 w-20 bg-transparent px-1 text-sm text-ink outline-none"
        />
      </span>
    </label>
  );
}

function BetList({ bets }: { bets: Bet[] }) {
  const { settle, remove } = useBetMutations();
  return (
    <section aria-labelledby="bets-heading" className="mt-8">
      <h2 id="bets-heading" className="font-display text-lg font-semibold">Bets</h2>
      <ul className="mt-3 space-y-3">
        {[...bets].reverse().map((b) => (
          <li key={b.id} className="rounded-control border border-border bg-panel p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="font-display text-lg font-semibold leading-tight">
                  {b.name} <TierBadge tier={b.tier} className="ml-1 align-middle" />
                  {b.paper && <span className="ml-1.5 rounded-full border border-border px-2 py-0.5 align-middle font-sans text-xs font-normal text-muted">Paper</span>}
                </p>
                <p className="text-xs text-muted">
                  Logged {slateDate(b.created_at)} · {b.legs.length} legs · model {pct(b.model_p)} · {american(b.american_odds)}
                </p>
              </div>
              <div className="text-right">
                <p className="num text-lg font-semibold">{money(b.stake)}</p>
                <p className={cn("num text-xs", b.profit > 0 ? "text-safe-ink" : "text-muted")}>
                  {b.result === "pending" ? `pays ${money(b.stake * b.decimal_odds)}` : `${b.profit >= 0 ? "+" : "−"}${money(Math.abs(b.profit))}`}
                </p>
              </div>
            </div>
            <ul className="mt-2 text-sm">
              {b.legs.map((l) => (
                <li key={l.player_name + l.label} className="flex justify-between gap-2">
                  <span className="truncate">{l.player_name} <span className="text-accent-ink">{l.label}</span></span>
                  <span className="num text-muted">{american(l.american_odds)}</span>
                </li>
              ))}
            </ul>
            <div className="mt-3 flex flex-wrap items-center gap-1.5" role="group" aria-label={`Result for ${b.name}`}>
              {(["won", "lost", "push", "pending"] as BetResult[]).map((r) => (
                <button
                  key={r}
                  type="button"
                  aria-pressed={b.result === r}
                  onClick={() => settle.mutate({ id: b.id, result: r })}
                  className={cn(
                    "rounded-control border px-2.5 py-1 text-xs font-medium transition-colors",
                    b.result === r
                      ? r === "won" ? "border-safe bg-safe/15 text-safe-ink" : r === "lost" ? "border-danger bg-danger/10 text-danger-ink" : "border-ink bg-panel-2"
                      : "border-border text-muted hover:text-ink",
                  )}
                >
                  {r === "pending" ? "Unsettle" : RESULT_LABEL[r]}
                </button>
              ))}
              <button type="button" onClick={() => remove.mutate(b.id)} className="ml-auto text-xs text-muted hover:text-danger-ink">
                Delete
              </button>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
