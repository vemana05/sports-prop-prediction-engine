"use client";

import Link from "next/link";
import { useLimits } from "@/lib/api";
import { money } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { LimitPeriod, LimitStatus, LossLimit } from "@/lib/types";

export const PERIOD_LABEL: Record<LimitPeriod, string> = { daily: "daily", weekly: "weekly", monthly: "monthly" };
export const limitNames = (periods: LimitPeriod[]) => periods.map((p) => `${PERIOD_LABEL[p]} loss limit`).join(" and ");

export const etTime = (iso: string) =>
  new Date(iso).toLocaleString("en-US", { timeZone: "America/New_York", weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) + " ET";

/** Builder and slip: a warning at 80% of any limit, a stronger one at 100%, and the break if one is running. */
export function LimitBanner({ status, className, compact }: { status?: LimitStatus; className?: string; compact?: boolean }) {
  const { data: fetched } = useLimits();
  const s = status ?? fetched;
  if (!s) return null;
  if (s.paused_until)
    return (
      <div role="alert" className={cn("rounded-control border-2 border-danger bg-danger/10 px-3 py-2 text-sm", className)}>
        <span className="font-semibold text-danger-ink">You're on a break until {etTime(s.paused_until)}.</span> Real bets can't be logged
        until then; paper bets still can.
      </div>
    );
  if (s.level === "ok") return null;
  const hit = s.limits.filter((l) => l.level === "hit");
  const warn = s.limits.filter((l) => l.level === "warn");
  if (hit.length)
    return (
      <div role="alert" className={cn("rounded-control border-2 border-danger bg-danger/10 px-3 py-2 text-sm", className)}>
        <p className="font-semibold text-danger-ink">
          You've hit your {limitNames(hit.map((l) => l.period))}
          {hit.map((l) => ` (${money(l.net_loss)} of ${money(l.limit!)})`).join("")}.
        </p>
        {!compact && (
          <p className="mt-0.5 text-muted">
            Consider stopping here. Logging another real bet asks you to confirm first.{" "}
            <Link href="/settings" className="text-accent-ink underline-offset-2 hover:underline">Limits and breaks</Link> · 1-800-GAMBLER
          </p>
        )}
      </div>
    );
  return (
    <div role="note" className={cn("rounded-control border border-risky/50 bg-risky/10 px-3 py-2 text-sm", className)}>
      <span className="font-semibold text-risky-ink">Close to your {limitNames(warn.map((l) => l.period))}: </span>
      {warn.map((l) => `${money(l.net_loss)} of ${money(l.limit!)} (${Math.round((l.used ?? 0) * 100)}%)`).join(", ")}.
    </div>
  );
}

function Bar({ l, warnAt }: { l: LossLimit; warnAt: number }) {
  const used = Math.min(l.used ?? 0, 1);
  const tone = l.level === "hit" ? "bg-danger" : l.level === "warn" ? "bg-risky" : "bg-safe";
  return (
    <div>
      <div className="flex items-baseline justify-between text-sm">
        <span className="font-medium capitalize">{l.period}</span>
        <span className="num text-muted">
          {money(l.net_loss)} of {money(l.limit!)} lost
        </span>
      </div>
      <div
        role="progressbar"
        aria-label={`${l.period} loss limit used`}
        aria-valuemin={0}
        aria-valuemax={l.limit!}
        aria-valuenow={Math.min(l.net_loss, l.limit!)}
        className="relative mt-1 h-2.5 overflow-hidden rounded-full bg-panel-2"
      >
        <div className={cn("h-full rounded-full", tone)} style={{ width: `${used * 100}%` }} />
        <div aria-hidden className="absolute inset-y-0 w-px bg-ink/40" style={{ left: `${warnAt * 100}%` }} />
      </div>
      {l.pending && (
        <p className="mt-0.5 text-xs text-muted">
          Changes to {l.pending.amount == null ? "no limit" : money(l.pending.amount)} on {etTime(l.pending.effective_at)}.
        </p>
      )}
    </div>
  );
}

/** Bet tracker: one bar per set limit; real money only. */
export function LimitBars() {
  const { data } = useLimits();
  if (!data) return null;
  const set = data.limits.filter((l) => l.limit != null);
  const bonus = data.limits.find((l) => l.period === "weekly");
  return (
    <section aria-labelledby="limits-heading" className="mt-6 rounded-control border border-border bg-panel p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="limits-heading" className="font-display text-lg font-semibold">Loss limits</h2>
        <Link href="/settings" className="text-sm text-accent-ink underline-offset-2 hover:underline">{set.length ? "Change limits" : "Set limits"}</Link>
      </div>
      {set.length === 0 ? (
        <p className="mt-2 text-sm text-muted">No loss limits set. Daily and weekly limits help you stop before a bad run gets expensive.</p>
      ) : (
        <div className="mt-3 space-y-3">
          {set.map((l) => <Bar key={l.period} l={l} warnAt={data.warn_at} />)}
        </div>
      )}
      <p className="mt-3 text-xs text-muted">
        Real bets settled as lost, minus real winnings, in the ET day, week (from Monday) or month. Paper bets don't count. Bonus bets are left out
        because losing one costs you nothing{bonus && bonus.bonus_bets ? ` (${bonus.bonus_bets} settled this week, ${money(bonus.bonus_winnings)} won)` : ""}.
      </p>
      {data.paused_until && <LimitBanner status={data} className="mt-3" />}
    </section>
  );
}
