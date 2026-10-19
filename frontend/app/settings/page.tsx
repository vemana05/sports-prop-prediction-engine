"use client";

import { useEffect, useState } from "react";
import { etTime } from "@/components/loss-limits";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useLimits, useSettings, useSettingsMutations } from "@/lib/api";
import { money, pct } from "@/lib/format";
import { useStored } from "@/lib/storage";
import { cn } from "@/lib/utils";
import type { LimitPeriod, LossLimit, TierName } from "@/lib/types";

const PERIOD_HINT: Record<LimitPeriod, string> = {
  daily: "Resets at midnight ET.",
  weekly: "Monday to Sunday, ET.",
  monthly: "Optional. Calendar month, ET.",
};
const TIERS: TierName[] = ["safe", "risky", "dangerous"];
const parseMoney = (s: string) => Number(s.replace(/[$,\s]/g, ""));

function Section({ id, title, children, className }: { id: string; title: string; children: React.ReactNode; className?: string }) {
  return (
    <section aria-labelledby={id} className={cn("rounded-control border border-border bg-panel p-4", className)}>
      <h2 id={id} className="font-display text-lg font-semibold">{title}</h2>
      {children}
    </section>
  );
}

export default function SettingsPage() {
  return (
    <div className="mx-auto max-w-[900px] space-y-5 px-4 pb-16 pt-8 lg:px-6">
      <h1 className="font-display text-2xl font-bold">Settings</h1>
      <LossLimits />
      <TakeABreak />
      <Bankroll />
      <Theme />
      <ApiKeys />
      <Advanced />
      <p className="text-sm text-muted">
        Gambling problem? Call <a href="tel:18004262537" className="font-semibold text-ink underline-offset-2 hover:underline">1-800-GAMBLER</a>.
      </p>
    </div>
  );
}

function LossLimits() {
  const { data, error } = useLimits();
  return (
    <Section id="limits-h" title="Loss limits">
      <p className="mt-1 text-sm text-muted">
        Counts real bets settled as lost, minus real winnings, in the window. Paper bets don't count. Bonus bets are tracked separately and
        left out, since losing one costs you nothing. You'll see a warning at {data ? Math.round(data.warn_at * 100) : 80}% and a stronger
        one at 100%; past that, logging a real bet asks you to confirm.
      </p>
      <p className="mt-1 text-sm text-muted">
        Lowering a limit takes effect now. Raising or removing one takes effect after {data?.raise_delay_hours ?? 24} hours.
      </p>
      {error ? (
        <p className="mt-3 text-danger-ink">Couldn't load limits: {(error as Error).message}</p>
      ) : !data ? (
        <Skeleton className="mt-3 h-40 w-full" />
      ) : (
        <div className="mt-4 divide-y divide-border">
          {data.limits.map((l) => <LimitRow key={l.period} l={l} />)}
        </div>
      )}
    </Section>
  );
}

function LimitRow({ l }: { l: LossLimit }) {
  const { setLimit, cancelPending } = useSettingsMutations();
  const [draft, setDraft] = useState(l.limit != null ? String(l.limit) : "");
  useEffect(() => setDraft(l.limit != null ? String(l.limit) : ""), [l.limit]);
  const amount = parseMoney(draft);
  const valid = draft.trim() !== "" && Number.isFinite(amount) && amount > 0;
  const raising = valid && l.limit != null && amount > l.limit;
  const inputId = `limit-${l.period}`;
  const err = (setLimit.error ?? cancelPending.error) as Error | null;

  return (
    <div className="py-3 first:pt-0 last:pb-0">
      <div className="flex flex-wrap items-end gap-3">
        <div className="min-w-[180px] flex-1">
          <label htmlFor={inputId} className="text-sm font-medium capitalize">{l.period} limit</label>
          <p className="text-xs text-muted">{PERIOD_HINT[l.period]}</p>
        </div>
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (valid) setLimit.mutate({ period: l.period, amount: Math.round(amount * 100) / 100 });
          }}
        >
          <div className="flex w-32 items-center rounded-control border border-border bg-surface focus-within:border-ink">
            <span className="pl-2 text-muted">$</span>
            <input
              id={inputId}
              inputMode="decimal"
              placeholder="No limit"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              className="num h-9 w-full bg-transparent px-1.5 outline-none"
            />
          </div>
          <Button type="submit" size="sm" variant="primary" disabled={!valid || amount === l.limit || setLimit.isPending}>
            {l.limit == null ? "Set" : raising ? "Raise in 24h" : "Save"}
          </Button>
          {l.limit != null && !(l.pending && l.pending.amount == null) && (
            <Button type="button" size="sm" variant="ghost" onClick={() => setLimit.mutate({ period: l.period, amount: null })}>
              Remove
            </Button>
          )}
        </form>
      </div>
      <p className="mt-1.5 text-sm">
        {l.limit == null ? (
          <span className="text-muted">No limit set.</span>
        ) : (
          <>
            <span className="num">{money(l.net_loss)}</span> of <span className="num">{money(l.limit)}</span> lost this {l.period === "daily" ? "day" : l.period.replace("ly", "")}
            {l.level === "hit" && <span className="ml-1 font-semibold text-danger-ink">· limit hit</span>}
            {l.level === "warn" && <span className="ml-1 font-semibold text-risky-ink">· {Math.round((l.used ?? 0) * 100)}% used</span>}
          </>
        )}
      </p>
      {l.pending && (
        <div className="mt-2 flex flex-wrap items-center gap-2 rounded-control border border-risky/50 bg-risky/10 px-3 py-2 text-sm">
          <span>
            Pending: {l.pending.amount == null ? "remove this limit" : `raise to ${money(l.pending.amount)}`} on {etTime(l.pending.effective_at)}.
          </span>
          <Button size="sm" variant="outline" className="ml-auto" onClick={() => cancelPending.mutate(l.period)} disabled={cancelPending.isPending}>
            Cancel change
          </Button>
        </div>
      )}
      {err && <p className="mt-1 text-xs text-danger-ink">{err.message}</p>}
    </div>
  );
}

function TakeABreak() {
  const { data } = useLimits();
  const { pause } = useSettingsMutations();
  const [choice, setChoice] = useState<number | null>(null);
  const label = (d: number) => (d === 1 ? "24 hours" : `${d} days`);
  return (
    <Section id="break-h" title="Take a break">
      <p className="mt-1 text-sm text-muted">
        Pause logging real bets. A break can't be ended early. Paper bets still work, and the rest of the app stays open.
      </p>
      {data?.paused_until ? (
        <p role="status" className="mt-3 rounded-control border-2 border-danger bg-danger/10 px-3 py-2 text-sm font-medium">
          Real-bet logging is paused until {etTime(data.paused_until)}.
        </p>
      ) : null}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {(data?.pause_days ?? [1, 7, 30]).map((d) => (
          <Button key={d} size="sm" variant={choice === d ? "primary" : "outline"} onClick={() => setChoice(d)}>
            {label(d)}
          </Button>
        ))}
      </div>
      {choice != null && (
        <div className="mt-3 rounded-control border border-border bg-panel-2 p-3 text-sm">
          <p>
            Pause real-bet logging for <span className="font-semibold">{label(choice)}</span>? You won't be able to undo this until it ends.
            {data?.paused_until && " If your current break runs longer, it stays as is."}
          </p>
          <div className="mt-2 flex gap-2">
            <Button size="sm" variant="primary" disabled={pause.isPending} onClick={() => pause.mutate(choice, { onSuccess: () => setChoice(null) })}>
              Start {label(choice)} break
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setChoice(null)}>Cancel</Button>
          </div>
        </div>
      )}
      <p className="mt-3 text-sm">
        Need to talk to someone? Call <a href="tel:18004262537" className="font-semibold underline-offset-2 hover:underline">1-800-GAMBLER</a>, free and confidential, 24/7.
      </p>
    </Section>
  );
}

function Bankroll() {
  const [bankroll, setBankroll] = useStored<number>("bankroll", 500, { parse: (r) => (Number(r) > 0 ? Number(r) : undefined) });
  const [draft, setDraft] = useState("");
  const [saved, setSaved] = useState(false);
  useEffect(() => setDraft(String(bankroll)), [bankroll]);
  const n = parseMoney(draft);
  const valid = Number.isFinite(n) && n > 0 && n <= 10_000_000;
  return (
    <Section id="bankroll-h" title="Bankroll">
      <p className="mt-1 text-sm text-muted">Typed in only, never connected to a sportsbook. Stakes are sized from it. Saved in this browser.</p>
      <form
        className="mt-3 flex items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (!valid) return;
          setBankroll(Math.round(n * 100) / 100);
          setSaved(true);
        }}
      >
        <label htmlFor="settings-bankroll" className="sr-only">Bankroll</label>
        <div className="flex w-40 items-center rounded-control border border-border bg-surface focus-within:border-ink">
          <span className="pl-2 text-muted">$</span>
          <input id="settings-bankroll" inputMode="decimal" value={draft} onChange={(e) => { setDraft(e.target.value); setSaved(false); }} className="num h-9 w-full bg-transparent px-1.5 outline-none" aria-invalid={!valid} />
        </div>
        <Button type="submit" size="sm" variant="primary" disabled={!valid}>Save</Button>
        {saved && <span role="status" className="text-sm text-muted">Saved.</span>}
      </form>
    </Section>
  );
}

function Theme() {
  const [dark, setDark] = useState(false);
  useEffect(() => setDark(document.documentElement.classList.contains("dark")), []);
  const choose = (d: boolean) => {
    document.documentElement.classList.toggle("dark", d);
    try {
      localStorage.setItem("theme", d ? "dark" : "light");
    } catch {}
    setDark(d);
  };
  return (
    <Section id="theme-h" title="Theme">
      <div className="mt-3 flex gap-2" role="group" aria-label="Theme">
        <Button size="sm" variant={!dark ? "primary" : "outline"} aria-pressed={!dark} onClick={() => choose(false)}>Light</Button>
        <Button size="sm" variant={dark ? "primary" : "outline"} aria-pressed={dark} onClick={() => choose(true)}>Dark</Button>
      </div>
    </Section>
  );
}

function ApiKeys() {
  const { data } = useSettings();
  return (
    <Section id="keys-h" title="API keys">
      <p className="mt-1 text-sm text-muted">Set in <code>.env</code>. Only whether each key is set is shown here, never its value.</p>
      {!data ? (
        <Skeleton className="mt-3 h-24 w-full" />
      ) : (
        <ul className="mt-3 divide-y divide-border text-sm">
          {data.api_keys.map((k) => (
            <li key={k.name} className="flex items-center justify-between gap-3 py-2">
              <span>
                {k.label} <code className="ml-1 text-xs text-muted">{k.name}</code>
              </span>
              <span className={cn("rounded-full px-2 py-0.5 text-xs font-medium", k.status === "set" ? "bg-safe/15 text-safe-ink" : "bg-danger/15 text-danger-ink")}>
                {k.status === "set" ? "Set" : "Missing"}
              </span>
            </li>
          ))}
          {data.llm_features.map((f) => (
            <li key={f.name} className="flex items-center justify-between gap-3 py-2">
              <span>
                {f.label} <code className="ml-1 text-xs text-muted">{f.name}</code>
              </span>
              <span className="text-xs text-muted">{f.enabled ? "On" : "Off"}</span>
            </li>
          ))}
        </ul>
      )}
    </Section>
  );
}

function Advanced() {
  const { data } = useSettings();
  const { setKelly } = useSettingsMutations();
  const [draft, setDraft] = useState("");
  useEffect(() => {
    if (data) setDraft(String(data.kelly_fraction));
  }, [data?.kelly_fraction]); // eslint-disable-line react-hooks/exhaustive-deps
  const k = Number(draft);
  const valid = Number.isFinite(k) && k > 0 && k <= 1;
  return (
    <Section id="advanced-h" title="Advanced">
      <details className="mt-2">
        <summary className="cursor-pointer text-sm font-medium">Kelly fraction and tier thresholds</summary>
        <div className="mt-3 space-y-5">
          <div>
            <label htmlFor="kelly" className="text-sm font-medium">Kelly fraction</label>
            <p className="text-xs text-muted">
              Share of the full Kelly stake to bet. Default {data?.kelly_default ?? 0.25} (quarter Kelly). Stakes stay capped by each tier's max.
            </p>
            <form
              className="mt-2 flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (valid) setKelly.mutate(k);
              }}
            >
              <input id="kelly" inputMode="decimal" value={draft} onChange={(e) => setDraft(e.target.value)} className="num h-9 w-24 rounded-control border border-border bg-surface px-2 outline-none focus:border-ink" aria-invalid={!valid} />
              <Button type="submit" size="sm" variant="primary" disabled={!valid || k === data?.kelly_fraction || setKelly.isPending}>Save</Button>
              {data && data.kelly_fraction !== data.kelly_default && (
                <Button type="button" size="sm" variant="ghost" onClick={() => setKelly.mutate(data.kelly_default)}>Reset to {data.kelly_default}</Button>
              )}
            </form>
            {valid && k > (data?.kelly_default ?? 0.25) && (
              <p role="alert" className="mt-2 rounded-control border border-risky/50 bg-risky/10 px-3 py-2 text-sm">
                <span className="font-semibold text-risky-ink">Above quarter Kelly.</span> Model probabilities have error, and bigger fractions turn
                that error into much bigger swings. 0.25 or less is recommended.
              </p>
            )}
            {setKelly.error && <p className="mt-1 text-xs text-danger-ink">{(setKelly.error as Error).message}</p>}
          </div>

          <div>
            <h3 className="text-sm font-medium">Tier thresholds</h3>
            <p className="text-xs text-muted">Read from <code>config/tiers.yaml</code>. Edit that file to change them.</p>
            {!data ? (
              <Skeleton className="mt-2 h-24 w-full" />
            ) : (
              <div className="mt-2 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-xs text-muted">
                    <tr>
                      <th className="py-1 pr-3 font-medium">Tier</th>
                      <th className="py-1 pr-3 font-medium">Legs</th>
                      <th className="py-1 pr-3 font-medium">Min leg chance</th>
                      <th className="py-1 pr-3 font-medium">Parlay chance</th>
                      <th className="py-1 font-medium">Max stake</th>
                    </tr>
                  </thead>
                  <tbody className="num">
                    {TIERS.filter((t) => data.tiers[t]).map((t) => {
                      const c = data.tiers[t];
                      return (
                        <tr key={t} className="border-t border-border">
                          <td className="py-1.5 pr-3 font-sans capitalize">{t}</td>
                          <td className="py-1.5 pr-3">{c.legs_min === c.legs_max ? c.legs_min : `${c.legs_min}–${c.legs_max}`}</td>
                          <td className="py-1.5 pr-3">{pct(c.min_leg_prob)}</td>
                          <td className="py-1.5 pr-3">{pct(c.target_parlay_prob_min)}–{pct(Math.min(c.target_parlay_prob_max, 1))}</td>
                          <td className="py-1.5">{pct(c.max_stake_pct_bankroll)} of bankroll</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </details>
    </Section>
  );
}
