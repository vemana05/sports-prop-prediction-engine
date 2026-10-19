"use client";

import { useState } from "react";
import Link from "next/link";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { LegRow } from "./leg-row";
import { TierBadge } from "./tier-badge";
import { LimitBanner, limitNames } from "./loss-limits";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useBetMutations, useLimits } from "@/lib/api";
import { american, evPerDollar, money, pct } from "@/lib/format";
import { usePaperMode } from "@/lib/storage";
import type { League, Leg, LegPrice, Parlay, TierName } from "@/lib/types";

const LEG_STAGGER = 0.08;

export interface SlipNotice {
  tone: "info" | "warn";
  text: string;
  onUndo?: () => void;
}

/**
 * The bet slip (spec 10.2): a tall ticket card with a perforated tear line,
 * stacked legs, and the totals printed at the bottom. Signature motion
 * (10.5): legs drop in one after another, then the totals "print".
 * Reduced motion is handled by <MotionConfig reducedMotion="user">.
 *
 * `onLegAction` puts a button on every leg: "Veto" for suggested parlays
 * (swap in a replacement), "Remove" in manual mode.
 */
export function BetSlip({
  parlay,
  tier,
  league,
  legAction,
  onLegAction,
  busyLegId,
  notice,
  emptyText = "Pick a suggested parlay to load it here.",
  prices,
  onPrice,
}: {
  parlay: Parlay | null;
  tier: TierName;
  league: League;
  legAction?: "veto" | "remove";
  onLegAction?: (leg: Leg) => void;
  busyLegId?: string | null;
  notice?: SlipNotice | null;
  emptyText?: string;
  /** NFL: typed book prices keyed by base leg id, and the setter. */
  prices?: Record<string, LegPrice>;
  onPrice?: (baseId: string, price: LegPrice | null) => void;
}) {
  return (
    <section aria-label="Bet slip" className="drop-shadow-none">
      <div className="rounded-t-slip bg-panel shadow-slip">
        <header className="flex items-center justify-between px-5 pb-3 pt-4">
          <h2 className="font-display text-xl font-semibold">{parlay?.name ?? "Bet slip"}</h2>
          <TierBadge tier={tier} />
        </header>

        {notice && (
          <div
            role="status"
            className={cn(
              "mx-5 mb-3 flex items-start gap-3 rounded-control border px-3 py-2 text-xs",
              notice.tone === "warn" ? "border-risky/60 bg-risky/10" : "border-border bg-panel-2",
            )}
          >
            <p className="flex-1">{notice.text}</p>
            {notice.onUndo && (
              <button type="button" onClick={notice.onUndo} className="shrink-0 font-medium text-accent-ink underline-offset-2 hover:underline">
                Undo
              </button>
            )}
          </div>
        )}

        {!parlay ? (
          <p className="px-5 pb-8 pt-2 text-sm text-muted">{emptyText}</p>
        ) : (
          <AnimatePresence mode="wait">
            <motion.div key={parlay.name} initial={{ opacity: 1 }} exit={{ opacity: 0, transition: { duration: 0.12 } }}>
              <ol className="space-y-4 px-5 pb-4">
                <AnimatePresence initial={true}>
                  {parlay.legs.map((leg, i) => (
                    <motion.li
                      key={leg.id}
                      layout
                      initial={{ opacity: 0, y: -14 }}
                      animate={{ opacity: busyLegId === leg.id ? 0.45 : 1, y: 0 }}
                      exit={{ opacity: 0, x: 24, transition: { duration: 0.18 } }}
                      transition={{ delay: i * LEG_STAGGER, duration: 0.22, ease: "easeOut" }}
                      className={cn(i > 0 && "border-t border-border pt-4")}
                    >
                      <LegRow leg={leg} league={league} />
                      {onPrice && <PriceInputs leg={leg} price={prices?.[leg.base_id ?? leg.id]} onChange={(p) => onPrice(leg.base_id ?? leg.id, p)} />}
                      {onLegAction && (
                        <div className="mt-1.5 flex justify-end">
                          <button
                            type="button"
                            onClick={() => onLegAction(leg)}
                            disabled={!!busyLegId}
                            aria-label={`${legAction === "remove" ? "Remove" : "Veto"} ${leg.player_name} ${leg.label}`}
                            className="inline-flex items-center gap-1 rounded-control px-2 py-1 text-xs font-medium text-muted hover:bg-panel-2 hover:text-ink disabled:opacity-50"
                          >
                            <X className="h-3.5 w-3.5" aria-hidden />
                            {busyLegId === leg.id ? "Finding a replacement…" : legAction === "remove" ? "Remove" : "Veto"}
                          </button>
                        </div>
                      )}
                    </motion.li>
                  ))}
                </AnimatePresence>
              </ol>

              <div className="perforation" aria-hidden />

              <motion.div
                initial={{ clipPath: "inset(0 0 100% 0)" }}
                animate={{ clipPath: "inset(0 0 0% 0)" }}
                transition={{ delay: parlay.legs.length * LEG_STAGGER + 0.05, duration: 0.25, ease: "linear" }}
              >
                <SlipTotals parlay={parlay} />
                <LogBet parlay={parlay} tier={tier} league={league} />
              </motion.div>
            </motion.div>
          </AnimatePresence>
        )}
      </div>
      <div className="tear-edge" aria-hidden />
    </section>
  );
}

function SlipTotals({ parlay }: { parlay: Parlay }) {
  const correlationShift = parlay.p_hit - parlay.independent_p_hit;
  const unpriced = parlay.american_odds == null;
  return (
    <div className="px-5 pb-3 pt-2">
      <div className="flex items-end justify-between">
        <div>
          <p className="text-xs text-muted">Model hit chance</p>
          <p className="num text-2xl font-bold leading-none">{pct(parlay.p_hit)}</p>
        </div>
        {unpriced ? (
          parlay.fair_american_odds != null && (
            <div className="text-right">
              <p className="text-xs text-muted">Fair odds (model)</p>
              <p className="num text-xl font-semibold leading-none text-muted">{american(parlay.fair_american_odds)}</p>
            </div>
          )
        ) : (
          <div className="text-right">
            <p className="text-xs text-muted">Book no-vig</p>
            <p className="num text-xl font-semibold leading-none text-muted">{pct(parlay.no_vig_p ?? parlay.book_p)}</p>
          </div>
        )}
      </div>
      {parlay.legs.length > 1 && (
        <p className="mt-1.5 text-xs text-muted">
          Per leg: {parlay.legs.map((l) => `${l.player_name.split(" ").slice(-1)[0]} ${pct(l.model_p)}`).join(" · ")}
        </p>
      )}
      {Math.abs(correlationShift) >= 0.005 && (
        <p className="mt-1 text-xs text-muted">
          {pct(parlay.independent_p_hit)} if the legs were independent; correlation {correlationShift > 0 ? "adds" : "removes"}{" "}
          {pct(Math.abs(correlationShift), 1)}.
        </p>
      )}
      {parlay.no_vig_p != null && (
        <p className="mt-1 text-xs text-muted">
          Stake and EV use <span className="num font-semibold text-ink">{pct(parlay.p_sized)}</span> — the model pulled 30% toward the book.
        </p>
      )}
      {!parlay.in_tier_band && (
        <p className="mt-1 text-xs text-risky-ink">This hit chance is outside the selected tier's range.</p>
      )}
      <SlipNotes parlay={parlay} />

      {unpriced ? (
        <p className="mt-4 border-t border-dashed border-border pt-3 text-sm">
          <span className="font-medium">Hit chance only.</span>{" "}
          <span className="text-muted">
            Type the book's odds on {parlay.n_priced ? "the remaining" : "every"} leg
            {parlay.n_priced ? ` (${parlay.n_priced} of ${parlay.legs.length} priced)` : ""} to see combined odds, EV and a stake.
          </span>
        </p>
      ) : (
      <dl className="mt-4 grid grid-cols-3 gap-2 border-t border-dashed border-border pt-3 text-sm">
        <div>
          <dt className="text-xs text-muted">Odds</dt>
          <dd className="num text-lg font-semibold">{american(parlay.american_odds!)}</dd>
        </div>
        <div>
          <dt className="text-xs text-muted">EV per $1</dt>
          {parlay.flagged ? (
            <dd className="text-lg font-medium text-muted" title="This parlay tripped the data check, so its EV can't be trusted.">
              unreliable
            </dd>
          ) : (
            <dd className="num text-lg font-semibold">{evPerDollar(parlay.ev_per_dollar ?? 0).replace(" per $1", "")}</dd>
          )}
        </div>
        <div className="text-right">
          <dt className="text-xs text-muted">Suggested</dt>
          <dd className="num text-lg font-semibold">{money(parlay.stake)}</dd>
        </div>
      </dl>
      )}
      {!unpriced && parlay.priced && parlay.legs.some((l) => !l.vig_removed) && (
        <p className="mt-2 text-xs text-muted">
          Legs with only an over price shrink toward the book's implied probability with the vig still in. Add the under price to remove it.
        </p>
      )}

      {unpriced ? null : parlay.stake > 0 ? (
        <p className="mt-2 text-xs text-muted">
          Pays {money(parlay.payout)} if it hits{parlay.capped_by_tier ? " · capped at this tier's max stake" : ""}.
        </p>
      ) : parlay.flagged ? (
        <div className="mt-3 rounded-control border border-risky/60 bg-risky/10 p-3 text-xs">
          <p className="font-medium text-risky-ink">Not recommended — check the data</p>
          <p className="mt-1 text-ink">
            The model and the book disagree by more than we trust, which usually means a data problem upstream, not a real edge.
          </p>
          <ul className="mt-1.5 list-disc space-y-0.5 pl-4 text-muted">
            {parlay.flag_reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
      ) : (
        parlay.reason && <p className="mt-3 text-xs text-muted">{parlay.reason}</p>
      )}
    </div>
  );
}

function LogBet({ parlay, tier, league }: { parlay: Parlay; tier: TierName; league: League }) {
  const [paper, setPaper] = usePaperMode();
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [logged, setLogged] = useState<string | null>(null);
  const [bonus, setBonus] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const { log } = useBetMutations();
  const { data: limitStatus } = useLimits();
  const stake = Number(draft.replace(/[$,\s]/g, ""));
  const valid = Number.isFinite(stake) && stake > 0;
  // An unpriced NFL parlay can only be paper-traded, at the model's fair odds.
  const unpriced = parlay.decimal_odds == null;
  const fair = (p: number) => Math.round(p >= 0.5 ? -100 * p / (1 - p) : 100 * (1 - p) / p);
  const asPaper = paper || unpriced;
  // Real bets only: a break blocks them; a hit limit needs an explicit, named confirmation. Paper is never blocked.
  const pausedUntil = asPaper ? null : limitStatus?.paused_until ?? null;
  const hitLimits = asPaper ? [] : limitStatus?.hit ?? [];
  const needsConfirm = hitLimits.length > 0 && !confirmed;

  if (logged === parlay.id)
    return (
      <p role="status" className="border-t border-border px-5 py-3 text-sm">
        <span className="font-medium">{asPaper ? "Paper bet logged." : "Bet logged."}</span>{" "}
        <Link href="/tracker" className="text-accent-ink underline-offset-2 hover:underline">Open tracker</Link>
      </p>
    );

  if (!open)
    return (
      <div className="border-t border-border px-5 py-3">
        <Button
          variant="outline"
          size="sm"
          className="w-full"
          onClick={() => {
            setDraft(parlay.stake > 0 ? String(parlay.stake) : "");
            setConfirmed(false);
            setOpen(true);
          }}
        >
          {asPaper ? "Log paper bet" : "Log bet"}
        </Button>
      </div>
    );

  return (
    <form
      className="space-y-2 border-t border-border px-5 py-3"
      onSubmit={(e) => {
        e.preventDefault();
        if (!valid || pausedUntil || needsConfirm) return;
        log.mutate(
          {
            league,
            tier,
            name: unpriced ? `${parlay.name} (fair odds, no book price)` : parlay.name,
            legs: parlay.legs.map((l) => ({ player_name: l.player_name, label: l.label, american_odds: l.american_odds ?? fair(l.model_p) })),
            decimal_odds: parlay.decimal_odds ?? 1 / parlay.p_hit,
            american_odds: parlay.american_odds ?? parlay.fair_american_odds ?? fair(parlay.p_hit),
            model_p: parlay.p_hit,
            stake,
            paper: asPaper,
            bonus: !asPaper && bonus,
            confirm_over_limit: !asPaper && confirmed,
          },
          { onSuccess: () => { setLogged(parlay.id); setOpen(false); } },
        );
      }}
    >
      <div className="flex items-center gap-2">
        <label htmlFor="log-stake" className="text-sm text-muted">Stake</label>
        <div className="flex flex-1 items-center rounded-control border border-border bg-surface focus-within:border-ink">
          <span className="pl-2 text-muted">$</span>
          <input
            id="log-stake"
            inputMode="decimal"
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            className="num h-8 w-full bg-transparent px-1.5 outline-none"
            aria-invalid={!valid}
          />
        </div>
      </div>
      <label className="flex items-center gap-2 text-xs">
        <input type="checkbox" checked={asPaper} disabled={unpriced} onChange={(e) => setPaper(e.target.checked)} className="h-4 w-4 accent-[var(--accent)]" />
        Paper trade (no real money)
      </label>
      {!asPaper && (
        <label className="flex items-center gap-2 text-xs">
          <input type="checkbox" checked={bonus} onChange={(e) => setBonus(e.target.checked)} className="h-4 w-4 accent-[var(--accent)]" />
          Bonus bet (the book's credit; left out of loss limits)
        </label>
      )}
      {!asPaper && <LimitBanner status={limitStatus} compact />}
      {hitLimits.length > 0 && !pausedUntil && (
        <label className="flex items-start gap-2 rounded-control border border-danger/60 bg-danger/5 p-2 text-xs">
          <input type="checkbox" checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} className="mt-0.5 h-4 w-4 accent-[var(--tier-danger)]" />
          <span>I know I've hit my {limitNames(hitLimits)} and want to log this real bet anyway.</span>
        </label>
      )}
      {unpriced && <p className="text-xs text-muted">No book odds typed, so this logs as a paper bet at the model's fair odds.</p>}
      {parlay.stake === 0 && <p className="text-xs text-muted">The model suggests $0 here. Logging it anyway is your call.</p>}
      {log.error && <p className="text-xs text-danger-ink">{(log.error as Error).message}</p>}
      <div className="flex justify-end gap-2">
        <Button type="button" variant="ghost" size="sm" onClick={() => setOpen(false)}>Cancel</Button>
        <Button type="submit" variant="primary" size="sm" disabled={!valid || log.isPending || !!pausedUntil || needsConfirm}>
          {asPaper ? "Log paper bet" : "Log bet"}
        </Button>
      </div>
    </form>
  );
}

function SlipNotes({ parlay }: { parlay: Parlay }) {
  const items = [...(parlay.notes ?? []).map((t) => ({ t, warn: false })), ...(parlay.warnings ?? []).map((t) => ({ t, warn: true }))];
  if (!items.length) return null;
  return (
    <ul className="mt-2 space-y-1.5">
      {items.map(({ t, warn }) => (
        <li key={t} className={cn("rounded-control border px-2.5 py-1.5 text-xs", warn ? "border-risky/50 bg-risky/10" : "border-border bg-panel-2")}>
          {t}
        </li>
      ))}
    </ul>
  );
}

const parseAmerican = (s: string): number | undefined => {
  const v = Number(s.trim().replace(/^\+/, ""));
  return Number.isFinite(v) && Math.abs(v) >= 100 ? v : undefined;
};

/** NFL: type the book's line and American odds for one leg. Applied on blur / Enter. */
function PriceInputs({ leg, price, onChange }: { leg: Leg; price?: LegPrice; onChange: (p: LegPrice | null) => void }) {
  const [line, setLine] = useState(price?.line ?? "");
  const [over, setOver] = useState(price?.over != null ? String(price.over) : "");
  const [under, setUnder] = useState(price?.under != null ? String(price.under) : "");
  const bad = (s: string) => s.trim() !== "" && parseAmerican(s) === undefined;
  const commit = () => {
    if (bad(over) || bad(under)) return;
    const next: LegPrice = {};
    if (line.trim()) next.line = line.trim();
    if (over.trim()) next.over = parseAmerican(over);
    if (under.trim()) next.under = parseAmerican(under);
    onChange(Object.keys(next).length ? next : null);
  };
  const field = "num h-7 w-full rounded-control border border-border bg-surface px-1.5 text-xs outline-none placeholder:text-muted/40 focus:border-ink aria-[invalid=true]:border-danger";
  return (
    <form
      className="mt-2 grid grid-cols-3 gap-1.5"
      onSubmit={(e) => { e.preventDefault(); commit(); }}
      aria-label={`Book price for ${leg.player_name} ${leg.label}`}
    >
      <label className="text-[11px] text-muted">
        Book line
        <input className={field} value={line} placeholder={leg.rung != null ? `${leg.rung}+` : String(leg.line)} onChange={(e) => setLine(e.target.value)} onBlur={commit} />
      </label>
      <label className="text-[11px] text-muted">
        Over odds
        <input className={field} value={over} placeholder="-150" inputMode="numeric" aria-invalid={bad(over)} onChange={(e) => setOver(e.target.value)} onBlur={commit} />
      </label>
      <label className="text-[11px] text-muted">
        Under (opt.)
        <input className={field} value={under} placeholder="+120" inputMode="numeric" aria-invalid={bad(under)} onChange={(e) => setUnder(e.target.value)} onBlur={commit} />
      </label>
      <button type="submit" className="sr-only">Apply</button>
    </form>
  );
}
