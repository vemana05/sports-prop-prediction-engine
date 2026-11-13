"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { useNbaModelVsLines } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { LinesBucket, NbaGroup, NbaModelVsLines, NbaParlay, NbaSide, NbaTierSummary } from "@/lib/types";

const pct = (x: number | null | undefined, d = 1) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
const num = (x: number | null | undefined, d = 4) => (x == null ? "—" : x.toFixed(d));
const signed = (x: number | null | undefined, d = 1) => (x == null ? "—" : `${x > 0 ? "+" : ""}${x.toFixed(d)}`);

const STATUS_LABELS: Record<string, string> = {
  graded: "Graded",
  push: "Push",
  no_line: "No DraftKings/FanDuel line",
  no_features_date: "Logged before features were dated",
  features_not_before_game: "Features not dated before the game",
  not_finished: "Game not finished",
  no_box_score: "Waiting on box score",
  did_not_play: "Did not play (void)",
};

const RESULT_STYLE: Record<NbaParlay["result"], string> = {
  won: "text-safe-ink",
  lost: "text-danger-ink",
  pending: "text-muted",
  reduced: "text-risky-ink",
  void: "text-muted",
};

export default function NbaLinesPage() {
  const { data, error } = useNbaModelVsLines();
  return (
    <div className="mx-auto max-w-[1100px] px-4 pb-16 pt-8 lg:px-6">
      <h1 className="font-display text-2xl font-bold">NBA model vs. lines</h1>
      <p className="mt-1 text-sm text-muted">
        Points predictions logged before tip-off, graded against the sportsbook line and the box score, and the parlays the Builder suggested from them.
      </p>
      {error ? (
        <p className="mt-4 text-danger-ink">Couldn't load the report: {(error as Error).message}</p>
      ) : !data ? (
        <div className="mt-6 space-y-3">
          <Skeleton className="h-28 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : !data.available ? (
        <p className="mt-5 rounded-slip border border-border bg-panel p-5">
          No report yet. Build it with <code>python -m backend.backtest.nba_model_vs_lines</code>.
        </p>
      ) : (
        <Body r={data} />
      )}
    </div>
  );
}

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return <th scope="col" className={cn("px-4 py-2.5 font-medium", right && "text-right")}>{children}</th>;
}

function Td({ children, right, className, colSpan }: { children: React.ReactNode; right?: boolean; className?: string; colSpan?: number }) {
  return <td colSpan={colSpan} className={cn("px-4 py-2.5", right && "num text-right", className)}>{children}</td>;
}

function Table({ head, children, min = 560 }: { head: React.ReactNode; children: React.ReactNode; min?: number }) {
  return (
    <div className="mt-4 overflow-x-auto rounded-control border border-border bg-panel">
      <table className="w-full text-left text-sm" style={{ minWidth: min }}>
        <thead className="border-b border-border text-muted"><tr>{head}</tr></thead>
        <tbody className="divide-y divide-border">{children}</tbody>
      </table>
    </div>
  );
}

function Section({ id, title, note, children }: { id: string; title: string; note?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section aria-labelledby={id} className="mt-10">
      <h2 id={id} className="font-display text-xl font-bold">{title}</h2>
      {note && <p className="mt-1 max-w-prose text-sm text-muted">{note}</p>}
      {children}
    </section>
  );
}

function Empty({ cols, children = "No graded props yet." }: { cols: number; children?: React.ReactNode }) {
  return <tr><Td colSpan={cols} className="text-muted">{children}</Td></tr>;
}

function Calibration({ model, book }: { model: LinesBucket[]; book: LinesBucket[] }) {
  return (
    <Table head={<><Th>Predicted</Th><Th right>Model props</Th><Th right>Model said</Th><Th right>Hit</Th><Th right>Book props</Th><Th right>Book said</Th><Th right>Hit</Th></>}>
      {model.length === 0 && book.length === 0 ? <Empty cols={7} /> : Array.from({ length: 10 }, (_, i) => {
        const lo = i / 10;
        const m = model.find((b) => Math.abs(b.lo - lo) < 1e-6);
        const b = book.find((x) => Math.abs(x.lo - lo) < 1e-6);
        if (!m && !b) return null;
        return (
          <tr key={i}>
            <th scope="row" className="num px-4 py-2.5 font-medium">{(lo * 100).toFixed(0)}–{(lo * 100 + 10).toFixed(0)}%</th>
            <Td right>{m?.n ?? 0}</Td><Td right>{pct(m?.mean_p)}</Td><Td right>{pct(m?.hit_rate)}</Td>
            <Td right>{b?.n ?? 0}</Td><Td right>{pct(b?.mean_p)}</Td><Td right>{pct(b?.hit_rate)}</Td>
          </tr>
        );
      })}
    </Table>
  );
}

function SideLine({ label, s }: { label: string; s: NbaSide }) {
  return (
    <p className="mt-3 text-sm">
      <span className="font-semibold">{label}:</span> {s.n} props, hit {pct(s.hit_rate)}; model said {pct(s.mean_model)}, book {pct(s.mean_book)}.
      Calibration error: model {pct(s.ece_model)}, book {pct(s.ece_book)}.
    </p>
  );
}

function TierTable({ tiers }: { tiers: Record<string, NbaTierSummary> }) {
  return (
    <Table min={760} head={<><Th>Tier</Th><Th right>Suggested</Th><Th right>Settled</Th><Th right>Claimed</Th><Th right>After 0.30 shrink</Th><Th right>Book no-vig</Th><Th right>Actual</Th><Th right>Profit (1u each)</Th><Th right>Unflagged: claimed / actual</Th></>}>
      {Object.entries(tiers).map(([tier, t]) => (
        <tr key={tier}>
          <th scope="row" className="px-4 py-2.5 font-medium capitalize">{tier}</th>
          <Td right>{t.suggested}</Td>
          <Td right>{t.settled}{t.pending > 0 && <span className="text-muted"> · {t.pending} pending</span>}</Td>
          <Td right>{pct(t.claimed)}</Td>
          <Td right>{pct(t.claimed_sized)}</Td>
          <Td right>{pct(t.book)}</Td>
          <Td right className={cn(t.actual != null && t.claimed != null && (t.actual >= t.claimed ? "text-safe-ink" : "text-danger-ink"))}>{pct(t.actual)}</Td>
          <Td right>{t.profit == null ? "—" : signed(t.profit, 2)}</Td>
          <Td right>{t.clean_settled ? `${pct(t.clean_claimed)} / ${pct(t.clean_actual)} (${t.clean_settled})` : "—"}</Td>
        </tr>
      ))}
    </Table>
  );
}

function ParlayList({ parlays }: { parlays: NbaParlay[] }) {
  if (parlays.length === 0) return <p className="mt-3 text-sm text-muted">None.</p>;
  return (
    <Table min={760} head={<><Th>Day</Th><Th>Tier</Th><Th>Legs</Th><Th right>Claimed</Th><Th right>Odds</Th><Th right>Result</Th></>}>
      {parlays.map((p) => (
        <tr key={`${p.date}-${p.tier}-${p.rank}`} className="align-top">
          <Td className="whitespace-nowrap">{p.date}</Td>
          <Td className="capitalize">{p.tier} {String.fromCharCode(64 + p.rank)}{p.flagged && <span className="ml-1 text-xs text-risky-ink">check data</span>}</Td>
          <Td>
            <ul className="space-y-0.5">
              {p.legs.map((l) => (
                <li key={`${l.player_name}-${l.side}`}>
                  {l.player_name} {l.side} {l.line}{" "}
                  <span className={cn("text-xs", l.result === "hit" ? "text-safe-ink" : l.result === "miss" ? "text-danger-ink" : "text-muted")}>
                    {l.actual != null ? `${l.actual} pts · ` : ""}{l.result}
                  </span>
                </li>
              ))}
            </ul>
          </Td>
          <Td right>{pct(p.p_hit)}</Td>
          <Td right>{p.decimal_odds.toFixed(2)}</Td>
          <Td right className={cn("font-medium capitalize", RESULT_STYLE[p.result])}>{p.result}</Td>
        </tr>
      ))}
    </Table>
  );
}

function GroupRow({ label, g }: { label: string; g: NbaGroup }) {
  return (
    <tr>
      <th scope="row" className="px-4 py-2.5 font-medium">{label}</th>
      <Td right>{g.n}</Td>
      <Td right>{pct(g.model_side_hit_rate)}</Td>
      <Td right>{pct(g.over_rate)}</Td>
      <Td right>{num(g.brier_model)} / {num(g.brier_book)}</Td>
      <Td right>{signed(g.mean_error)}</Td>
      <Td right>{g.mae == null ? "—" : g.mae.toFixed(1)}</Td>
    </tr>
  );
}

function Body({ r }: { r: NbaModelVsLines }) {
  const o = r.overall!;
  const b = r.builder!;
  const rec = r.recommendation!;
  return (
    <>
      {r.box_scores_stale_days != null && (
        <p role="status" className="mt-5 rounded-slip border border-risky/40 bg-risky/10 p-4 text-sm text-risky-ink">
          The newest game log is from {r.box_scores_through}, {r.box_scores_stale_days} days ago. Recent games can't be graded until
          game logs load; check the daily job's game-log step (or it's a break in the schedule).
        </p>
      )}
      <section aria-label="Summary" className="mt-5 rounded-slip border border-border bg-panel p-5 sm:p-6">
        {!o.enough_sample && (
          <p className="mb-3 inline-flex rounded-full bg-risky/15 px-2.5 py-0.5 text-sm font-medium text-risky-ink">
            Sample under {r.min_sample}: too small to conclude anything ({o.graded} graded props)
          </p>
        )}
        <div className="max-w-prose space-y-3 text-base">
          {r.summary!.map((s) => <p key={s}>{s}</p>)}
          <p className="text-sm text-muted">
            {r.book_rule} {r.features_rule} Report as of {r.as_of}.
            {r.books_used && Object.keys(r.books_used).length > 0 &&
              <> Graded props by book: {Object.entries(r.books_used).map(([k, n]) => `${k} ${n}`).join(", ")}.</>}
          </p>
        </div>
      </section>

      <Section id="acc-heading" title="Accuracy vs. the book" note="Brier score is the mean squared error of the over probability (lower is better; 0.25 is a coin flip), and it's the same for overs and unders. Pushes are left out.">
        <Table head={<><Th>Graded</Th><Th right>Pushes</Th><Th right>Brier model</Th><Th right>Brier book</Th><Th right>Model (uncalibrated)</Th><Th right>Model side hit rate</Th></>}>
          <tr>
            <Td>{o.graded}</Td>
            <Td right>{o.pushes}</Td>
            <Td right className={cn(o.brier_model != null && o.brier_book != null && o.brier_model < o.brier_book && "text-safe-ink")}>{num(o.brier_model)}</Td>
            <Td right>{num(o.brier_book)}</Td>
            <Td right className="text-muted">{num(o.brier_model_raw)}</Td>
            <Td right>{pct(o.model_side_hit_rate)} <span className="text-muted">({o.model_side_n})</span></Td>
          </tr>
        </Table>
      </Section>

      <Section id="over-heading" title="Calibration: overs" note="Props grouped by the over probability each side gave: how often the over hit. Well calibrated means the said and hit columns match.">
        <SideLine label="Overs" s={o.over} />
        <Calibration model={o.over.calibration_model} book={o.over.calibration_book} />
      </Section>

      <Section id="under-heading" title="Calibration: unders" note="The same props from the under side.">
        <SideLine label="Unders" s={o.under} />
        <Calibration model={o.under.calibration_model} book={o.under.calibration_book} />
      </Section>

      <Section id="gap-heading" title="Hit rate by model-vs-book gap" note="How far the model's over probability was from the book's no-vig one, in percentage points. The model's side is over when it rates the over higher than the book, under otherwise. About 52.4% breaks even at -110.">
        <Table head={<><Th>Gap</Th><Th right>Props</Th><Th right>Model side hit</Th><Th right>Overs picked</Th><Th right>Over hit</Th><Th right>Unders picked</Th><Th right>Under hit</Th></>}>
          {o.gap.every((g) => g.n === 0) ? <Empty cols={7} /> : o.gap.map((g) => (
            <tr key={g.lo}>
              <th scope="row" className="num px-4 py-2.5 font-medium">{g.hi == null ? `${g.lo}+` : `${g.lo}–${g.hi}`} pts</th>
              <Td right>{g.n}</Td>
              <Td right>{pct(g.hit_rate)}</Td>
              <Td right>{g.n_over}</Td>
              <Td right>{pct(g.over_hit_rate)}</Td>
              <Td right>{g.n_under}</Td>
              <Td right>{pct(g.under_hit_rate)}</Td>
            </tr>
          ))}
        </Table>
      </Section>

      <Section id="stars-heading" title="Stars vs. everyone else" note={<>{r.star_rule} Error is the model's projection minus the actual points: positive means it projected too high.</>}>
        <Table min={680} head={<><Th>Players</Th><Th right>Props</Th><Th right>Model side hit</Th><Th right>Went over</Th><Th right>Brier model / book</Th><Th right>Avg. error</Th><Th right>Avg. miss</Th></>}>
          <GroupRow label="Stars (top 30)" g={o.stars} />
          <GroupRow label="Others" g={o.others} />
        </Table>
      </Section>

      <Section id="builder-heading" title="Builder parlays, replayed" note={<>{r.replay_rule} Claimed is the hit chance the Builder shows. Only parlays that settled won or lost count; ones with a push or a player who sat are listed but left out.</>}>
        <TierTable tiers={b.tiers} />
        {b.days.length > 0 && (
          <p className="mt-2 text-sm text-muted">
            {b.days.length} slate days, {b.days[0].date} to {b.days[b.days.length - 1].date}
            {b.days.some((d) => d.early_season) && " (early-season sanity limits on)"}.
          </p>
        )}
        <details className="mt-3">
          <summary className="cursor-pointer text-sm font-medium">Every replayed parlay ({b.parlays.length})</summary>
          <ParlayList parlays={b.parlays} />
        </details>
      </Section>

      <Section id="logged-heading" title="Builder parlays, logged live" note="Logged when the slate is built, from now on: each tier's default suggestions (both sides, nothing vetoed).">
        <TierTable tiers={b.logged.tiers} />
        <details className="mt-3">
          <summary className="cursor-pointer text-sm font-medium">Every logged parlay ({b.logged.parlays.length})</summary>
          <ParlayList parlays={b.logged.parlays} />
        </details>
      </Section>

      <Section id="fix-heading" title="Suggested fix (not applied)" note="Shrink the model's over probability toward the book's no-vig price, then cap how far it may sit from it: book + clip((1 − shrink) × (model − book), ±cap). Chosen by Brier score on the graded props above.">
        {!rec.available ? (
          <p className="mt-3 text-sm">{rec.reason}</p>
        ) : (
          <>
            <p className="mt-3 text-sm">
              Shrink <span className="num font-semibold">{(rec.shrink * 100).toFixed(0)}%</span> toward the book,{" "}
              {rec.cap == null ? "no cap" : <>cap the gap at <span className="num font-semibold">{(rec.cap * 100).toFixed(0)} points</span></>}.
              Brier {num(rec.brier_before)} → {num(rec.brier_after)} (book {num(rec.brier_book)}); calibration error {pct(rec.ece_before)} → {pct(rec.ece_after)} (book {pct(rec.ece_book)}).
              {!rec.enough_sample && <span className="text-risky-ink"> Fit on only {rec.n} props, so it may not hold.</span>}
            </p>
            {b.adjusted && (
              <>
                <p className="mt-4 text-sm font-medium">The replayed Builder with this fix applied to every leg:</p>
                <TierTable tiers={b.adjusted.tiers} />
                <details className="mt-3">
                  <summary className="cursor-pointer text-sm font-medium">Every adjusted parlay ({b.adjusted.parlays.length})</summary>
                  <ParlayList parlays={b.adjusted.parlays} />
                </details>
              </>
            )}
          </>
        )}
      </Section>

      <Section id="cov-heading" title="Coverage" note={r.other_markets}>
        {r.coverage && (
          <p className="mt-3 text-sm">
            Points predictions: {Object.entries(r.coverage).filter(([, v]) => v > 0).map(([k, v]) => `${STATUS_LABELS[k] ?? k} ${v}`).join(" · ")}.
            {r.box_scores_through && <> Box scores loaded through {r.box_scores_through}.</>}
          </p>
        )}
        {r.names && (
          <p className="mt-1 text-sm">
            Sportsbook names resolved: <span className="num font-semibold">{r.names.resolved} of {r.names.total}</span>
            {r.names.unresolved.length > 0 && <span className="text-muted"> — not found: {r.names.unresolved.join(", ")}</span>}
          </p>
        )}
      </Section>
    </>
  );
}
