"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { useModelVsLines } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { LinesSummary, ModelVsLines } from "@/lib/types";

const pct = (x: number | null | undefined, d = 1) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
const num = (x: number | null | undefined, d = 4) => (x == null ? "—" : x.toFixed(d));

const STATUS_LABELS: Record<string, string> = {
  graded: "Graded",
  unmatched_name: "Name not matched",
  no_projection: "No projection logged",
  game_not_final: "Game not final",
  did_not_play: "Did not play (void)",
  no_spread: "Model version missing",
};

export default function LinesPage() {
  const { data, error } = useModelVsLines();
  return (
    <div className="mx-auto max-w-[1100px] px-4 pb-16 pt-8 lg:px-6">
      <h1 className="font-display text-2xl font-bold">Model vs. lines</h1>
      <p className="mt-1 text-sm text-muted">NFL projections logged before kickoff, checked against real closing prop lines and results.</p>
      {error ? (
        <p className="mt-4 text-danger-ink">Couldn't load the report: {(error as Error).message}</p>
      ) : !data ? (
        <div className="mt-6 space-y-3">
          <Skeleton className="h-28 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : !data.available ? (
        <p className="mt-5 rounded-slip border border-border bg-panel p-5">
          No report yet. It's built by the Tuesday results job, after closing lines arrive.
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

function Td({ children, right, className }: { children: React.ReactNode; right?: boolean; className?: string }) {
  return <td className={cn("px-4 py-2.5", right && "num text-right", className)}>{children}</td>;
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

function Body({ r }: { r: ModelVsLines }) {
  const o = r.overall!;
  const rows: [string, LinesSummary][] = [...Object.values(r.markets!).map((m) => [m.label, m] as [string, LinesSummary]), ["All markets", o]];
  const names = r.names!;
  return (
    <>
      <section aria-label="Summary" className="mt-5 rounded-slip border border-border bg-panel p-5 sm:p-6">
        {!o.enough_sample && (
          <p className="mb-3 inline-flex rounded-full bg-risky/15 px-2.5 py-0.5 text-sm font-medium text-risky-ink">
            Sample too small to conclude anything ({o.graded} of {r.min_sample} props)
          </p>
        )}
        <div className="max-w-prose space-y-3 text-base">
          {r.summary!.map((s) => <p key={s}>{s}</p>)}
          <p className="text-sm text-muted">
            {r.book_rule} Report as of {r.as_of}.
            {r.books_used && Object.keys(r.books_used).length > 0 &&
              <> Graded props by book: {Object.entries(r.books_used).map(([b, n]) => `${b} ${n}`).join(", ")}.</>}
          </p>
        </div>
      </section>

      <Section id="acc-heading" title="Accuracy vs. the book" note="Brier score is the mean squared error of the over probability (lower is better; 0.25 is a coin flip). Pushes are left out. The model's side is over when it gives the over a higher chance than the book's no-vig price, under otherwise.">
        <Table min={720} head={<><Th>Market</Th><Th right>Graded</Th><Th right>Pushes</Th><Th right>Brier model</Th><Th right>Brier book</Th><Th right>Model (uncalibrated)</Th><Th right>Model side hit rate</Th></>}>
          {rows.map(([label, m]) => (
            <tr key={label} className={cn(label === "All markets" && "font-semibold")}>
              <th scope="row" className="px-4 py-2.5 font-medium">
                {label}
                {!m.enough_sample && m.graded > 0 && <span className="ml-2 text-xs font-normal text-risky-ink">small sample</span>}
              </th>
              <Td right>{m.graded}</Td>
              <Td right>{m.pushes}</Td>
              <Td right className={cn(m.brier_model != null && m.brier_book != null && m.brier_model < m.brier_book && "text-safe-ink")}>{num(m.brier_model)}</Td>
              <Td right>{num(m.brier_book)}</Td>
              <Td right className="text-muted">{num(m.brier_model_raw)}</Td>
              <Td right>{pct(m.side_hit_rate)} <span className="text-muted">({m.side_n})</span></Td>
            </tr>
          ))}
        </Table>
      </Section>

      <Section id="roi-heading" title="Flat-stake ROI at the closing price" note="One unit on the model's side for every prop where the model and the book's no-vig probability differ by at least the margin. Pushes return the stake. Closing prices are hard to beat, so this is a strict test.">
        <Table head={<><Th>Disagreement</Th><Th right>Bets</Th><Th right>Won</Th><Th right>Pushed</Th><Th right>Profit (units)</Th><Th right>ROI</Th></>}>
          {o.roi.length === 0 ? (
            <tr><Td className="text-muted">No graded props yet.</Td><Td>{""}</Td><Td>{""}</Td><Td>{""}</Td><Td>{""}</Td><Td>{""}</Td></tr>
          ) : o.roi.map((x) => (
            <tr key={x.margin}>
              <th scope="row" className="px-4 py-2.5 font-medium">{x.margin === 0 ? "Any" : `${(x.margin * 100).toFixed(0)}+ pts`}</th>
              <Td right>{x.bets}</Td>
              <Td right>{x.wins}</Td>
              <Td right>{x.pushes}</Td>
              <Td right>{x.profit.toFixed(2)}</Td>
              <Td right className={cn(x.roi != null && (x.roi > 0 ? "text-safe-ink" : "text-danger-ink"))}>{x.roi == null ? "—" : `${(x.roi * 100).toFixed(1)}%`}</Td>
            </tr>
          ))}
        </Table>
      </Section>

      <Section id="clv-heading" title="Closing line value" note="The model's side at the opening line (same book), compared with where that line closed. Beat: the closing number was worse for that side, or the same number at a worse price.">
        <Table head={<><Th>Market</Th><Th right>Props</Th><Th right>Beat close</Th><Th right>Same</Th><Th right>Worse</Th><Th right>Beat rate</Th><Th right>Avg. line move</Th></>}>
          {rows.map(([label, m]) => (
            <tr key={label} className={cn(label === "All markets" && "font-semibold")}>
              <th scope="row" className="px-4 py-2.5 font-medium">{label}</th>
              <Td right>{m.clv.n}</Td>
              <Td right>{m.clv.beat}</Td>
              <Td right>{m.clv.same}</Td>
              <Td right>{m.clv.worse}</Td>
              <Td right>{pct(m.clv.beat_rate, 0)}</Td>
              <Td right>{m.clv.mean_line_move == null ? "—" : `${m.clv.mean_line_move > 0 ? "+" : ""}${m.clv.mean_line_move}`}</Td>
            </tr>
          ))}
        </Table>
      </Section>

      <Section id="cal-heading" title="Calibration buckets (all markets)" note="Props grouped by predicted over probability: how often they actually went over. Well calibrated means the two columns match.">
        <Table head={<><Th>Predicted</Th><Th right>Model props</Th><Th right>Model said</Th><Th right>Went over</Th><Th right>Book props</Th><Th right>Book said</Th><Th right>Went over</Th></>}>
          {o.calibration_model.length === 0 && o.calibration_book.length === 0 ? (
            <tr><Td className="text-muted">No graded props yet.</Td><Td>{""}</Td><Td>{""}</Td><Td>{""}</Td><Td>{""}</Td><Td>{""}</Td><Td>{""}</Td></tr>
          ) : Array.from({ length: 10 }, (_, i) => {
            const lo = i / 10;
            const m = o.calibration_model.find((b) => Math.abs(b.lo - lo) < 1e-6);
            const b = o.calibration_book.find((x) => Math.abs(x.lo - lo) < 1e-6);
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
      </Section>

      <Section id="cov-heading" title="Coverage and name matching" note={r.match_rule}>
        <p className="mt-3 text-sm">
          {names.total === 0 ? "No sportsbook players with closing lines yet." : (
            <>Names matched: <span className="num font-semibold">{names.matched} of {names.total}</span> ({pct(names.match_rate, 0)})
              {Object.keys(names.by_method).length > 0 && <> — {Object.entries(names.by_method).map(([k, v]) => `${v} ${k === "exact" ? "exact" : "by last name + initial"}`).join(", ")}</>}.</>
          )}
        </p>
        {r.coverage && Object.keys(r.coverage).length > 0 && (
          <p className="mt-1 text-sm">
            Props: {Object.entries(r.coverage).map(([k, v]) => `${STATUS_LABELS[k] ?? k} ${v}`).join(" · ")}
          </p>
        )}
        {names.unmatched.length > 0 && (
          <Table head={<><Th>Unmatched name</Th><Th>Game</Th><Th>Why</Th></>}>
            {names.unmatched.map((u) => (
              <tr key={`${u.game_id}-${u.name}`}><Td>{u.name}</Td><Td>{u.teams}</Td><Td className="text-muted">{u.reason}</Td></tr>
            ))}
          </Table>
        )}
        {r.not_graded && r.not_graded.length > 0 && (
          <Table head={<><Th>Matched but not graded</Th><Th>Market</Th><Th>Game</Th><Th>Why</Th></>}>
            {r.not_graded.map((u) => (
              <tr key={`${u.game_id}-${u.player_id}-${u.market}`}>
                <Td>{u.name}</Td><Td>{r.markets?.[u.market]?.label ?? u.market}</Td><Td>{u.game_id}</Td>
                <Td className="text-muted">{STATUS_LABELS[u.status] ?? u.status}</Td>
              </tr>
            ))}
          </Table>
        )}
      </Section>
    </>
  );
}
