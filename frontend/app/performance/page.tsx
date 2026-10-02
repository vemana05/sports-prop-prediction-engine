"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { usePerformance } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { Performance } from "@/lib/types";

const pctPts = (x: number) => `${(x * 100).toFixed(1)} pts`;

function summary(p: Performance) {
  const wf = p.walk_forward.rows;
  const beats = wf.filter((r) => r.model_mae < r.naive_mae);
  const pts = wf.find((r) => r.market === "points");
  const ptsCal = p.calibration.rows.find((r) => r.market === "points");
  return { beats, pts, ptsCal, ties: wf.filter((r) => r.model_mae >= r.naive_mae) };
}

export default function PerformancePage() {
  const { data, error } = usePerformance();

  return (
    <div className="mx-auto max-w-[1100px] px-4 pb-16 pt-8 lg:px-6">
      <h1 className="font-display text-2xl font-bold">Model performance</h1>

      {error ? (
        <p className="mt-4 text-danger-ink">Couldn't load the performance report: {(error as Error).message}</p>
      ) : !data ? (
        <div className="mt-6 space-y-3">
          <Skeleton className="h-28 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : (
        <Body p={data} />
      )}
    </div>
  );
}

function Body({ p }: { p: Performance }) {
  const { beats, pts, ptsCal, ties } = summary(p);
  return (
    <>
      <section aria-label="Summary" className="mt-5 rounded-slip border border-border bg-panel p-5 sm:p-6">
        {!p.tested_against_real_lines && (
          <p className="mb-3 inline-flex rounded-full bg-risky/15 px-2.5 py-0.5 text-sm font-medium text-risky-ink">
            Not yet tested against real sportsbook lines
          </p>
        )}
        <div className="max-w-prose space-y-3 text-base">
          <p>
            Everything on this page is measured against past games, not against real betting lines. {p.real_line_note}
          </p>
          <p>
            Against past seasons it never saw, the model predicts {beats.map((r) => r.label.toLowerCase()).join(", ")} a little
            better than just using a player's recent average
            {pts && (
              <>
                {" "}— for points, it's off by <span className="num font-semibold">{pts.model_mae.toFixed(2)}</span> on average vs.{" "}
                <span className="num font-semibold">{pts.naive_mae.toFixed(2)}</span>
              </>
            )}
            . That edge is small.
            {ties.length > 0 && (
              <> For {ties.map((r) => r.label.toLowerCase()).join(", ")}, the recent average does as well or better, so that's what the app uses.</>
            )}
          </p>
          {ptsCal?.error_before != null && ptsCal.error_after != null && (
            <p>
              Raw probabilities were overconfident. After calibration, when the model says a points prop hits X% of the time, it hit
              within about <span className="num font-semibold">{pctPts(ptsCal.error_after)}</span> of X on held-out games (down from{" "}
              <span className="num font-semibold">{pctPts(ptsCal.error_before)}</span>).
            </p>
          )}
          <p className="text-sm text-muted">
            Only {p.live_markets.join(", ")} props have live odds right now. Report as of {p.as_of}.
          </p>
        </div>
      </section>

      <section aria-labelledby="wf-heading" className="mt-10">
        <h2 id="wf-heading" className="font-display text-xl font-bold">Prediction error vs. a naive baseline</h2>
        <p className="mt-1 max-w-prose text-sm text-muted">
          Mean absolute error (lower is better). {p.walk_forward.source} Baseline: {p.walk_forward.baseline_name.toLowerCase()}.
        </p>
        <div className="mt-4 overflow-x-auto rounded-control border border-border bg-panel">
          <table className="w-full min-w-[560px] text-left text-sm">
            <thead className="border-b border-border text-muted">
              <tr>
                <th scope="col" className="px-4 py-2.5 font-medium">Market</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">Model MAE</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">Naive MAE</th>
                <th scope="col" className="px-4 py-2.5 text-right font-medium">Model vs. naive</th>
                <th scope="col" className="px-4 py-2.5 font-medium">Used live</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {p.walk_forward.rows.map((r) => {
                const gain = (r.naive_mae - r.model_mae) / r.naive_mae;
                return (
                  <tr key={r.market}>
                    <th scope="row" className="px-4 py-2.5 font-medium">{r.label}</th>
                    <td className="num px-4 py-2.5 text-right text-base font-semibold">{r.model_mae.toFixed(3)}</td>
                    <td className="num px-4 py-2.5 text-right text-base text-muted">{r.naive_mae.toFixed(3)}</td>
                    <td className={cn("num px-4 py-2.5 text-right text-base", gain > 0 ? "text-safe-ink" : "text-muted")}>
                      {Math.abs(gain * 100).toFixed(1)}% {gain > 0 ? "better" : "worse"}
                    </td>
                    <td className="px-4 py-2.5">{r.used_live === "model" ? "Model" : "Recent average"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section aria-labelledby="cal-heading" className="mt-10">
        <h2 id="cal-heading" className="font-display text-xl font-bold">Calibration, before and after isotonic</h2>
        <p className="mt-1 max-w-prose text-sm text-muted">
          Each dot is a decile of predictions: how often the model said a prop would go over (x) vs. how often it did (y). On the
          dashed line, a 70% call hits 70% of the time. Red is the raw model; blue is after calibration. {p.calibration.source}
        </p>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          {p.calibration.rows.map((r) => (
            <figure key={r.market} className="rounded-control border border-border bg-panel p-3">
              <figcaption className="flex flex-wrap items-baseline justify-between gap-x-3 px-1 pb-2">
                <span className="font-display text-lg font-semibold">{r.label}</span>
                <span className="text-xs text-muted">
                  {r.error_before != null && r.error_after != null ? (
                    <>
                      Avg. miss <span className="num font-semibold text-ink">{pctPts(r.error_before)}</span> →{" "}
                      <span className="num font-semibold text-safe-ink">{pctPts(r.error_after)}</span>
                    </>
                  ) : (
                    "Error not recorded — see chart"
                  )}
                </span>
              </figcaption>
              {r.chart_url ? (
                // The charts are matplotlib PNGs on a white background; keep them on white in dark mode too.
                <img
                  src={r.chart_url}
                  alt={`${r.label} calibration: predicted vs. actual hit rate, before and after isotonic calibration`}
                  className="aspect-square w-full rounded bg-white object-contain"
                  loading="lazy"
                />
              ) : (
                <p className="px-1 py-8 text-sm text-muted">Chart missing from reports/.</p>
              )}
            </figure>
          ))}
        </div>
      </section>
    </>
  );
}
