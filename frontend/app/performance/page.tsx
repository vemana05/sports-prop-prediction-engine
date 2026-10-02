"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { usePerformance } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { NflPerformance, Performance } from "@/lib/types";

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

      {p.nfl && <NflSection nfl={p.nfl} />}
    </>
  );
}

const better = (base: number, model: number) => (base - model) / base;

function NflSection({ nfl }: { nfl: NflPerformance }) {
  const seasons = `${nfl.test_seasons[0]}–${nfl.test_seasons[nfl.test_seasons.length - 1]}`;
  // How much of the model's improvement over a plain last-5 average a no-model shrunk average already gets.
  const shrinkShare = nfl.rows.map((r) => (r.last5_mae - r.shrunk_mae) / (r.last5_mae - r.model_mae));
  const minShare = Math.min(...shrinkShare);
  const overShrunk = nfl.rows.map((r) => better(r.shrunk_mae, r.model_mae) * 100);
  const priorWins = nfl.rows.filter((r) => r.no_history && r.no_history.position_prior_mae < r.no_history.model_mae);
  const calWorse = nfl.rows.filter((r) => r.error_after > r.error_before);
  return (
    <section aria-labelledby="nfl-heading" data-league="nfl" className="mt-14 border-t border-border pt-8">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 id="nfl-heading" className="font-display text-xl font-bold">NFL</h2>
        <span className="rounded-full bg-accent/15 px-2.5 py-0.5 text-sm font-medium text-accent-ink">Projections only, odds not enabled</span>
      </div>
      <div className="mt-3 max-w-prose space-y-3 text-base">
        <p>
          Walk-forward by season ({seasons}; the latest season is partial): each season is predicted by a model trained only on earlier
          seasons. The model beats a player's own last-5-game average in all four markets.
        </p>
        <p>
          <span className="font-semibold">Most of that gain is shrinkage, not the model.</span> Just pulling a player's last-5 average
          toward the typical number for his position and depth-chart slot (no model at all) gets at least{" "}
          <span className="num font-semibold">{Math.round(minShare * 100)}%</span> of the way there in every market. The model only adds
          another <span className="num">{Math.min(...overShrunk).toFixed(1)}–{Math.max(...overShrunk).toFixed(1)}%</span> on top.
        </p>
        {(priorWins.length > 0 || calWorse.length > 0) && (
          <p>
            Where it doesn't win:
            {priorWins.length > 0 && (
              <> for a player's first game, the plain position average beats the model on {priorWins.map((r) => r.label.toLowerCase()).join(" and ")} (a small sample).</>
            )}
            {calWorse.length > 0 && (
              <> Isotonic calibration made {calWorse.map((r) => r.label.toLowerCase()).join(" and ")} slightly <em>worse</em>, so those raw probabilities were already about as honest as calibration could make them.</>
            )}
          </p>
        )}
      </div>

      <div className="mt-5 overflow-x-auto rounded-control border border-border bg-panel">
        <table className="w-full min-w-[680px] text-left text-sm">
          <thead className="border-b border-border text-muted">
            <tr>
              <th scope="col" className="px-4 py-2.5 font-medium">Market</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">Model MAE</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">Last-5 avg</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">Shrunk avg (no model)</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">vs. last-5</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">vs. shrunk</th>
              <th scope="col" className="px-4 py-2.5 text-right font-medium">Games</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {nfl.rows.map((r) => {
              const g5 = better(r.last5_mae, r.model_mae);
              const gs = better(r.shrunk_mae, r.model_mae);
              return (
                <tr key={r.market}>
                  <th scope="row" className="px-4 py-2.5 font-medium">{r.label}</th>
                  <td className="num px-4 py-2.5 text-right text-base font-semibold">{r.model_mae.toFixed(r.model_mae < 10 ? 3 : 1)}</td>
                  <td className="num px-4 py-2.5 text-right text-base text-muted">{r.last5_mae.toFixed(r.last5_mae < 10 ? 3 : 1)}</td>
                  <td className="num px-4 py-2.5 text-right text-base text-muted">{r.shrunk_mae.toFixed(r.shrunk_mae < 10 ? 3 : 1)}</td>
                  <td className={cn("num px-4 py-2.5 text-right text-base", g5 > 0 ? "text-safe-ink" : "text-muted")}>
                    {Math.abs(g5 * 100).toFixed(1)}% {g5 > 0 ? "better" : "worse"}
                  </td>
                  <td className={cn("num px-4 py-2.5 text-right text-base", gs > 0 ? "text-ink" : "text-muted")}>
                    {Math.abs(gs * 100).toFixed(1)}% {gs > 0 ? "better" : "worse"}
                  </td>
                  <td className="num px-4 py-2.5 text-right text-muted">{r.n.toLocaleString("en-US")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-sm text-muted">
        Mean absolute error on the same games for every column (games where the player has at least one earlier game). "Shrunk avg" is
        last-5 pulled toward last season's average for the same position and depth slot (e.g. WR2).
      </p>

      <h3 className="mt-8 font-display text-lg font-bold">NFL calibration against synthetic lines</h3>
      <p className="mt-1 max-w-prose text-sm text-muted">
        Line = the player's last-5 average rounded to .5. Red is the raw model, blue is after isotonic calibration fit only on earlier
        seasons.
      </p>
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        {nfl.rows.map((r) => (
          <figure key={r.market} className="rounded-control border border-border bg-panel p-3">
            <figcaption className="flex flex-wrap items-baseline justify-between gap-x-3 px-1 pb-2">
              <span className="font-display text-lg font-semibold">{r.label}</span>
              <span className="text-xs text-muted">
                Avg. miss <span className="num font-semibold text-ink">{pctPts(r.error_before)}</span> →{" "}
                <span className={cn("num font-semibold", r.error_after <= r.error_before ? "text-safe-ink" : "text-danger-ink")}>
                  {pctPts(r.error_after)}
                </span>
              </span>
            </figcaption>
            {r.chart_url ? (
              <img
                src={r.chart_url}
                alt={`NFL ${r.label.toLowerCase()} calibration: predicted vs. actual hit rate, before and after isotonic calibration`}
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
  );
}
