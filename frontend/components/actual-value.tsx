import { cn } from "@/lib/utils";

export type MissSide = "in" | "above" | "below";

export const missSide = (actual: number, lo: number, hi: number): MissSide =>
  actual > hi ? "above" : actual < lo ? "below" : "in";

const STYLE: Record<MissSide, { cls: string; mark: string; title: string }> = {
  in: { cls: "text-miss-in", mark: "", title: "Inside the 25th–75th range" },
  above: { cls: "text-miss-above", mark: "▲", title: "Above the range: actual higher than projected" },
  below: { cls: "text-miss-below", mark: "▼", title: "Below the range: actual lower than projected" },
};

/** An actual result colored by where it landed against the 25th–75th range. */
export function ActualValue({ actual, lo, hi, digits = 0, prefix, className }: {
  actual: number;
  lo: number;
  hi: number;
  digits?: number;
  prefix?: string;
  className?: string;
}) {
  const s = STYLE[missSide(actual, lo, hi)];
  return (
    <span className={cn("num font-semibold", s.cls, className)} title={s.title}>
      {prefix}
      {s.mark && <span aria-hidden className="mr-0.5 text-[0.8em]">{s.mark}</span>}
      {actual.toFixed(digits)}
      <span className="sr-only"> ({s.title.toLowerCase()})</span>
    </span>
  );
}

export function MissLegend({ className }: { className?: string }) {
  return (
    <div className={cn("text-sm text-muted", className)} aria-label="Legend for actual results">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
        <span><span className="num font-semibold text-miss-in">12</span> Yellow: inside the 25th–75th range</span>
        <span><span className="num font-semibold text-miss-above">▲12</span> Green: actual higher than the projected range</span>
        <span><span className="num font-semibold text-miss-below">▼12</span> Red: actual lower than the projected range</span>
      </div>
      <p className="mt-1 text-xs">Green and red show which direction the projection missed, not a good or bad result.</p>
    </div>
  );
}
