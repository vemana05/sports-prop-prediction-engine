import { cn } from "@/lib/utils";
import { pct } from "@/lib/format";
import type { League } from "@/lib/types";

/**
 * Spec 10.4 B: model probability (solid --ink marker) vs book no-vig
 * probability (hollow --muted marker). The gap is shaded --edge-pos when the
 * model is higher, --edge-neg (grey — "no edge", not an error) when lower.
 * NFL mode adds yard-line ticks every 10%.
 */
export function ProbabilityBar({
  model,
  book,
  league,
  className,
}: {
  model: number;
  book: number | null;
  league: League;
  className?: string;
}) {
  const lo = book == null ? model : Math.min(model, book);
  const hi = book == null ? model : Math.max(model, book);
  const edge = book == null ? null : model >= book ? "pos" : "neg";
  const label = `Model ${pct(model)}${book == null ? "" : `, book ${pct(book)}`}`;

  return (
    <div className={cn("w-full", className)}>
      <div role="img" aria-label={label} className="relative h-4">
        <div className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-panel-2" />
        {league === "nfl" && (
          <div className="absolute inset-x-0 top-1/2 h-3 -translate-y-1/2" aria-hidden>
            {Array.from({ length: 9 }, (_, i) => (
              <span key={i} className="absolute top-0 h-full w-px bg-border" style={{ left: `${(i + 1) * 10}%` }} />
            ))}
          </div>
        )}
        {edge && (
          <div
            className={cn(
              "absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full",
              edge === "pos" ? "bg-edge-pos" : "bg-edge-neg/60",
            )}
            style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }}
          />
        )}
        {book != null && (
          <span
            className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-muted bg-panel"
            style={{ left: `${book * 100}%` }}
          />
        )}
        <span
          className="absolute top-1/2 h-4 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-ink"
          style={{ left: `${model * 100}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-xs">
        <span className="text-ink">
          Model <span className="num text-sm font-semibold">{pct(model)}</span>
        </span>
        <span className="text-muted">
          Book <span className="num text-sm font-semibold">{pct(book)}</span>
        </span>
      </div>
    </div>
  );
}
