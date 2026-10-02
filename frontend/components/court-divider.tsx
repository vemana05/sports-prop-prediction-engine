import type { League } from "@/lib/types";

/**
 * Section divider that takes its look from the playing surface (spec 10.2):
 * NBA — the arc of a three-point line; NFL — yard-line hash marks.
 */
export function CourtDivider({ league }: { league: League }) {
  return (
    <div className="pointer-events-none my-4 text-accent" aria-hidden>
      {league === "nba" ? (
        <svg viewBox="0 0 400 36" preserveAspectRatio="none" className="h-8 w-full">
          <path d="M0 2 H40 V8 C40 26 112 34 200 34 C288 34 360 26 360 8 V2 H400" fill="none" stroke="currentColor" strokeWidth="1.5" vectorEffect="non-scaling-stroke" opacity="0.55" />
        </svg>
      ) : (
        <svg viewBox="0 0 400 16" preserveAspectRatio="none" className="h-4 w-full">
          <line x1="0" y1="8" x2="400" y2="8" stroke="currentColor" strokeWidth="1.5" vectorEffect="non-scaling-stroke" opacity="0.55" />
          {Array.from({ length: 41 }, (_, i) => (
            <line key={i} x1={i * 10} x2={i * 10} y1={i % 5 === 0 ? 0 : 4} y2={i % 5 === 0 ? 16 : 12} stroke="currentColor" strokeWidth="1" vectorEffect="non-scaling-stroke" opacity="0.55" />
          ))}
        </svg>
      )}
    </div>
  );
}
