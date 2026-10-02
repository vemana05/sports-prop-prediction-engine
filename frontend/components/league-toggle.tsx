"use client";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { League } from "@/lib/types";

export function LeagueToggle({ value, onChange }: { value: League; onChange: (l: League) => void }) {
  return (
    <ToggleGroup type="single" value={value} onValueChange={(v) => v && onChange(v as League)} aria-label="League">
      <ToggleGroupItem value="nba" className="data-[state=on]:text-accent-ink">NBA</ToggleGroupItem>
      <ToggleGroupItem value="nfl" className="data-[state=on]:text-accent-ink">NFL</ToggleGroupItem>
    </ToggleGroup>
  );
}
