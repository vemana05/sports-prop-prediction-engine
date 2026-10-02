"use client";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { Sides } from "@/lib/types";

const OPTIONS: { value: Sides; label: string }[] = [
  { value: "both", label: "Both" },
  { value: "over", label: "Overs" },
  { value: "under", label: "Unders" },
];

/** Which sides of a prop the search may use: overs only, unders only, or both. */
export function SideSelector({ value, onChange }: { value: Sides; onChange: (s: Sides) => void }) {
  return (
    <ToggleGroup
      type="single"
      value={value}
      onValueChange={(v) => v && onChange(v as Sides)}
      aria-label="Over or under"
      className="grid w-full grid-cols-3"
    >
      {OPTIONS.map((o) => (
        <ToggleGroupItem key={o.value} value={o.value} className="px-2 text-base">
          {o.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}
