"use client";

import { useState } from "react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { money } from "@/lib/format";

/** Bankroll is typed in only — never connected to a sportsbook (spec Section 8). */
export function BankrollInput({ value, onChange }: { value: number; onChange: (n: number) => void }) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(String(value));
  const parsed = Number(draft.replace(/[$,\s]/g, ""));
  const valid = Number.isFinite(parsed) && parsed > 0 && parsed <= 10_000_000;

  return (
    <Popover
      open={open}
      onOpenChange={(o) => {
        setOpen(o);
        if (o) setDraft(String(value));
      }}
    >
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted">Bankroll</span>
        <span className="num text-xl font-semibold">{money(value)}</span>
        <PopoverTrigger asChild>
          <Button variant="ghost" size="sm" className="text-accent-ink">Edit</Button>
        </PopoverTrigger>
      </div>
      <PopoverContent>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!valid) return;
            onChange(Math.round(parsed * 100) / 100);
            setOpen(false);
          }}
        >
          <label htmlFor="bankroll" className="text-sm font-medium">Bankroll</label>
          <div className="mt-1.5 flex items-center rounded-control border border-border bg-surface focus-within:border-ink">
            <span className="pl-3 text-muted">$</span>
            <input
              id="bankroll"
              inputMode="decimal"
              autoFocus
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              className="num h-10 w-full bg-transparent px-2 text-lg outline-none"
              aria-invalid={!valid}
            />
          </div>
          <p className="mt-2 text-xs text-muted">Stakes are sized from this. It stays in your browser.</p>
          <div className="mt-3 flex justify-end">
            <Button type="submit" variant="primary" size="sm" disabled={!valid}>Save bankroll</Button>
          </div>
        </form>
      </PopoverContent>
    </Popover>
  );
}
