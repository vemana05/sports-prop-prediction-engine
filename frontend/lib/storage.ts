"use client";

import { useEffect, useState } from "react";

/** State mirrored to localStorage or sessionStorage. Storage can throw
 * (private windows, blocked site data), so every access is guarded and the
 * app works without it. */
export function useStored<T>(key: string, initial: T, opts: { session?: boolean; parse?: (raw: string) => T | undefined } = {}) {
  const [value, setValue] = useState<T>(initial);
  const store = () => (opts.session ? sessionStorage : localStorage);
  useEffect(() => {
    try {
      const raw = store().getItem(key);
      if (raw == null) return;
      const parsed = opts.parse ? opts.parse(raw) : (JSON.parse(raw) as T);
      if (parsed !== undefined) setValue(parsed);
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const set = (v: T) => {
    setValue(v);
    try {
      store().setItem(key, opts.parse ? String(v) : JSON.stringify(v));
    } catch {}
  };
  return [value, set] as const;
}

/** Paper trading: bets logged while on are tracked without real money. Shared by the slip and the tracker. */
export const usePaperMode = () => useStored<boolean>("paper-trading", false);
