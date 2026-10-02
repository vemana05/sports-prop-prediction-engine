"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { initials } from "@/lib/format";

export function Headshot({ src, name, className }: { src: string | null; name: string; className?: string }) {
  const [failed, setFailed] = useState(!src);
  return (
    <div
      className={cn(
        "relative flex shrink-0 items-end justify-center overflow-hidden rounded-full border border-border bg-panel-2",
        className,
      )}
    >
      {failed ? (
        <span className="self-center font-display text-sm font-semibold text-muted">{initials(name)}</span>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src!} alt="" loading="lazy" className="h-[115%] w-auto max-w-none object-cover" onError={() => setFailed(true)} />
      )}
    </div>
  );
}
