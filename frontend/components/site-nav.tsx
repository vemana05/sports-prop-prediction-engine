"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", short: "Build", long: "Build" },
  { href: "/schedule", short: "Schedule", long: "Schedule" },
  { href: "/tracker", short: "Tracker", long: "Bet tracker" },
  { href: "/performance", short: "Model", long: "Model performance" },
];

export function SiteNav() {
  const path = usePathname();
  return (
    <nav className="flex gap-4 text-sm">
      {LINKS.map((l) => {
        const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
        return (
          <Link key={l.href} href={l.href} aria-current={active ? "page" : undefined} className={cn(active ? "font-medium text-ink" : "text-muted hover:text-ink")}>
            <span className="sm:hidden">{l.short}</span>
            <span className="hidden sm:inline">{l.long}</span>
          </Link>
        );
      })}
    </nav>
  );
}
