import Link from "next/link";
import { SiteNav } from "./site-nav";
import { ThemeToggle } from "./theme-toggle";

export function SiteHeader() {
  return (
    <header className="border-b border-border bg-panel">
      <div className="mx-auto flex h-14 max-w-[1600px] items-center gap-4 px-4 sm:gap-6 lg:px-6">
        <Link href="/" className="font-display text-xl font-bold tracking-tight">
          Parlay<span className="text-accent-ink">Builder</span>
        </Link>
        <SiteNav />
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
