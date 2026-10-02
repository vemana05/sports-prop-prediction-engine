import type { Metadata, Viewport } from "next";
import { Barlow, Barlow_Condensed } from "next/font/google";
import { Providers } from "./providers";
import { SiteHeader } from "@/components/site-header";
import { SiteFooter } from "@/components/site-footer";
import "./globals.css";

const barlow = Barlow({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-barlow" });
const barlowCondensed = Barlow_Condensed({ subsets: ["latin"], weight: ["600", "700"], variable: "--font-barlow-condensed" });

export const metadata: Metadata = { title: "Parlay Builder", description: "Model-driven parlays with honest math." };
export const viewport: Viewport = { width: "device-width", initialScale: 1 };

// Set theme + league before first paint (no flash). `?theme=` / `?league=`
// override the saved choice — handy for screenshot review.
const initScript = `(() => {
  try {
    const q = new URLSearchParams(location.search);
    const theme = q.get("theme") || localStorage.getItem("theme") ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const league = q.get("league") || localStorage.getItem("league") || "nba";
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.dataset.league = league === "nfl" ? "nfl" : "nba";
  } catch {}
})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-league="nba" className={`${barlow.variable} ${barlowCondensed.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: initScript }} />
      </head>
      <body className="flex min-h-dvh flex-col">
        <Providers>
          <SiteHeader />
          <main className="flex-1">{children}</main>
          <SiteFooter />
        </Providers>
      </body>
    </html>
  );
}
