import type { Config } from "tailwindcss";

const v = (name: string) => `var(--${name})`;

export default {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    // Spec type scale: 12 / 14 / 16 / 20 / 28 / 40 / 56
    fontSize: {
      xs: ["12px", "16px"],
      sm: ["14px", "20px"],
      base: ["16px", "24px"],
      lg: ["20px", "26px"],
      xl: ["28px", "32px"],
      "2xl": ["40px", "44px"],
      "3xl": ["56px", "56px"],
    },
    extend: {
      colors: {
        surface: v("surface"),
        panel: v("panel"),
        "panel-2": v("panel-2"),
        border: v("border"),
        ink: v("ink"),
        muted: v("muted"),
        accent: { DEFAULT: v("accent"), ink: v("accent-ink") },
        safe: { DEFAULT: v("tier-safe"), ink: v("tier-safe-ink") },
        risky: { DEFAULT: v("tier-risky"), ink: v("tier-risky-ink") },
        danger: { DEFAULT: v("tier-danger"), ink: v("tier-danger-ink") },
        "edge-pos": v("edge-pos"),
        "edge-neg": v("edge-neg"),
        "miss-in": v("miss-in"),
        "miss-above": v("miss-above"),
        "miss-below": v("miss-below"),
      },
      fontFamily: {
        sans: ["var(--font-barlow)", "system-ui", "sans-serif"],
        display: ["var(--font-barlow-condensed)", "var(--font-barlow)", "system-ui", "sans-serif"],
      },
      borderRadius: {
        slip: v("radius-slip"),
        control: v("radius-control"),
      },
      boxShadow: { slip: v("slip-shadow") },
      maxWidth: { prose: "70ch" },
    },
  },
  plugins: [],
} satisfies Config;
