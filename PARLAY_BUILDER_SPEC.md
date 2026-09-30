# Parlay Builder — Project Spec for Claude Code (v2)

> Hand this whole file to Claude Code. It covers what to build, every data source and how they fit together, how the prediction model works, how bet sizing works, the full UI design system, and the build order.
>
> **v2 changes:** many more data sources (Section 3), a source registry + player ID crosswalk so sources can be combined safely (Section 4), and a complete visual design spec with a Next.js frontend replacing Streamlit (Section 10).

---

## 1. What we're building

A web app that builds NFL or NBA parlays from **player props and game lines**, using **its own stat-based probability model** (not the sportsbook's implied odds) to estimate how likely each leg and the whole parlay are to hit.

### Core features
1. **League toggle:** NFL or NBA.
2. **Risk tier:** Safe / Risky / Dangerous (Section 7).
3. **Bankroll input:** user enters their balance; app recommends a stake per parlay (Section 8).
4. **Model probability per leg and per parlay**, built from many data sources.
5. **Model vs. sportsbook comparison** for every leg (the "edge").
6. **Player deep-dive drawer:** game log chart with the prop line drawn on it, hit rates, matchup, injuries, minutes/snap trend, weather (NFL).
7. **Plain-English reasons** for each leg.
8. **Bet tracker:** log bets, results, bankroll curve, ROI, closing line value.
9. **Model performance page:** calibration and backtest results, so the user can see how trustworthy the numbers are.
10. **Data freshness panel:** shows when each source last updated and flags anything stale.

### Timing note
The NFL season is underway now; the 2026–27 NBA regular season starts October 20, 2026. NBA-first is still the right build order (more games per player = better training data), and it can be backtested on past seasons before tip-off.

---

## 2. AI vs. algorithm — the recommended approach

**Use a statistical / machine-learning algorithm for predictions. Do NOT ask an LLM to predict outcomes.**

- LLMs aren't calibrated number predictors; they produce confident percentages that aren't grounded in the data.
- The real problem is: *"given this player's history and this matchup, what's the probability distribution of his points / yards / receptions?"* Gradient-boosted models + probability distributions handle that well and can be **tested against real results**.
- An algorithm is reproducible, cheap, and measurable.

**Where an LLM helps (both optional, off by default):**
1. **Explanations:** turning feature values into a one-line reason per leg.
2. **News → structured flags:** reading injury/news blurbs and extracting structured signals like `minutes_restriction=true` or `role_change=starter`. These flags go into the review queue and feature pipeline; the LLM never outputs a probability.

### Pipeline
```
Many sources (stats, tracking, lineups, injuries, weather, news, odds)
        ↓  source registry + player ID crosswalk + validation
Unified warehouse (DuckDB)
        ↓
Feature engineering (rolling form, matchup, pace, minutes/snaps, rest, travel, refs, weather…)
        ↓
Minutes/snaps model → per-stat LightGBM models → mean + spread
        ↓
Probability distribution → P(over line) per prop
        ↓
Calibration (isotonic) so 70% really means ~70%
        ↓
Correlation-aware Monte Carlo → P(parlay hits)
        ↓
Parlay search per tier → EV + fractional Kelly sizing with caps
        ↓
Next.js UI
```

---

## 3. Data sources (researched September 2026)

Verify pricing/limits at signup; they change often. Everything below is grouped by what it contributes to the model.

### 3.1 NFL — core stats (FREE)

**nflverse via `nflreadpy`** — the backbone for NFL.
- `pip install nflreadpy` (replaces the deprecated `nfl_data_py`). Returns Polars DataFrames.
- Loaders to use:
  - `load_pbp()` — play-by-play with EPA, WPA, air yards, etc.
  - `load_player_stats()` — weekly player stats
  - `load_team_stats()`, `load_schedules()` (includes roof, surface, and game-time conditions)
  - `load_rosters_weekly()`, `load_depth_charts()`
  - `load_snap_counts()` — snap share
  - `load_injuries()` — weekly injury designations
  - `load_nextgen_stats()` — NFL Next Gen Stats (separation, time to throw, rush yards over expected)
  - `load_ftn_charting()` — FTN charted data (play action, motion, blitzes)
  - `load_participation()` — who was on the field (historical)
  - `load_ff_opportunity()` — expected fantasy points / opportunity model
  - `load_ff_playerids()` — **player ID crosswalk across platforms** (critical, see Section 4)
- License: CC-BY 4.0 (FTN data is CC-BY-SA 4.0). Attribution in app footer.
- Docs: https://nflreadpy.nflverse.com/

### 3.2 NBA — core stats

1. **`nba_api`** (FREE) — wraps stats.nba.com: game logs, advanced/tracking stats, shot data, box score summaries (which include **game officials**). Unofficial and can break or rate-limit; server-side only; add delays, retries, caching.
2. **`pbpstats`** (FREE) — parses NBA play-by-play into **possessions with the lineup on the floor for every event**, plus shot/rebound/assist breakdowns by zone. `pip install pbpstats`. Use for:
   - On/off splits: how a player's usage, points, and assists change when a specific teammate sits
   - Possession counts (better pace estimates than box-score formulas)
   - Configure its local data directory so games are cached to disk and never re-downloaded.
3. **BALLDONTLIE** (https://www.balldontlie.io) — clean documented REST API with SDKs. Free tier ~5 req/min; game player stats + injuries need the ~$9.99/mo ALL-STAR tier; higher tiers include advanced stats, odds, and player props. Use as the reliable daily updater and fallback.
4. **`sportsdataverse` (Python)** — ESPN + stats.nba.com wrappers, also mirrors nflverse. Use as a **second source for cross-checking** box scores and for ESPN game IDs. Install from GitHub per its README.

### 3.3 Injuries, availability & lineups

| Source | League | What it adds | Cost |
|---|---|---|---|
| **`nbainjuries`** (PyPI) | NBA | Parses the **official NBA injury report** PDFs into structured data; historical reports back to 2021-22 in hourly/15-min snapshots. Needs Java (uses tabula-py). | Free |
| **`nba-injury-report`** (PyPI) | NBA | Alternate parser for the same official reports; returns JSON/DataFrame. Use as fallback. | Free |
| nflreadpy `load_injuries()` | NFL | Weekly game-status designations | Free |
| **Sleeper API** (`api.sleeper.app/v1`) | NFL + NBA | Player status, injury status, depth-chart order, **trending adds/drops** (a fast signal that news just broke). Public, no key. Pull `/players/{sport}` at most once a day (~5 MB); stay under ~1,000 calls/min. | Free |
| ESPN (unofficial) | Both | Team news, injuries, depth charts via `site.api.espn.com/apis/site/v2/sports/{sport}/{league}/...` | Free |
| BALLDONTLIE injuries | NBA | Structured injury feed | Paid tier |

**Why historical injury snapshots matter:** the backtest must use the injury report *as it looked before tip-off*, not the final outcome. `nbainjuries`' timestamped snapshots make that possible for NBA.

### 3.4 Game context

- **Weather (NFL outdoor games): Open-Meteo** (https://open-meteo.com)
  - No API key, free for non-commercial use up to ~10,000 calls/day, CC BY 4.0.
  - Forecast API for upcoming games (wind, precipitation, temperature at kickoff hour, using stadium lat/long).
  - **Historical Forecast archive (from 2021)**: use this for backtesting, not the observed-weather archive, so the model trains on what forecasts said before kickoff (no look-ahead).
  - Skip weather for domes/closed roofs (roof type from nflverse schedules).
- **Stadium/arena table** (build once, store in repo as CSV): lat/long, altitude (Denver effect), roof type, surface, timezone.
- **Travel & rest** (computed, no API): miles traveled, timezone shift, days of rest, back-to-backs, 3-in-4 nights, from schedules + arena coordinates.
- **Referees (NBA):** officials per game from nba_api box score summaries → build per-crew tendencies (fouls/game, free throw rate) → affects points and free throws.
- **Scores/schedules/news backup: ESPN unofficial API** — scoreboard, `summary?event={id}` (box score + play-by-play), teams, news. Undocumented; can change without notice; cache heavily and keep volume low.

### 3.5 Sportsbook odds & props

1. **The Odds API** (https://the-odds-api.com) — primary.
   - Free: 500 credits/month, no historical. Paid from $30/mo (20K credits) with historical snapshots back to 2020. Player props for major US books (DraftKings, FanDuel, BetMGM, Caesars…).
   - ⚠️ Other companies use nearly identical names (e.g. theoddsapi.com is a separate service). Use **the-odds-api.com, v4** only.
2. **Secondary odds source** (pick one, free tier) for cross-checking lines and filling gaps: **SportsGameOdds**, **OddsPapi**, or BALLDONTLIE's odds/props endpoints.
3. **Odds snapshot store:** save every pull with a timestamp. This enables:
   - Line-movement charts in the UI
   - **Closing line value (CLV)** tracking: did the model's picks beat the closing line? That's the best early indicator the model has real edge, long before win/loss results are statistically meaningful.

### 3.6 Premium upgrades (only if the project gets serious)
- **SportsDataIO** — stats, injuries, depth charts, projections, news, odds. Free trial data is *scrambled* (good for integration code only). Real data needs a commercial agreement.
- **Sportradar** — enterprise official data; trial keys, expensive production.

### 3.7 Explicitly avoid
- Scraping Basketball-Reference / Pro-Football-Reference (terms restrict automated scraping).
- Anything behind a login, or paid sites without an API license (e.g. copying Cleaning the Glass tables).
- Social-media scraping for news.

### 3.8 Source summary (MVP cost ≈ $0–$40/mo)
| Need | NFL | NBA |
|---|---|---|
| Historical box/weekly stats | nflreadpy | nba_api (+ sportsdataverse cross-check) |
| Play-by-play / lineups | nflreadpy pbp, participation, FTN | pbpstats |
| Tracking / advanced | Next Gen Stats, ff_opportunity | nba_api tracking endpoints |
| Daily updates | nflreadpy | BALLDONTLIE (fallback nba_api) |
| Injuries | nflreadpy + Sleeper + ESPN | nbainjuries + Sleeper + ESPN |
| Depth charts | nflreadpy + Sleeper | Sleeper + starters from recent box scores |
| Weather | Open-Meteo | — |
| Refs | — | nba_api officials |
| Odds/props | The Odds API + secondary | The Odds API + secondary |

---

## 4. Combining sources safely

Many sources only help if they're joined correctly. Build this layer before any modeling.

### 4.1 Source registry (`config/sources.yaml`)
For each source: name, league, data types provided, priority (primary/fallback), rate limit, auth env var, refresh schedule, license/attribution text.

```yaml
- name: nbainjuries
  league: nba
  provides: [injuries]
  priority: primary
  refresh: "every 15m on game days"
  rate_limit_per_min: 20
- name: sleeper
  league: [nba, nfl]
  provides: [player_status, depth_chart, trending]
  priority: secondary
  refresh: "hourly"
```

The ingestion runner reads this file; if a primary source fails, it falls back automatically and logs it.

### 4.2 Player & team ID crosswalk
Every source uses different IDs and name spellings ("Lively II, Dereck" vs "Dereck Lively II").
- Table `player_xref`: internal `player_id` ↔ nba.com ID, nflverse `gsis_id`, ESPN ID, Sleeper ID, BALLDONTLIE ID, odds-API name strings.
- Seed NFL from `nflreadpy.load_ff_playerids()`; seed NBA from nba.com IDs + Sleeper's player dump.
- Fuzzy matching (normalize suffixes, accents, punctuation) for new names, with a `manual_overrides.csv` for failures.
- **Unmatched names go to a review queue shown in the UI** — never silently dropped.

### 4.3 Validation & reconciliation
- When two sources report the same box score, compare them; flag discrepancies > tolerance.
- Freshness checks: each source has an expected update cadence; the UI's data-freshness panel turns amber/red when a source is late.
- Great Expectations–style checks (or plain pytest): no duplicate player-games, minutes within 0–60, snap % within 0–100, etc.

### 4.4 Point-in-time correctness
Every row stores `source_published_at` and `ingested_at`. Feature builders query "as of" the game's start time so backtests only see information that existed before the game.

---

## 5. The prediction model

### 5.1 Markets
- **NBA:** points, rebounds, assists, 3PM, PRA, steals+blocks
- **NFL:** passing yards, passing TDs, completions, rushing yards, receiving yards, receptions, anytime TD
- **Game lines (v2):** spread, total, moneyline

### 5.2 Features (all computed only from data available before the game)

Shared:
- Rolling means/std: last 3/5/10, season-to-date, prior season (blended early season)
- Home/away, rest days, travel miles, timezone shift, altitude
- Opponent defense vs. position (rolling), opponent pace
- Player injury status + games since return
- **Teammate availability:** who's out, and that teammate's measured effect on this player (from on/off data)
- Optional LLM news flags (minutes restriction, role change)

NBA:
- Projected minutes (separate model; the most important input)
- Usage rate, shots/min, touches, potential assists, rebound chances (tracking)
- On/off splits from pbpstats for key teammates
- Possessions-based pace for both teams
- Referee crew foul/FT tendencies
- Back-to-back, 3-in-4

NFL:
- Snap share, route participation, target share, air-yards share, carry share, red-zone usage
- Next Gen Stats (separation, RYOE, time to throw), FTN charting (play action, motion rates)
- QB quality (EPA/play), opponent pass/run defense EPA, pressure rate
- **Weather:** wind speed, precipitation, temperature (Open-Meteo), roof, surface
- Depth-chart position and changes

> **"Not based on the sportsbook":** default `USE_MARKET_FEATURES=false` keeps all betting lines out of the features. Odds are only used afterward for payout/EV. The flag lets this be tested the other way later.

### 5.3 Model design
1. Minutes/snaps model (LightGBM regression)
2. Per-market LightGBM models → predicted mean
3. Distributions → P(over line):
   - Counts (reb, ast, 3PM, receptions, TDs): Negative Binomial
   - NBA points/PRA: Normal or Negative Binomial with player-specific variance
   - NFL yards: LightGBM quantile regression (skewed)
   - Anytime TD: binary classifier
4. Isotonic calibration on out-of-sample predictions
5. Push handling for whole-number lines

### 5.4 Validation (required before the UI shows numbers)
- Walk-forward backtest by season; never random splits
- Brier score, log loss, calibration plots
- Compare against the book's no-vig probability on the same props; if the model is worse, the UI says so
- Simulated ROI and **CLV** using historical odds
- **Feature ablation report:** retrain with each source group removed (weather, refs, on/off, tracking, news). Keep a source only if it measurably improves calibration. More sources ≠ better unless proven.
- Results saved to `reports/` and shown on the Model Performance page

### 5.5 Retraining
Nightly: ingest, rebuild features, re-score upcoming games. Weekly retrain; monthly calibration review.

---

## 6. Parlay probability (correlation)

Don't multiply leg probabilities when legs are related (QB yards + his WR's yards; teammates in a shootout; RB rushing vs. opposing QB passing via game script).

- Estimate pairwise correlations from historical model residuals (same player, teammates, opponents).
- Monte Carlo with a Gaussian copula: 10k–50k simulations per finalist parlay.
- Cross-game legs = independent.
- For same-game parlays, use the book's quoted SGP price (books reduce payouts for correlation).

---

## 7. Risk tiers (`config/tiers.yaml`)

| Tier | Legs | Min per-leg model prob | Target parlay hit prob | Max stake (% bankroll) |
|---|---|---|---|---|
| **Safe** | 2–3 | ≥ 70% | ≥ 35% | 2% |
| **Risky** | 3–5 | ≥ 58% | 12–35% | 1% |
| **Dangerous** | 5–8 | ≥ 50% | < 12% | 0.5% |

### Parlay search
1. Build today's leg pool (model prob, line, odds).
2. Remove duplicates/contradictions.
3. Filter by tier min probability.
4. Beam search (width ~200) maximizing EV within tier constraints; simulate finalists.
5. Return top 3–5 per tier.
6. **Manual mode:** user adds their own legs and the app scores them.

---

## 8. Bankroll & stake recommendation

Inputs: bankroll (typed in; never connect to sportsbook accounts), decimal odds `d`, model probability `p`.

1. `EV per $1 = p × (d − 1) − (1 − p)`
2. `Kelly f* = (p × d − 1) / (d − 1)`
3. Shrink `p` toward the book's no-vig probability by `MODEL_SHRINK_TO_MARKET` (default 0.30) before sizing
4. `stake = bankroll × f* × 0.25` (quarter Kelly)
5. `stake = min(stake, bankroll × tier_cap)`
6. **EV ≤ 0 → $0**, with: "The model doesn't see an edge on this parlay. If you play it anyway, treat it as entertainment money."
7. Round to $0.50; respect book minimums

Show the math in the UI: *Model 31% · Book 24% · +320 · EV +$0.30 per $1 · Suggested $12.50*.

---

## 9. Honest-math & responsible-gambling requirements

- Parlays multiply the sportsbook margin; most lose long-term. The app's value is finding the minority with real model edge.
- Never use words like "lock," "guaranteed," "can't miss."
- Model Performance page is always one click away.
- Footer on every page: 21+ (including Virginia), **1-800-GAMBLER**.
- Optional daily/weekly loss limit with warnings.

---

## 10. UI design spec (make it look great)

### 10.1 Stack
- **Next.js (App Router) + TypeScript**
- **Tailwind CSS** with the design tokens below as CSS variables
- **shadcn/ui** (Radix primitives) for accessible dialogs, drawers, tabs, sliders, toasts
- **Recharts** (or Visx for custom charts) for game logs, distributions, calibration, bankroll curve
- **Framer Motion** for the one signature animation (10.5)
- **TanStack Query** for data fetching/caching against the FastAPI backend
- Light and dark themes; fully responsive down to phone width (people build parlays on their phones)

Drop Streamlit for the user-facing app. Keep it only as an optional internal dev dashboard if useful.

### 10.2 Design concept: "the court and the field"
The UI takes its look from the playing surface of the selected league, so switching NFL ↔ NBA visibly changes the room:
- **NBA mode:** warm maple hardwood accents, subtle court-line geometry (the arc of a three-point line as a section divider on the builder page).
- **NFL mode:** turf green accents, yard-line hash marks as the ruler on probability bars.

The parlay itself is presented as a **bet slip**: a tall ticket card with a perforated tear edge, stacked legs, and the totals printed at the bottom. That slip is the one bold, memorable element; everything around it stays calm and disciplined.

### 10.3 Design tokens

Color (define as CSS variables; dark theme values in parentheses):
| Token | Light | Dark | Use |
|---|---|---|---|
| `--surface` | `#F3F5F8` | `#18222F` | App background (cool, not cream; slate, not near-black) |
| `--panel` | `#FFFFFF` | `#213043` | Cards, drawers |
| `--ink` | `#16233A` | `#E8EDF4` | Primary text |
| `--muted` | `#5E6B80` | `#9AA7BA` | Secondary text, book probabilities |
| `--nba-accent` | `#C47A3D` | `#D99155` | Hardwood maple |
| `--nfl-accent` | `#2E7A4C` | `#46A56B` | Turf green |
| `--tier-safe` | `#23897C` | `#3DB3A3` | Safe tier |
| `--tier-risky` | `#C98A12` | `#E3A93A` | Risky tier |
| `--tier-danger` | `#C43D3D` | `#E26464` | Dangerous tier |
| `--edge-pos` | `#23897C` | `#3DB3A3` | Model above book |
| `--edge-neg` | `#8A94A6` | `#6F7B8F` | Model below book (grey, not red — it's "no edge," not an error) |

The active league accent is exposed as `--accent` and swapped by a `data-league` attribute on `<html>`.

Typography:
- **Display & numbers: Barlow Condensed** (600/700). Condensed athletic lettering, like jersey numbers and scoreboards. Used for page titles, probabilities, odds, stakes.
- **Body & UI: Barlow** (400/500). Same family, clearly different width, so the system stays coherent.
- `font-variant-numeric: tabular-nums` on every number so columns of odds and percentages line up.
- Type scale: 12 / 14 / 16 / 20 / 28 / 40 / 56px. Body 16px, line-height 1.5; max line length ~70 characters.
- Sentence case everywhere. No all-caps labels, no tracked-out eyebrows.

Shape & depth:
- Radius by hierarchy: 14px for the bet slip and drawers, 8px for inputs/buttons, 999px only for small chips (tier badges, injury tags). Not one radius on everything.
- Depth from color steps (`--surface` → `--panel`) and 1px borders; a real shadow only on the bet slip so it feels like a physical ticket.

### 10.4 Screens

**A. Builder (home)**
```
┌─────────────────────────────────────────────────────────────┐
│ [NBA | NFL]  Tonight: 9 games          Bankroll $500  [edit] │
├───────────────┬─────────────────────────────┬───────────────┤
│ Tier          │  Suggested parlays           │  Bet slip     │
│ ○ Safe        │  ┌ Parlay A  3 legs  38% ┐   │  ┌─────────┐  │
│ ● Risky       │  └ EV +$0.22 · $8.50     ┘   │  │ leg 1   │  │
│ ○ Dangerous   │  ┌ Parlay B  4 legs  21% ┐   │  │ leg 2   │  │
│               │  └ EV +$0.31 · $5.00     ┘   │  │ leg 3   │  │
│ Filters       │  ...                         │  │ - - - - │  │
│ games, markets│                              │  │ 38% hit │  │
│               │  [Build my own]              │  │ $8.50   │  │
│               │                              │  └─────────┘  │
└───────────────┴─────────────────────────────┴───────────────┘
```
- Tier selector is a three-step control colored with tier tokens; switching tiers re-runs the search and swaps results.
- Each suggested parlay row: legs count, model hit %, payout, EV, suggested stake. Click → loads it into the bet slip.
- On mobile, the bet slip collapses into a bottom sheet with a summary bar ("3 legs · 38% · $8.50").

**B. Leg row (inside slip and lists)**
- Player headshot (ESPN/Sleeper headshot URLs; initials fallback), name, team, matchup, market ("Over 24.5 pts").
- **Probability bar:** a horizontal bar with two markers — model probability (solid, `--ink`) and book implied probability (hollow, `--muted`). The gap between them is shaded `--edge-pos` when the model is higher. In NFL mode the bar has yard-line tick marks every 10%.
- Injury chip if the player or a key teammate has a status.
- One-line reason in `--muted`.

**C. Player drawer** (slides in from the right on desktop, full-screen sheet on mobile)
- Game log bar chart, last 15 games, with the **prop line drawn across it**; bars that cleared the line in the league accent, others in `--muted`.
- Hit rate chips: L5, L10, season, vs this opponent.
- Distribution chart: the model's predicted distribution curve with the line marked and the "over" area filled.
- Minutes/snap share trend sparkline.
- Context block: opponent rank vs position, pace, rest/travel, refs (NBA), weather (NFL outdoor), teammates out.
- Odds across books with line-movement sparkline.

**D. Bet tracker**
- Log a bet (prefilled from the slip); mark won/lost/push.
- Bankroll line chart over time, ROI, units, hit rate by tier, average CLV.
- Loss-limit progress bar if a limit is set.

**E. Model performance**
- Calibration chart (predicted vs actual hit rate by bucket) per league and market.
- Backtest table: Brier score model vs book, ROI, CLV.
- Plain-language summary at the top: "Over the last 1,200 NBA props, legs the model rated 70% hit 68% of the time."

**F. Data sources & freshness**
- One row per source: status dot (fresh / late / failed), last updated, next scheduled pull, records pulled.
- Unmatched-player review queue from Section 4.2.

**G. Settings**
- Bankroll, Kelly fraction (advanced), tier thresholds (advanced), loss limits, theme, optional LLM features, API keys status (never display key values).

### 10.5 Motion
- **One signature moment:** when a parlay is loaded into the slip, legs drop in one after another, then the totals "print" at the bottom of the ticket. ~600ms total.
- Otherwise motion only responds to user actions (drawer open, tier switch cross-fade, toast on "Bet logged").
- Respect `prefers-reduced-motion` (instant state changes).

### 10.6 Copy
- Plain, specific, active: "Build parlays," "Add to slip," "Log bet," "Bet logged."
- Empty state (no games today): "No NBA games today. Next slate: Tue, Oct 20 — 11 games." with a button to switch leagues.
- Errors say what happened and what to do: "Odds didn't load from The Odds API (monthly credits used up). Showing lines from 2:14 PM."
- Probabilities always labeled by source: "Model 64%" vs "Book 55%."

### 10.7 Quality floor
- WCAG AA contrast in both themes (check tier colors on both surfaces).
- Visible keyboard focus rings; full keyboard navigation of builder and slip.
- Skeleton loaders shaped like the real content.
- Lighthouse performance ≥ 90 on the builder page.
- Test at 375px, 768px, 1280px, 1600px widths.

---

## 11. Tech stack & repo

- **Backend:** Python 3.11+, FastAPI, DuckDB + Parquet, polars/pandas, lightgbm, scikit-learn, scipy, APScheduler, pytest
- **Frontend:** Next.js + TypeScript, Tailwind, shadcn/ui, Recharts, Framer Motion, TanStack Query
- **Extras:** Java runtime (for `nbainjuries`), Docker Compose to run everything with one command
- **Optional LLM:** Anthropic API behind feature flags

```
parlay-builder/
├── docker-compose.yml
├── .env.example
├── config/
│   ├── sources.yaml
│   ├── tiers.yaml
│   ├── settings.yaml
│   └── venues.csv            # stadium/arena lat/long, altitude, roof, surface, tz
├── data/
│   ├── raw/                  # cached responses per source
│   └── warehouse.duckdb
├── backend/
│   ├── ingest/
│   │   ├── runner.py         # reads sources.yaml, handles fallback + logging
│   │   ├── nfl_nflverse.py
│   │   ├── nba_nbaapi.py
│   │   ├── nba_pbpstats.py
│   │   ├── nba_balldontlie.py
│   │   ├── sportsdataverse_espn.py
│   │   ├── injuries_nba_official.py
│   │   ├── sleeper.py
│   │   ├── weather_openmeteo.py
│   │   ├── odds_theoddsapi.py
│   │   └── odds_secondary.py
│   ├── xref/                 # player/team ID crosswalk + fuzzy matching
│   ├── validate/             # reconciliation + freshness checks
│   ├── features/
│   ├── models/
│   ├── parlay/
│   ├── bankroll/
│   ├── backtest/
│   ├── explain/              # optional LLM explanations + news flags
│   └── api/                  # FastAPI routes
├── frontend/
│   ├── app/                  # Next.js routes: builder, tracker, performance, sources, settings
│   ├── components/           # BetSlip, LegRow, ProbabilityBar, PlayerDrawer, TierSelector…
│   ├── lib/
│   └── styles/tokens.css
├── reports/
└── tests/
```

### Environment variables
```
ODDS_API_KEY=
SECONDARY_ODDS_API_KEY=
BALLDONTLIE_API_KEY=
ANTHROPIC_API_KEY=            # optional
ENABLE_LLM_EXPLANATIONS=false
ENABLE_LLM_NEWS_FLAGS=false
USE_MARKET_FEATURES=false
KELLY_FRACTION=0.25
MODEL_SHRINK_TO_MARKET=0.30
```

---

## 12. Build order

**Phase 1 — Data foundation (NBA)**
Source registry, ingestion runner with fallback, nba_api game logs (4–5 seasons), pbpstats possessions/lineups, official injury reports via nbainjuries, Sleeper player dump, schedules/venues/refs. Player ID crosswalk. Validation + freshness checks. Stop and summarize the data.

**Phase 2 — Features + models (NBA)**
Point-in-time feature builders with leakage tests; minutes model; per-market models; distributions; calibration; walk-forward backtest; feature ablation by source group.

**Phase 3 — Odds**
The Odds API + secondary source, snapshot store, name mapping, no-vig conversion, historical ROI/CLV backtest.

**Phase 4 — Parlays + sizing**
Correlations, Monte Carlo, beam search per tier, Kelly module with caps and EV ≤ 0 handling. Unit-test all math.

**Phase 5 — Frontend**
Design tokens and theme first, then components (ProbabilityBar, LegRow, BetSlip, TierSelector, PlayerDrawer), then pages (Builder → Player drawer → Tracker → Model performance → Sources → Settings). Screenshot review at all four breakpoints in both themes.

**Phase 6 — NFL**
nflreadpy (incl. NGS, FTN, snaps, participation, depth charts), Sleeper, Open-Meteo weather with historical forecasts, NFL features and models, NFL visual mode.

**Phase 7 — Polish**
Scheduler, optional LLM explanations and news flags, loss limits, Docker Compose one-command start.

---

## 13. Acceptance criteria

- [ ] `docker compose up` starts backend, frontend, and scheduler.
- [ ] Choose league + tier, enter bankroll, get 3–5 parlays with per-leg and parlay probabilities.
- [ ] Every source in `sources.yaml` shows status on the Data Sources page; failures fall back automatically.
- [ ] No unmatched player silently dropped; review queue works.
- [ ] Backtests use point-in-time data only (leakage tests pass).
- [ ] Feature ablation report exists; each source group's contribution is documented.
- [ ] Stake math follows Section 8; negative EV → $0.
- [ ] Parlay probabilities account for correlation.
- [ ] UI matches Section 10: league theming, bet slip, probability bars, player drawer, both themes, responsive, AA contrast, reduced motion respected.
- [ ] Responsible-gambling footer on every page.

---

## 14. First prompt to give Claude Code

> "Read PARLAY_BUILDER_SPEC.md fully. Set up the repo in Section 11 with Docker Compose. Start Phase 1: build the source registry and ingestion runner, then NBA ingestion from nba_api (4 seasons of player game logs), pbpstats (possessions with lineups, cached to disk), official injury reports via nbainjuries, and the Sleeper player dump. Build the player ID crosswalk and validation checks. Stop and show me a data summary, including any unmatched players, before Phase 2."
