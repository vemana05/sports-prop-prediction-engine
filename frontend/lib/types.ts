export type League = "nba" | "nfl";
export type TierName = "safe" | "risky" | "dangerous";
export type Sides = "both" | "over" | "under";

export interface TierConfig {
  legs_min: number;
  legs_max: number;
  min_leg_prob: number;
  target_parlay_prob_min: number;
  target_parlay_prob_max: number;
  max_stake_pct_bankroll: number;
}

export interface Leg {
  id: string;
  player_id: string;
  player_name: string;
  team: string | null;
  opponent: string | null;
  is_home: boolean | null;
  game_id: string;
  tip_utc: string | null;
  market: string;
  side: "over" | "under";
  line: number;
  label: string;
  model_p: number;
  book_p: number | null;
  /** model_p shrunk 30% toward book_p — what tier eligibility uses */
  shrunk_p: number;
  decimal_odds: number;
  american_odds: number;
  predicted_mean: number;
  reason: string;
  headshot_url: string | null;
  injury: string | null;
}

export interface Parlay {
  id: string;
  name: string;
  legs: Leg[];
  p_hit: number;
  independent_p_hit: number;
  /** p_hit shrunk toward the book's no-vig parlay probability — what EV and stake use */
  p_sized: number;
  no_vig_p: number | null;
  book_p: number;
  decimal_odds: number;
  american_odds: number;
  ev_per_dollar: number;
  model_ev_per_dollar: number;
  stake: number;
  payout: number;
  capped_by_tier: boolean;
  reason: string | null;
  flagged: boolean;
  flag_reasons: string[];
  in_tier_band: boolean;
}

export interface SearchReport {
  n_pool: number;
  n_eligible: number;
  n_eligible_players: number;
  best_leg_p: number | null;
  n_combinations: number;
  combo_p_min: number | null;
  combo_p_max: number | null;
  n_in_range: number;
  reason: string | null;
}

export type SlateStatus = "idle" | "warming" | "ready" | "error" | "unavailable";

export interface SlateInfo {
  status: SlateStatus;
  error?: string | null;
  message?: string;
  n_games?: number;
  n_legs?: number;
  games?: GameInfo[];
  first_tip_utc?: string | null;
  odds_pulled_at?: string | null;
}

export interface ParlaysResponse extends SlateInfo {
  tier?: TierName;
  sides?: Sides;
  report?: SearchReport;
  parlays: Parlay[];
}

export interface ReplaceResponse {
  parlay: Parlay | null;
  added: Leg | null;
  message: string | null;
}

export type BetResult = "pending" | "won" | "lost" | "push";

export interface Bet {
  id: number;
  created_at: string;
  league: League;
  tier: TierName;
  name: string;
  legs: { player_name: string; label: string; american_odds: number }[];
  decimal_odds: number;
  american_odds: number;
  model_p: number;
  stake: number;
  paper: boolean;
  result: BetResult;
  settled_at: string | null;
  profit: number;
}

export interface Performance {
  as_of: string;
  tested_against_real_lines: boolean;
  real_line_note: string;
  live_markets: string[];
  walk_forward: {
    source: string;
    baseline_name: string;
    rows: { market: string; label: string; model_mae: number; naive_mae: number; used_live: "model" | "naive" }[];
  };
  calibration: {
    source: string;
    rows: { market: string; label: string; chart: string; chart_url: string | null; error_before: number | null; error_after: number | null }[];
  };
}

export interface GameInfo {
  game_id: string;
  teams: string[];
  tip_utc: string | null;
}
