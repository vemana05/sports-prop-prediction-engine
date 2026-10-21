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
  /** null for an NFL leg with no typed price */
  decimal_odds: number | null;
  american_odds: number | null;
  predicted_mean: number;
  reason: string;
  headshot_url: string | null;
  injury: string | null;
  // NFL only
  base_id?: string;
  position?: string;
  rung?: number | null;
  priced?: boolean;
  vig_removed?: boolean;
  under_american?: number | null;
  over_american?: number | null;
  warnings?: string[];
}

/** NFL Build my own: one projected player and each market's ladder. */
export interface NflPickPlayer {
  player_id: string;
  player_name: string;
  position: string;
  team: string;
  opponent?: string;
  game_id?: string;
  matchup?: string;
  tip_utc?: string | null;
  injury: string | null;
  out: boolean;
  markets: Record<string, { mean: number; ladder: { rung: number; line: number; p_over: number }[] }>;
}

export interface NflPlayers {
  status: string;
  message?: string;
  week?: number;
  players: NflPickPlayer[];
  out: NflPickPlayer[];
  markets: string[];
  market_labels: Record<string, string>;
  untested_note: string;
}

/** A sportsbook price typed into the slip for one NFL leg. */
export interface LegPrice {
  line?: string;
  over?: number;
  under?: number;
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
  /** null when any NFL leg is unpriced */
  book_p: number | null;
  decimal_odds: number | null;
  american_odds: number | null;
  ev_per_dollar: number | null;
  model_ev_per_dollar: number | null;
  stake: number;
  payout: number;
  capped_by_tier: boolean;
  reason: string | null;
  flagged: boolean;
  flag_reasons: string[];
  in_tier_band: boolean;
  // NFL only
  priced?: boolean;
  n_priced?: number;
  fair_american_odds?: number | null;
  notes?: string[];
  warnings?: string[];
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
  /** NBA: ET date (YYYY-MM-DD) of the earliest upcoming tip */
  slate_date?: string | null;
  odds_pulled_at?: string | null;
  // NFL only
  season?: number;
  week?: number;
  weeks?: NflWeekInfo[];
  injuries_posted?: boolean;
  calibrated?: boolean;
  untested_note?: string;
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
  bonus: boolean;
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
  nfl: NflPerformance | null;
}

export interface NflPerformanceRow {
  market: NflMarket;
  label: string;
  n: number;
  model_mae: number;
  last5_mae: number;
  season_mae: number;
  shrunk_mae: number;
  no_history: { n: number; model_mae: number; position_prior_mae: number } | null;
  error_before: number;
  error_after: number;
  chart_url: string | null;
}

export interface NflPerformance {
  test_seasons: number[];
  rows: NflPerformanceRow[];
}

export interface LinesBucket { lo: number; hi: number; n: number; mean_p: number; hit_rate: number }
export interface LinesRoi { margin: number; bets: number; wins: number; pushes: number; profit: number; roi: number | null }

export interface LinesSummary {
  props: number;
  graded: number;
  pushes: number;
  enough_sample: boolean;
  brier_model: number | null;
  brier_model_raw: number | null;
  brier_book: number | null;
  over_rate: number | null;
  side_hit_rate: number | null;
  side_n: number;
  calibration_model: LinesBucket[];
  calibration_book: LinesBucket[];
  roi: LinesRoi[];
  clv: { n: number; beat: number; same: number; worse: number; beat_rate: number | null; mean_line_move: number | null };
}

export interface ModelVsLines {
  available: boolean;
  as_of?: string;
  book_rule?: string;
  match_rule?: string;
  min_sample?: number;
  margins?: number[];
  books_used?: Record<string, number>;
  names?: {
    total: number;
    matched: number;
    match_rate: number | null;
    by_method: Record<string, number>;
    unmatched: { name: string; game_id: string; teams: string; reason: string }[];
  };
  coverage?: Record<string, number>;
  not_graded?: { name: string; player_id: string; game_id: string; market: NflMarket; status: string }[];
  overall?: LinesSummary;
  markets?: Record<NflMarket, LinesSummary & { label: string }>;
  summary?: string[];
}

export interface JobStatus {
  key: string;
  label: string;
  schedule: string;
  script: string;
  last_scheduled: string;
  next_scheduled: string;
  runs_logged: number;
  outcome: "success" | "success_with_errors" | "failed" | "running" | "never ran";
  log: string | null;
  started_at: string | null;
  finished_at: string | null;
  duration_s: number | null;
  last_error: string | null;
  late: boolean;
}

export type QueueCategory = "game_logs" | "college" | "duplicate" | "team";

export interface ApiUsage { name: string; limit: number; budget: number; used: number | null; last_call: string | null; note: string | null }

export interface TableStatus { table: string; rows: number; last_updated: string | null; time_column: string | null; cadence_days: number | null; late: boolean }

export interface DataSources {
  as_of: string;
  flags: string[];
  jobs: JobStatus[];
  usage: ApiUsage[];
  warehouses: { name: string; path: string; error: string | null; tables: TableStatus[] }[];
  models: { league: string; name: string; version: string | null; markets: string[]; modified: string }[];
  review_queue: {
    total: number;
    by_source: Record<string, number>;
    by_category: Record<QueueCategory, number>;
    category_labels: Record<QueueCategory, string>;
    truncated: boolean;
    rows: { source: string; name: string; raw_id: string | null; reason: string; best_candidate: string | null; best_score: number | null; created_at: string | null; context: string | null; category: QueueCategory }[];
  };
}

export type NflMarket = "passing_yards" | "rushing_yards" | "receiving_yards" | "receptions";

export interface NflProjection {
  mean: number;
  lo: number;
  hi: number;
  actual: number | null;
}

export interface NflPlayer {
  player_id: string;
  player_name: string;
  position: "QB" | "RB" | "WR" | "TE";
  depth_rank: number | null;
  questionable: boolean;
  projections: Partial<Record<NflMarket, NflProjection>>;
}

export interface NflTeam {
  team: string;
  opponent: string;
  players: NflPlayer[];
}

export interface NflGame {
  game_id: string;
  kickoff_utc: string | null;
  home: string;
  away: string;
  final: boolean;
  home_score: number | null;
  away_score: number | null;
  stadium: string | null;
  roof: string | null;
  indoor: boolean;
  weather: { temperature_c: number; wind_kph: number; precip_mm: number; source: string } | null;
  rest: { home: number | null; away: number | null };
  teams: { away: NflTeam; home: NflTeam };
}

export interface NflWeekInfo {
  week: number;
  label: string;
  first_kickoff_utc: string | null;
  last_kickoff_utc: string | null;
  n_games: number;
  n_final: number;
}

/** Sent instead of NflWeek while the NFL models train in the background. */
export interface NflWarming {
  status: "warming" | "error";
  league: "nfl";
  message: string;
}

export interface NflWeek {
  status: "ready";
  season: number;
  week: number | null;
  weeks: NflWeekInfo[];
  computed_at: string;
  models_trained_through: number;
  /** Saved model that projected this week: thru2025, or thru2026w03 (2026 through week 3). */
  model_version: string;
  model_label: string;
  injuries_posted: boolean;
  games: NflGame[];
}

export interface GameInfo {
  game_id: string;
  teams: string[];
  /** NBA: "AWAY @ HOME" once the odds event is matched to the schedule */
  label?: string | null;
  away?: string | null;
  home?: string | null;
  tip_utc: string | null;
}

export interface Projection {
  mean: number;
  lo: number;
  hi: number;
}

export type StatKey = "minutes" | "points" | "rebounds" | "assists" | "fg3m";

export interface ProjectedPlayer {
  player_id: string;
  player_name: string;
  last_game_date: string;
  opp_def_vs_position_pts: number | null;
  projections: Record<StatKey, Projection>;
}

export interface TeamRest {
  rest_days: number | null;
  back_to_back: boolean;
  season_opener: boolean;
}

export interface TeamProjection {
  team: string;
  opponent: string;
  opp_pace: number | null;
  opp_pace_rank: number | null;
  n_teams_ranked: number;
  players: ProjectedPlayer[];
}

export interface ScheduledGame {
  game_id: string;
  tip_utc: string;
  home: string;
  away: string;
  rest: { home: TeamRest; away: TeamRest };
  teams: { home: TeamProjection; away: TeamProjection };
  builder_game_id: string | null;
  lines_posted: boolean;
}

export interface ScheduleDay {
  status: "ready" | "unavailable";
  message?: string;
  today: string;
  date: string;
  computed_at: string;
  form_as_of: string;
  prev_game_day: string | null;
  next_game_day: string | null;
  games: ScheduledGame[];
}

// ---- Replay (GET /api/replay) ----
export interface ReplaySummary {
  n: number;
  model_mae: number;
  naive_mae: number | null;
  model_mae_same_rows: number | null;
  n_naive: number;
  coverage: number;
  bias: number;
}

export interface ReplayRow {
  player_name: string;
  team: string | null;
  opponent: string | null;
  predicted: number;
  lo: number;
  hi: number;
  actual: number;
  miss: number;
  inside: boolean;
  naive: number | null;
}

export interface ReplayOption {
  key: string;
  label: string;
}

export type Replay =
  | { status: "unavailable"; league: League; detail: string }
  | {
      status: "ready";
      league: League;
      period: ReplayOption;
      periods: ReplayOption[];
      market: ReplayOption;
      markets: ReplayOption[];
      summary: ReplaySummary | null;
      biggest_misses: ReplayRow[];
      rows: ReplayRow[];
      rolling: (ReplaySummary & { label: string; model_version?: string | null })[];
      weeks: number;
      /** NFL: the saved model version(s) that projected this week, e.g. thru2026w03. */
      model_version?: string | null;
      source: string;
      baseline_name: string;
      naive_is_model: boolean;
      note: string;
    };

/** NBA early-season window: until every team has played `threshold` games. */
export interface EarlySeason {
  as_of: string;
  active: boolean;
  season: string | null;
  threshold: number;
  min_games_played: number | null;
  games_remaining: number;
  teams_below: string[];
  games_played: Record<string, number>;
}

export type LimitPeriod = "daily" | "weekly" | "monthly";
export type LimitLevel = "none" | "ok" | "warn" | "hit";

export interface LossLimit {
  period: LimitPeriod;
  limit: number | null;
  pending: { amount: number | null; effective_at: string } | null;
  used: number | null;
  level: LimitLevel;
  start: string;
  end: string;
  lost: number;
  won: number;
  net_loss: number;
  bonus_bets: number;
  bonus_winnings: number;
}

export interface LimitStatus {
  now: string;
  limits: LossLimit[];
  level: "ok" | "warn" | "hit";
  hit: LimitPeriod[];
  paused_until: string | null;
  warn_at: number;
  raise_delay_hours: number;
  pause_days: number[];
}

export interface AppSettings {
  kelly_fraction: number;
  kelly_default: number;
  tiers: Record<TierName, TierConfig>;
  api_keys: { name: string; label: string; status: "set" | "missing" }[];
  llm_features: { name: string; label: string; enabled: boolean }[];
}
