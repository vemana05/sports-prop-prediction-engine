"""Separate DuckDB warehouse for NFL (Phase 6), so NFL ingestion and
training never contend for warehouse.duckdb's write lock with the daily NBA
odds job (see spec Section 5.8's lock-contention fix).

Same point-in-time conventions as backend/db.py (Section 4.4). nflverse
publishes weekly files rather than timestamped records, so most tables key
on (season, week) instead of a published-at timestamp: a week's injury
report and depth chart are published before that week's games, and player
stats only after. Feature builders must only read stats from strictly
earlier games — see backend/validate/test_nfl_features.py.

Betting columns that nflverse's schedule file carries (spread_line,
total_line, moneylines) are deliberately NOT stored: USE_MARKET_FEATURES is
false (Section 5.2), so nothing from a sportsbook belongs in the feature
warehouse.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

DEFAULT_NFL_DB_PATH = "data/nfl_warehouse.duckdb"

NFL_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_log (
    id              BIGINT PRIMARY KEY,
    source          VARCHAR NOT NULL,
    status          VARCHAR NOT NULL,
    rows_written    BIGINT,
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    error           VARCHAR,
    fallback_from   VARCHAR
);
CREATE SEQUENCE IF NOT EXISTS ingestion_log_seq START 1;

-- One row per game. kickoff_utc is derived from nflverse's gameday +
-- gametime, which are US Eastern. roof is nflverse's value, or the venue's
-- default from config/nfl_venues.csv when nflverse hasn't filled it in yet
-- (common for future games).
CREATE TABLE IF NOT EXISTS nfl_schedule (
    game_id        VARCHAR PRIMARY KEY,
    season         INTEGER NOT NULL,
    week           INTEGER NOT NULL,
    game_type      VARCHAR NOT NULL,     -- REG | WC | DIV | CON | SB
    kickoff_utc    TIMESTAMPTZ,
    home_team      VARCHAR NOT NULL,
    away_team      VARCHAR NOT NULL,
    home_score     INTEGER,
    away_score     INTEGER,
    home_rest      INTEGER,
    away_rest      INTEGER,
    div_game       BOOLEAN,
    roof           VARCHAR,
    surface        VARCHAR,
    stadium_id     VARCHAR,
    stadium        VARCHAR,
    venue_id       VARCHAR,              -- config/nfl_venues.csv key (matched by stadium name first)
    ingested_at    TIMESTAMP NOT NULL
);

-- Weekly offensive player stats (nflverse load_player_stats). Only players
-- who recorded a stat line appear; a player who was active but touched
-- nothing has no row (same gap as nba_player_games — see backend/db.py).
CREATE TABLE IF NOT EXISTS nfl_player_weekly (
    player_id          VARCHAR NOT NULL,   -- gsis id, e.g. 00-0033873
    player_name        VARCHAR,
    position           VARCHAR,
    position_group     VARCHAR,
    season             INTEGER NOT NULL,
    week               INTEGER NOT NULL,
    season_type        VARCHAR,
    game_id            VARCHAR NOT NULL,
    team               VARCHAR,
    opponent           VARCHAR,
    completions        DOUBLE,
    attempts           DOUBLE,
    passing_yards      DOUBLE,
    passing_tds        DOUBLE,
    interceptions      DOUBLE,
    sacks              DOUBLE,
    passing_air_yards  DOUBLE,
    passing_epa        DOUBLE,
    carries            DOUBLE,
    rushing_yards      DOUBLE,
    rushing_tds        DOUBLE,
    rushing_epa        DOUBLE,
    receptions         DOUBLE,
    targets            DOUBLE,
    receiving_yards    DOUBLE,
    receiving_tds      DOUBLE,
    receiving_air_yards DOUBLE,
    target_share       DOUBLE,
    air_yards_share    DOUBLE,
    ingested_at        TIMESTAMP NOT NULL,
    PRIMARY KEY (player_id, game_id)
);

-- Snap counts come keyed by PFR id; nfl_rosters_weekly carries the
-- pfr_id -> gsis_id crosswalk (filled into player_id at load time).
CREATE TABLE IF NOT EXISTS nfl_snap_counts (
    game_id         VARCHAR NOT NULL,
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    pfr_player_id   VARCHAR NOT NULL,
    player_id       VARCHAR,              -- gsis id via rosters, nullable if unmatched
    player_name     VARCHAR,
    position        VARCHAR,
    team            VARCHAR,
    offense_snaps   DOUBLE,
    offense_pct     DOUBLE,               -- 0..1
    ingested_at     TIMESTAMP NOT NULL,
    PRIMARY KEY (game_id, pfr_player_id)
);

-- Final weekly game-status designation (published before the week's games).
CREATE TABLE IF NOT EXISTS nfl_injuries (
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    game_type       VARCHAR,
    team            VARCHAR NOT NULL,
    player_id       VARCHAR NOT NULL,
    player_name     VARCHAR,
    position        VARCHAR,
    report_status   VARCHAR,              -- Out | Doubtful | Questionable | NULL
    practice_status VARCHAR,
    primary_injury  VARCHAR,
    ingested_at     TIMESTAMP NOT NULL,
    PRIMARY KEY (season, week, team, player_id)
);

-- Offensive skill-position depth charts, normalized across nflverse's two
-- formats: through 2024 one chart per (season, week); from 2025 on, dated
-- snapshots (`snapshot_at`, several per week) with no week number. Feature
-- code picks the latest chart published before kickoff.
CREATE TABLE IF NOT EXISTS nfl_depth_charts (
    season          INTEGER NOT NULL,
    week            INTEGER,              -- pre-2025 format only
    snapshot_at     TIMESTAMPTZ,          -- 2025+ format only
    team            VARCHAR NOT NULL,
    player_id       VARCHAR NOT NULL,
    player_name     VARCHAR,
    position        VARCHAR NOT NULL,     -- QB | RB | WR | TE
    depth_rank      INTEGER NOT NULL      -- 1 = starter
);

CREATE TABLE IF NOT EXISTS nfl_rosters_weekly (
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    team            VARCHAR NOT NULL,
    player_id       VARCHAR NOT NULL,
    player_name     VARCHAR,
    position        VARCHAR,
    status          VARCHAR,
    pfr_id          VARCHAR,
    sleeper_id      VARCHAR,
    years_exp       INTEGER,
    rookie_year     INTEGER,
    ingested_at     TIMESTAMP NOT NULL,
    PRIMARY KEY (season, week, team, player_id)
);

-- Kickoff-hour weather for outdoor games (Open-Meteo). Played games use
-- the Historical Forecast archive (what forecasts said, not what was
-- observed — Section 3.4); upcoming games use the live forecast. Indoor
-- games get no row.
CREATE TABLE IF NOT EXISTS nfl_weather (
    game_id         VARCHAR PRIMARY KEY,
    kickoff_utc     TIMESTAMPTZ NOT NULL,
    temperature_c   DOUBLE,
    wind_kph        DOUBLE,
    precip_mm       DOUBLE,
    source          VARCHAR NOT NULL,     -- historical_forecast | forecast
    fetched_at      TIMESTAMP NOT NULL
);

-- Every NFL projection logged before kickoff (backend/predict/log_nfl_predictions.py),
-- the NFL counterpart of warehouse.duckdb's model_predictions_log: one row
-- per (game, player, market), never overwritten, so the projection on record
-- is the earliest one and can't drift toward the line or the result.
CREATE TABLE IF NOT EXISTS nfl_predictions_log (
    logged_at       TIMESTAMP NOT NULL,
    game_id         VARCHAR NOT NULL,
    player_id       VARCHAR NOT NULL,     -- gsis id
    market          VARCHAR NOT NULL,     -- passing_yards | rushing_yards | receiving_yards | receptions
    mean            DOUBLE NOT NULL,
    lo              DOUBLE,               -- 25th percentile
    hi              DOUBLE,               -- 75th percentile
    model_version   VARCHAR NOT NULL,     -- e.g. thru2025
    PRIMARY KEY (game_id, player_id, market)
);

-- OddsPapi forward collection (backend/ingest/odds_oddspapi_nfl.py). These
-- are sportsbook lines kept for evaluation against model projections, never
-- read by feature builders (USE_MARKET_FEATURES is false, Section 5.2).
--
-- Request accounting only: request_count / request_limit as read from
-- GET /account (never its body), plus one row per paid request this
-- collector made, carrying the running estimate.
CREATE TABLE IF NOT EXISTS oddspapi_usage_log (
    called_at       TIMESTAMP NOT NULL,
    endpoint        VARCHAR NOT NULL,
    request_count   INTEGER,              -- account count after this call (reported or estimated)
    request_limit   INTEGER,
    source          VARCHAR NOT NULL,     -- account | estimate
    status          VARCHAR NOT NULL,
    note            VARCHAR
);

-- One row per OddsPapi fixture we collected, matched to nflverse's game_id.
CREATE TABLE IF NOT EXISTS nfl_oddspapi_fixtures (
    fixture_id      VARCHAR PRIMARY KEY,
    game_id         VARCHAR,
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    kickoff_utc     TIMESTAMPTZ NOT NULL,
    home_team       VARCHAR,
    away_team       VARCHAR,
    odds_fetched_at TIMESTAMP,            -- Wednesday /odds pull
    hist_fetched_at TIMESTAMP             -- Tuesday /historical-odds pull
);

-- OddsPapi playerId -> name, saved from /odds before kickoff (a finished
-- fixture's /odds no longer carries names, and /historical-odds never does).
CREATE TABLE IF NOT EXISTS nfl_oddspapi_players (
    fixture_id      VARCHAR NOT NULL,
    op_player_id    VARCHAR NOT NULL,
    player_name     VARCHAR NOT NULL,
    player_id       VARCHAR,              -- gsis id when the name matches one roster player on either team
    PRIMARY KEY (fixture_id, op_player_id)
);

-- One main line per (fixture, book, market, player, kind). kind 'open' is
-- the line as of the Wednesday /odds pull; 'close' is rebuilt on Tuesday
-- from /historical-odds as the last pre-kickoff two-sided line.
CREATE TABLE IF NOT EXISTS nfl_prop_lines (
    fixture_id      VARCHAR NOT NULL,
    game_id         VARCHAR,
    bookmaker       VARCHAR NOT NULL,
    market          VARCHAR NOT NULL,     -- passing_yards | rushing_yards | receiving_yards | receptions
    op_player_id    VARCHAR NOT NULL,
    kind            VARCHAR NOT NULL,     -- open | close
    line            DOUBLE NOT NULL,
    over_price      DOUBLE,               -- decimal
    under_price     DOUBLE,
    observed_at     TIMESTAMPTZ,
    fetched_at      TIMESTAMP NOT NULL,
    PRIMARY KEY (fixture_id, bookmaker, market, op_player_id, kind)
);
"""


def get_nfl_connection(db_path: str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = db_path or os.environ.get("NFL_DUCKDB_PATH", DEFAULT_NFL_DB_PATH)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if read_only:
        return duckdb.connect(path, read_only=True)
    conn = duckdb.connect(path)
    conn.execute(NFL_SCHEMA)
    return conn
