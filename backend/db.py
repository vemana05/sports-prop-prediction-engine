"""Shared DuckDB connection + schema for Phase 1 (NBA data foundation).

Table design follows Section 4.4 (point-in-time correctness): every row that
represents a fact pulled from a source carries `source_published_at` (when
the source says the data was true / published) and `ingested_at` (when we
pulled it), so feature builders can later query "as of" a game's start time.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

DEFAULT_DB_PATH = "data/warehouse.duckdb"

# Full team roster per game (played AND DNP/inactive) from nba_api's
# BoxScoreTraditionalV3 — the only source we have that lists players who
# did NOT play, with a comment explaining why (e.g. "DNP - Coach's
# Decision", "NOT WITH TEAM", or blank for a true inactive/injury
# designation). This is the ground truth `nba_player_availability` (below)
# is built from — nba_player_games alone can't tell "DNP" from "not on the
# roster" (see the note on nba_player_games in SCHEMA below).
#
# Pulled out as its own constant (instead of inlined in SCHEMA) because the
# box_status backfill writes this same table into its own separate DuckDB
# file (see backend/ingest/nba_box_status.py) and needs the identical
# CREATE TABLE statement there — one definition, no drift.
NBA_BOX_STATUS_SCHEMA = """
CREATE TABLE IF NOT EXISTS nba_box_status (
    game_id          VARCHAR NOT NULL,
    team_id          VARCHAR NOT NULL,
    player_id_nba    VARCHAR NOT NULL,
    player_name_raw  VARCHAR,
    minutes_raw      VARCHAR,   -- "MM:SS" if played, NULL/blank if not
    comment          VARCHAR,   -- DNP/inactive reason if not played, blank if played
    ingested_at      TIMESTAMP NOT NULL,
    PRIMARY KEY (game_id, player_id_nba)
);
"""

SCHEMA = (
    """
CREATE TABLE IF NOT EXISTS ingestion_log (
    id              BIGINT PRIMARY KEY,
    source          VARCHAR NOT NULL,
    status          VARCHAR NOT NULL,       -- success | failed | fallback
    rows_written    BIGINT,
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    error           VARCHAR,
    fallback_from   VARCHAR
);

CREATE SEQUENCE IF NOT EXISTS ingestion_log_seq START 1;

-- IMPORTANT for Phase 2 feature engineering: this table comes from nba_api's
-- LeagueGameLog, which only includes players who recorded a box score line.
-- A player who checked in — even for a few seconds, rounding to minutes=0 —
-- DOES get a row here (verified: e.g. a player with a real "0:17" appearance
-- shows up with minutes=0). A player marked "DNP - Coach's Decision" or
-- inactive, who never checked in, gets NO row at all — there's no way to
-- distinguish "not on the active roster that game" from "on the roster, sat
-- the whole game" without joining a separate full-roster/inactive-list
-- source. Don't treat a missing row as "0 minutes" in rolling-average or
-- rest-day features without accounting for this gap.
CREATE TABLE IF NOT EXISTS nba_player_games (
    player_id_nba       VARCHAR NOT NULL,
    player_name         VARCHAR NOT NULL,
    team                VARCHAR,
    opponent            VARCHAR,
    game_id             VARCHAR NOT NULL,
    game_date           DATE NOT NULL,
    season              VARCHAR NOT NULL,
    home_away           VARCHAR,
    minutes             DOUBLE,
    pts                 DOUBLE,
    reb                 DOUBLE,
    ast                 DOUBLE,
    fg3m                DOUBLE,
    stl                 DOUBLE,
    blk                 DOUBLE,
    tov                 DOUBLE,
    fga                 DOUBLE,
    fgm                 DOUBLE,
    fta                 DOUBLE,
    ftm                 DOUBLE,
    oreb                DOUBLE,
    dreb                DOUBLE,
    source_published_at TIMESTAMP,
    ingested_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (player_id_nba, game_id)
);

-- Migration: fga/fgm/fta/ftm/oreb/dreb added after the table already
-- existed (needed for usage rate and a real pace estimate — Section 5.7).
-- Same free LeagueGameLog call nba_nbaapi.py already makes just needed to
-- select a few more columns; see that module's re-ingestion.
ALTER TABLE nba_player_games ADD COLUMN IF NOT EXISTS fga DOUBLE;
ALTER TABLE nba_player_games ADD COLUMN IF NOT EXISTS fgm DOUBLE;
ALTER TABLE nba_player_games ADD COLUMN IF NOT EXISTS fta DOUBLE;
ALTER TABLE nba_player_games ADD COLUMN IF NOT EXISTS ftm DOUBLE;
ALTER TABLE nba_player_games ADD COLUMN IF NOT EXISTS oreb DOUBLE;
ALTER TABLE nba_player_games ADD COLUMN IF NOT EXISTS dreb DOUBLE;

-- Possession-level data from pbpstats (live-data provider — see
-- backend/ingest/nba_pbpstats.py for why: pbpstats' stats_nba and data_nba
-- providers both hit dead NBA endpoints as of this writing). Only fields
-- verified safe to read for every possession are stored. On/off splits and
-- possession-based pace are DEFERRED past Phase 2 (see spec Section 5.2) —
-- only ~207 games (all early 2025-26) are ingested so far, out of 6,150
-- games across 5 seasons, nowhere near enough for statistically meaningful
-- on/off pairs. Build those from lineup_offense/lineup_defense once a
-- full-season possession backfill exists.
CREATE TABLE IF NOT EXISTS nba_possessions (
    game_id             VARCHAR NOT NULL,
    season              VARCHAR NOT NULL,
    possession_number   INTEGER NOT NULL,   -- sequential index within the game
    period              INTEGER,
    offense_team_id     VARCHAR,
    defense_team_id     VARCHAR,
    start_time          VARCHAR,   -- clock MM:SS remaining in period, from pbpstats
    end_time            VARCHAR,
    start_score_margin  INTEGER,   -- offense team's point differential at possession start
    lineup_offense      VARCHAR,   -- 5 player ids, dash-separated
    lineup_defense      VARCHAR,
    source_published_at DATE,      -- the game's date (from nba_player_games), for point-in-time queries
    ingested_at         TIMESTAMP NOT NULL,
    PRIMARY KEY (game_id, possession_number)
);

-- Migration for tables created before source_published_at existed (DuckDB
-- has no "ADD COLUMN IF NOT EXISTS" pre-table-creation ordering issue since
-- this runs after the CREATE TABLE IF NOT EXISTS above either creates the
-- table fresh with the column already present, or leaves an existing table
-- untouched for this ALTER to extend).
ALTER TABLE nba_possessions ADD COLUMN IF NOT EXISTS source_published_at DATE;

CREATE TABLE IF NOT EXISTS nba_injuries (
    player_name_raw      VARCHAR NOT NULL,
    player_id_nba        VARCHAR,          -- filled in after xref match, nullable
    team                 VARCHAR,
    game_date             DATE,
    report_time           TIMESTAMP NOT NULL,   -- snapshot timestamp (point-in-time)
    status                VARCHAR,              -- Out / Doubtful / Questionable / Probable / Available
    reason                VARCHAR,
    source_published_at   TIMESTAMP,
    ingested_at            TIMESTAMP NOT NULL,
    PRIMARY KEY (player_name_raw, team, report_time)
);

-- Tip-off times (Section 4.4 point-in-time correctness): lets feature/
-- injury-matching code check whether a given report_time is actually
-- before a specific game started, not just on the same calendar day.
CREATE TABLE IF NOT EXISTS nba_schedule (
    game_id        VARCHAR PRIMARY KEY,
    season         VARCHAR NOT NULL,
    game_date      DATE NOT NULL,
    home_team_id   VARCHAR NOT NULL,
    away_team_id   VARCHAR NOT NULL,
    tip_utc        TIMESTAMPTZ NOT NULL,
    ingested_at    TIMESTAMP NOT NULL
);
"""
    + NBA_BOX_STATUS_SCHEMA
    + """
-- Derived, point-in-time player availability per team-game (Section 5.2
-- "games since return" / teammate-availability features read this, not
-- nba_player_games directly). Built by backend/ingest/nba_availability.py
-- from nba_box_status (played vs. DNP) joined against nba_injuries (was a
-- DNP player listed on the injury report published before this game's
-- tip-off, per nba_schedule?).
CREATE TABLE IF NOT EXISTS nba_player_availability (
    game_id          VARCHAR NOT NULL,
    season           VARCHAR NOT NULL,
    team_id          VARCHAR NOT NULL,
    player_id_nba    VARCHAR NOT NULL,
    status           VARCHAR NOT NULL,   -- played | dnp_listed_injury_report | dnp_other
    detail           VARCHAR,            -- box score comment or injury report status/reason
    ingested_at      TIMESTAMP NOT NULL,
    PRIMARY KEY (game_id, player_id_nba)
);

CREATE TABLE IF NOT EXISTS sleeper_players (
    sleeper_id           VARCHAR PRIMARY KEY,
    full_name            VARCHAR,
    team                 VARCHAR,
    position              VARCHAR,
    status                VARCHAR,
    injury_status         VARCHAR,
    depth_chart_order      INTEGER,
    search_rank            INTEGER,
    ingested_at             TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS player_xref (
    player_id       VARCHAR PRIMARY KEY,   -- internal id, currently == nba_player_id
    nba_player_id   VARCHAR,
    sleeper_id      VARCHAR,
    full_name       VARCHAR NOT NULL,
    match_method    VARCHAR,   -- exact | fuzzy | manual_override
    match_score     DOUBLE,
    created_at      TIMESTAMP NOT NULL,
    updated_at      TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS xref_review_queue (
    source          VARCHAR NOT NULL,
    raw_name        VARCHAR NOT NULL,
    raw_id          VARCHAR,
    reason          VARCHAR NOT NULL,
    best_candidate  VARCHAR,
    best_score      DOUBLE,
    created_at      TIMESTAMP NOT NULL,
    resolved        BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (source, raw_name, raw_id)
);

-- Free-tier forward odds collection (The Odds API, Section 5.7). Raw player
-- points prop lines, one row per (event, player, bookmaker, pull). Kept
-- even across repeated pulls of the same not-yet-started game so line
-- movement is visible, not just the latest snapshot.
CREATE TABLE IF NOT EXISTS odds_snapshots (
    pulled_at        TIMESTAMP NOT NULL,
    event_id         VARCHAR NOT NULL,       -- The Odds API's event id
    game_id_nba      VARCHAR,                -- resolved against nba_schedule where possible, nullable
    commence_time     TIMESTAMPTZ,
    player_name_raw  VARCHAR NOT NULL,
    market            VARCHAR NOT NULL,       -- e.g. "player_points"
    bookmaker         VARCHAR NOT NULL,
    line              DOUBLE,
    over_price        DOUBLE,                 -- decimal odds, matches Section 8's EV formula (d)
    under_price       DOUBLE,
    PRIMARY KEY (pulled_at, event_id, player_name_raw, market, bookmaker)
);

-- The two teams of each odds event (The Odds API's home_team/away_team,
-- as 3-letter abbreviations). A leg's team must be one of these and its
-- opponent is the other (backend/parlay/leg_pool.py).
CREATE TABLE IF NOT EXISTS odds_events (
    event_id         VARCHAR PRIMARY KEY,
    commence_time    TIMESTAMPTZ,
    home_team        VARCHAR,
    away_team        VARCHAR,
    updated_at       TIMESTAMP NOT NULL
);

-- Final scores and status of each scheduled game, refreshed by
-- backend/ingest/nba_daily_games.py (gameStatus 3 = final).
CREATE TABLE IF NOT EXISTS nba_game_results (
    game_id          VARCHAR PRIMARY KEY,
    season           VARCHAR NOT NULL,
    game_status      INTEGER,
    home_score       INTEGER,
    away_score       INTEGER,
    updated_at       TIMESTAMP NOT NULL
);

-- One row per call to The Odds API, straight from its own rate-limit
-- headers (x-requests-used / x-requests-remaining) — this IS the credit
-- budget guard's source of truth, not a locally-computed estimate, since
-- the API tracks usage server-side per key per billing month itself.
CREATE TABLE IF NOT EXISTS odds_api_usage_log (
    called_at           TIMESTAMP NOT NULL,
    endpoint            VARCHAR NOT NULL,
    requests_used        INTEGER,
    requests_remaining   INTEGER,
    status              VARCHAR NOT NULL,     -- success | skipped_budget | error
    note                VARCHAR,
    PRIMARY KEY (called_at, endpoint)
);

-- Every model prediction logged before a game, for paper-trading against
-- real lines once odds_snapshots has them — the free-tier substitute for a
-- real CLV/ROI backtest (Section 3.5 / 5.7). Logged once per (game,
-- player, market); never overwritten, so a prediction can't quietly drift
-- after the fact.
CREATE TABLE IF NOT EXISTS model_predictions_log (
    predicted_at      TIMESTAMP NOT NULL,
    game_id           VARCHAR NOT NULL,
    player_id_nba     VARCHAR NOT NULL,
    market            VARCHAR NOT NULL,        -- points | rebounds | assists | fg3m | minutes
    predicted_mean    DOUBLE NOT NULL,
    dispersion_param  DOUBLE,                   -- sigma (Normal) or r (NegBinom), per market's distribution
    model_version      VARCHAR,
    PRIMARY KEY (game_id, player_id_nba, market)
);
"""
)


def get_connection(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    path = db_path or os.environ.get("DUCKDB_PATH", DEFAULT_DB_PATH)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(path)
    conn.execute(SCHEMA)
    return conn


def next_log_id(conn: duckdb.DuckDBPyConnection) -> int:
    return conn.execute("SELECT nextval('ingestion_log_seq')").fetchone()[0]
