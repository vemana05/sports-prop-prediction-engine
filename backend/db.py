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

SCHEMA = """
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
    source_published_at TIMESTAMP,
    ingested_at          TIMESTAMP NOT NULL,
    PRIMARY KEY (player_id_nba, game_id)
);

-- Possession-level data from pbpstats (live-data provider — see
-- backend/ingest/nba_pbpstats.py for why: pbpstats' stats_nba and data_nba
-- providers both hit dead NBA endpoints as of this writing). Only fields
-- verified safe to read for every possession are stored; on/off splits and
-- possession-based pace are derived from this in Phase 2 feature engineering
-- by aggregating on lineup_offense/lineup_defense.
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
    ingested_at         TIMESTAMP NOT NULL,
    PRIMARY KEY (game_id, possession_number)
);

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
"""


def get_connection(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    path = db_path or os.environ.get("DUCKDB_PATH", DEFAULT_DB_PATH)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(path)
    conn.execute(SCHEMA)
    return conn


def next_log_id(conn: duckdb.DuckDBPyConnection) -> int:
    return conn.execute("SELECT nextval('ingestion_log_seq')").fetchone()[0]
