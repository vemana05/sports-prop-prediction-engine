"""Point-in-time NBA feature builder (Section 5.2 "Shared" + minutes-relevant
NBA features).

Every rolling feature uses a window frame ending at "1 PRECEDING" — the
current game is never included in its own features. This is what makes the
table safe for training and backtesting: a row's features only ever depend
on games strictly before that row's `game_date` for that player.

Built from `nba_player_games` (already point-in-time — Section 4.4) joined
against `nba_schedule` for `season`. Feature set is deliberately the
"shared" + easy NBA slice from spec Section 5.2 to start: rolling
minutes/points/rebounds/assists at 3/5/10-game and season-to-date windows,
rest days, back-to-back flag, home/away. Opponent-defense-vs-position,
usage/tracking, referee, and on/off/pace features are NOT in this table —
the first three need more source data than we have wired into features
yet, and on/off/pace are explicitly deferred (spec Section 5.2).
"""

from __future__ import annotations

import duckdb

from backend.db import get_connection
from backend.ingest.util import IngestResult, logger, utcnow

FEATURE_TABLE = "nba_features"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {FEATURE_TABLE} (
    player_id_nba    VARCHAR NOT NULL,
    game_id          VARCHAR NOT NULL,
    game_date        DATE NOT NULL,
    season           VARCHAR NOT NULL,
    team             VARCHAR,
    opponent         VARCHAR,
    home_away        VARCHAR,
    rest_days        INTEGER,
    is_back_to_back  BOOLEAN,
    games_played_prior_season INTEGER,

    min_roll3        DOUBLE,
    min_roll5        DOUBLE,
    min_roll10       DOUBLE,
    min_roll_season  DOUBLE,
    min_std10        DOUBLE,

    pts_roll3        DOUBLE,
    pts_roll5        DOUBLE,
    pts_roll10       DOUBLE,
    pts_roll_season  DOUBLE,
    pts_std10        DOUBLE,

    reb_roll10       DOUBLE,
    ast_roll10       DOUBLE,
    fg3m_roll10      DOUBLE,
    stl_roll10       DOUBLE,
    blk_roll10       DOUBLE,
    tov_roll10       DOUBLE,

    -- actuals, kept alongside features for training/eval convenience only —
    -- a model must never be handed these as inputs, just as labels
    minutes          DOUBLE,
    pts              DOUBLE,
    reb              DOUBLE,
    ast              DOUBLE,
    fg3m             DOUBLE,
    stl              DOUBLE,
    blk              DOUBLE,
    tov              DOUBLE,

    ingested_at      TIMESTAMP NOT NULL,
    PRIMARY KEY (player_id_nba, game_id)
);
"""

_BUILD_SQL = f"""
INSERT OR REPLACE INTO {FEATURE_TABLE}
SELECT
    player_id_nba, game_id, game_date, season, team, opponent, home_away,
    DATEDIFF('day', LAG(game_date) OVER wplayer, game_date) AS rest_days,
    DATEDIFF('day', LAG(game_date) OVER wplayer, game_date) = 1 AS is_back_to_back,
    CAST(COUNT(*) OVER wseason AS INTEGER) AS games_played_prior_season,

    AVG(minutes) OVER w3 AS min_roll3,
    AVG(minutes) OVER w5 AS min_roll5,
    AVG(minutes) OVER w10 AS min_roll10,
    AVG(minutes) OVER wseason AS min_roll_season,
    STDDEV(minutes) OVER w10 AS min_std10,

    AVG(pts) OVER w3 AS pts_roll3,
    AVG(pts) OVER w5 AS pts_roll5,
    AVG(pts) OVER w10 AS pts_roll10,
    AVG(pts) OVER wseason AS pts_roll_season,
    STDDEV(pts) OVER w10 AS pts_std10,

    AVG(reb) OVER w10 AS reb_roll10,
    AVG(ast) OVER w10 AS ast_roll10,
    AVG(fg3m) OVER w10 AS fg3m_roll10,
    AVG(stl) OVER w10 AS stl_roll10,
    AVG(blk) OVER w10 AS blk_roll10,
    AVG(tov) OVER w10 AS tov_roll10,

    minutes, pts, reb, ast, fg3m, stl, blk, tov,
    ? AS ingested_at
FROM nba_player_games
WINDOW
    wplayer AS (PARTITION BY player_id_nba ORDER BY game_date, game_id),
    w3      AS (PARTITION BY player_id_nba ORDER BY game_date, game_id ROWS BETWEEN 3  PRECEDING AND 1 PRECEDING),
    w5      AS (PARTITION BY player_id_nba ORDER BY game_date, game_id ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING),
    w10     AS (PARTITION BY player_id_nba ORDER BY game_date, game_id ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
    wseason AS (PARTITION BY player_id_nba, season ORDER BY game_date, game_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
"""


def build(conn: duckdb.DuckDBPyConnection) -> IngestResult:
    conn.execute(_SCHEMA)
    conn.execute(f"DELETE FROM {FEATURE_TABLE}")
    conn.execute(_BUILD_SQL, [utcnow()])
    n = conn.execute(f"SELECT COUNT(*) FROM {FEATURE_TABLE}").fetchone()[0]
    logger.info("nba_features: built %d rows", n)
    return IngestResult(rows_written=n)


def run(conn: duckdb.DuckDBPyConnection | None = None) -> IngestResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        return build(conn)
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    result = run()
    print(f"nba_features build complete: {result.rows_written} rows")
