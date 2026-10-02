"""Point-in-time NBA feature builder (Section 5.2 "Shared" + NBA features).

Every rolling feature uses a window frame ending at "1 PRECEDING" — the
current game is never included in its own features. This is what makes the
table safe for training and backtesting: a row's features only ever depend
on games strictly before that row's `game_date` for that player (or team,
for the team-level features below).

Built from `nba_player_games` (already point-in-time — Section 4.4).
Feature set: rolling minutes/points/rebounds/assists/3PM/steals/blocks/
turnovers at 3/5/10-game and season-to-date windows, rest days,
back-to-back flag, home/away — plus, as of this build, three team-context
features that don't depend on the nba_box_status availability backfill:

- **Usage rate** (`usage_rate_roll10`): the standard box-score formula —
  USG% = 100*((FGA+0.44*FTA+TOV)*(TeamMin/5)) / (Min*(TeamFGA+0.44*TeamFTA+TeamTOV))
  — computed per game from the real FGA/FTA/TOV/minutes nba_nbaapi.py now
  pulls, then rolled forward over the player's own last 10 games (so a
  usage feature is itself point-in-time, same as every other roll10).
- **Team / opponent pace** (`team_pace_roll10` / `opp_pace_roll10`): a
  single-team pace proxy (FGA - OREB + TOV + 0.44*FTA, the standard
  individual-team possession estimate from box-score totals — no
  opponent-symmetric term, since that needs defensive rebound context from
  BOTH teams, more than a box-score pace proxy needs to carry) summed to
  the team level per game, then rolled forward per team. True
  possession-based pace (pbpstats) is still deferred — see Section 5.2 —
  this is the free-tier, box-score-only substitute.
- **Opponent defense vs. position** (`opp_def_vs_position_pts_roll10`):
  for the player's own position group (G/F/C, from Sleeper via
  `player_xref` — players with no Sleeper match get NULL here, same as any
  other missing feature LightGBM handles natively), the opponent's rolling
  average of points allowed to that position group, per game, over their
  last 10 games.

None of the above touches `nba_box_status` / `nba_player_availability` —
those are deliberately excluded until that backfill is complete (teammate-
out features come after, as a separate pass).
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

    usage_rate_roll10               DOUBLE,
    team_pace_roll10                 DOUBLE,
    opp_pace_roll10                  DOUBLE,
    opp_def_vs_position_pts_roll10   DOUBLE,

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

-- Migration: usage_rate_roll10/team_pace_roll10/opp_pace_roll10/
-- opp_def_vs_position_pts_roll10 added after this table already existed
-- in earlier databases — CREATE TABLE IF NOT EXISTS above is a no-op
-- against those, same bug class as nba_player_games' fga/fgm/etc columns
-- (see that table's migration comment and its regression test).
ALTER TABLE {FEATURE_TABLE} ADD COLUMN IF NOT EXISTS usage_rate_roll10 DOUBLE;
ALTER TABLE {FEATURE_TABLE} ADD COLUMN IF NOT EXISTS team_pace_roll10 DOUBLE;
ALTER TABLE {FEATURE_TABLE} ADD COLUMN IF NOT EXISTS opp_pace_roll10 DOUBLE;
ALTER TABLE {FEATURE_TABLE} ADD COLUMN IF NOT EXISTS opp_def_vs_position_pts_roll10 DOUBLE;
"""

# Each CREATE OR REPLACE TEMP TABLE step is kept separate (rather than one
# giant CTE) so it's debuggable on its own — `conn.execute("SELECT * FROM
# team_pace_roll LIMIT 5")` etc. works mid-pipeline.
_TEAM_GAME_TOTALS_SQL = """
CREATE OR REPLACE TEMP TABLE team_game_totals AS
SELECT team, game_id, game_date, season,
       SUM(fga) AS team_fga, SUM(fta) AS team_fta, SUM(tov) AS team_tov,
       SUM(oreb) AS team_oreb, SUM(minutes) AS team_minutes
FROM nba_player_games
GROUP BY team, game_id, game_date, season
"""

_PLAYER_POSITION_SQL = """
CREATE OR REPLACE TEMP TABLE player_position AS
SELECT x.player_id AS player_id_nba,
       CASE
           WHEN s.position IN ('PG', 'SG') THEN 'G'
           WHEN s.position IN ('SF', 'PF') THEN 'F'
           WHEN s.position = 'C' THEN 'C'
           ELSE NULL
       END AS position_group
FROM player_xref x
JOIN sleeper_players s ON s.sleeper_id = x.sleeper_id
WHERE s.position IS NOT NULL
"""

_PLAYER_GAME_USAGE_SQL = """
CREATE OR REPLACE TEMP TABLE player_game_usage AS
SELECT g.player_id_nba, g.game_id, g.game_date,
       100.0 * ((g.fga + 0.44 * g.fta + g.tov) * (t.team_minutes / 5.0))
             / NULLIF(g.minutes * (t.team_fga + 0.44 * t.team_fta + t.team_tov), 0) AS usage_rate
FROM nba_player_games g
JOIN team_game_totals t ON t.team = g.team AND t.game_id = g.game_id
"""
# Deliberately NOT `WHERE g.minutes > 0`: this needs a row for EVERY game
# (NULLIF above already turns a 0-minute game's usage_rate into NULL
# safely, with no division-by-zero), because usage_roll below windows
# "ROWS 10 PRECEDING" over THIS table, joined back onto nba_player_games
# by the same (player, game_id). Filtering a 0-minute game out here used
# to drop it from that join entirely, so usage_rate_roll10 for THAT row
# came out NULL regardless of the player's real usage history — purely
# because the row describing the player's OWN current (not yet known)
# game happened to be a 0-minute appearance, not because the history was
# actually missing. Caught by
# backend/validate/test_nba_current_form.py's point-in-time comparison
# against build_current_form, which (correctly) has no way to know a
# not-yet-played game's minutes in advance and always rolls over real
# history.

_USAGE_ROLL_SQL = """
CREATE OR REPLACE TEMP TABLE usage_roll AS
SELECT player_id_nba, game_id,
       AVG(usage_rate) OVER (
           PARTITION BY player_id_nba ORDER BY game_date, game_id ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
       ) AS usage_rate_roll10
FROM player_game_usage
"""

_TEAM_PACE_ROLL_SQL = """
CREATE OR REPLACE TEMP TABLE team_pace_roll AS
SELECT team, game_id,
       AVG(team_fga - team_oreb + team_tov + 0.44 * team_fta) OVER (
           PARTITION BY team ORDER BY game_date, game_id ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
       ) AS team_pace_roll10
FROM team_game_totals
"""

_POSITION_VS_OPPONENT_ROLL_SQL = """
CREATE OR REPLACE TEMP TABLE opp_def_vs_position_roll AS
WITH position_vs_opponent_game AS (
    SELECT g.opponent AS def_team, p.position_group, g.game_id, g.game_date,
           SUM(g.pts) AS pts_allowed
    FROM nba_player_games g
    JOIN player_position p ON p.player_id_nba = g.player_id_nba
    WHERE p.position_group IS NOT NULL
    GROUP BY g.opponent, p.position_group, g.game_id, g.game_date
)
SELECT def_team, position_group, game_id,
       AVG(pts_allowed) OVER (
           PARTITION BY def_team, position_group ORDER BY game_date, game_id ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
       ) AS opp_def_vs_position_pts_roll10
FROM position_vs_opponent_game
"""

_BUILD_SQL = f"""
INSERT OR REPLACE INTO {FEATURE_TABLE}
    (player_id_nba, game_id, game_date, season, team, opponent, home_away,
     rest_days, is_back_to_back, games_played_prior_season,
     min_roll3, min_roll5, min_roll10, min_roll_season, min_std10,
     pts_roll3, pts_roll5, pts_roll10, pts_roll_season, pts_std10,
     reb_roll10, ast_roll10, fg3m_roll10, stl_roll10, blk_roll10, tov_roll10,
     usage_rate_roll10, team_pace_roll10, opp_pace_roll10, opp_def_vs_position_pts_roll10,
     minutes, pts, reb, ast, fg3m, stl, blk, tov, ingested_at)
SELECT
    g.player_id_nba, g.game_id, g.game_date, g.season, g.team, g.opponent, g.home_away,
    DATEDIFF('day', LAG(g.game_date) OVER wplayer, g.game_date) AS rest_days,
    DATEDIFF('day', LAG(g.game_date) OVER wplayer, g.game_date) = 1 AS is_back_to_back,
    CAST(COUNT(*) OVER wseason AS INTEGER) AS games_played_prior_season,

    AVG(g.minutes) OVER w3 AS min_roll3,
    AVG(g.minutes) OVER w5 AS min_roll5,
    AVG(g.minutes) OVER w10 AS min_roll10,
    AVG(g.minutes) OVER wseason AS min_roll_season,
    STDDEV(g.minutes) OVER w10 AS min_std10,

    AVG(g.pts) OVER w3 AS pts_roll3,
    AVG(g.pts) OVER w5 AS pts_roll5,
    AVG(g.pts) OVER w10 AS pts_roll10,
    AVG(g.pts) OVER wseason AS pts_roll_season,
    STDDEV(g.pts) OVER w10 AS pts_std10,

    AVG(g.reb) OVER w10 AS reb_roll10,
    AVG(g.ast) OVER w10 AS ast_roll10,
    AVG(g.fg3m) OVER w10 AS fg3m_roll10,
    AVG(g.stl) OVER w10 AS stl_roll10,
    AVG(g.blk) OVER w10 AS blk_roll10,
    AVG(g.tov) OVER w10 AS tov_roll10,

    ur.usage_rate_roll10,
    tp_own.team_pace_roll10,
    tp_opp.team_pace_roll10 AS opp_pace_roll10,
    dvp.opp_def_vs_position_pts_roll10,

    g.minutes, g.pts, g.reb, g.ast, g.fg3m, g.stl, g.blk, g.tov,
    ? AS ingested_at
FROM nba_player_games g
LEFT JOIN usage_roll ur ON ur.player_id_nba = g.player_id_nba AND ur.game_id = g.game_id
LEFT JOIN team_pace_roll tp_own ON tp_own.team = g.team AND tp_own.game_id = g.game_id
LEFT JOIN team_pace_roll tp_opp ON tp_opp.team = g.opponent AND tp_opp.game_id = g.game_id
LEFT JOIN player_position pp ON pp.player_id_nba = g.player_id_nba
LEFT JOIN opp_def_vs_position_roll dvp
       ON dvp.def_team = g.opponent AND dvp.position_group = pp.position_group AND dvp.game_id = g.game_id
WINDOW
    wplayer AS (PARTITION BY g.player_id_nba ORDER BY g.game_date, g.game_id),
    w3      AS (PARTITION BY g.player_id_nba ORDER BY g.game_date, g.game_id ROWS BETWEEN 3  PRECEDING AND 1 PRECEDING),
    w5      AS (PARTITION BY g.player_id_nba ORDER BY g.game_date, g.game_id ROWS BETWEEN 5  PRECEDING AND 1 PRECEDING),
    w10     AS (PARTITION BY g.player_id_nba ORDER BY g.game_date, g.game_id ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
    wseason AS (PARTITION BY g.player_id_nba, g.season ORDER BY g.game_date, g.game_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
"""


def build(conn: duckdb.DuckDBPyConnection) -> IngestResult:
    conn.execute(_SCHEMA)
    conn.execute(f"DELETE FROM {FEATURE_TABLE}")

    conn.execute(_TEAM_GAME_TOTALS_SQL)
    conn.execute(_PLAYER_POSITION_SQL)
    conn.execute(_PLAYER_GAME_USAGE_SQL)
    conn.execute(_USAGE_ROLL_SQL)
    conn.execute(_TEAM_PACE_ROLL_SQL)
    conn.execute(_POSITION_VS_OPPONENT_ROLL_SQL)

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
