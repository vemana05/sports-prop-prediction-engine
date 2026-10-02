"""Point-in-time NFL player-game features (spec Sections 4.4 / 5.2, Phase 6).

One row per (player, game) for QB/RB/WR/TE who took an offensive snap or
recorded a stat line. Rows come from the UNION of snap counts and stat
lines, so a receiver who played 40 snaps and caught nothing is a real 0,
not a missing game. Dropping those games would bias every rolling average
(and every "under") upward.

Everything is known before kickoff:
- Rolling stats use `ROWS BETWEEN N PRECEDING AND 1 PRECEDING` ordered by
  kickoff, so the current game can never be in its own window (proved
  empirically by backend/validate/test_nfl_features.py).
- Depth chart: the week's chart (through 2024) or the latest dated snapshot
  strictly before kickoff (2025+).
- Injuries: the week's final game-status report (published before games).
- Opponent defense vs. position: the opponent's previous 6 games.
- Weather: Open-Meteo forecasts, not observations (weather_openmeteo.py).
- Prior-season averages and position priors read season-1 only.

Shrinkage toward position averages: each target stat also gets a
`*_shrunk` feature, (sum of last-5 games + K * prior) / (games + K), where
the prior is last season's average for the same position and depth-chart
rank (QB1, WR2, ...). A rookie WR1 with no games starts at the average WR1
line; after five games he's 5/(5+K) his own. The 2020 warm-up season has
no earlier season, so it uses its own position averages: a mild look-ahead
in a season that is only ever trained on, never tested.

    python -m backend.features.nfl_features
"""

from __future__ import annotations

import duckdb

from backend.nfl_db import get_nfl_connection

ROLL_STATS = [
    "passing_yards", "attempts", "completions", "passing_epa",
    "rushing_yards", "carries",
    "receptions", "targets", "receiving_yards", "receiving_air_yards",
    "target_share", "air_yards_share", "carry_share", "snap_pct",
]
TARGETS = ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]
SHRINK_K = 4.0  # pseudo-games of position prior


def _roll(stat: str, n: int, alias: str) -> str:
    return (
        f"AVG({stat}) OVER (PARTITION BY player_id ORDER BY kickoff_utc, game_id "
        f"ROWS BETWEEN {n} PRECEDING AND 1 PRECEDING) AS {alias}"
    )


def build_features(conn: duckdb.DuckDBPyConnection) -> int:
    roll_cols = []
    for s in ROLL_STATS:
        roll_cols.append(_roll(s, 3, f"{s}_roll3"))
        roll_cols.append(_roll(s, 5, f"{s}_roll5"))
        roll_cols.append(
            f"AVG({s}) OVER (PARTITION BY player_id, season ORDER BY kickoff_utc, game_id "
            f"ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS {s}_roll_season"
        )
    for t in TARGETS:
        roll_cols.append(
            f"STDDEV_SAMP({t}) OVER (PARTITION BY player_id ORDER BY kickoff_utc, game_id "
            f"ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING) AS {t}_std8"
        )
        roll_cols.append(
            f"SUM({t}) OVER (PARTITION BY player_id ORDER BY kickoff_utc, game_id "
            f"ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS {t}_sum5"
        )

    stat_cols = ", ".join(
        f"COALESCE(w.{c}, 0) AS {c}" for c in [
            "passing_yards", "attempts", "completions", "passing_epa", "rushing_yards", "carries",
            "receptions", "targets", "receiving_yards", "receiving_air_yards", "target_share", "air_yards_share",
        ]
    )
    prior_avgs = ", ".join(f"AVG({t}) AS prior_season_{t}" for t in TARGETS + ["targets", "carries", "attempts"])
    pos_prior_avgs = ", ".join(f"AVG({t}) AS pos_prior_{t}" for t in TARGETS)
    shrunk = ", ".join(
        f"(COALESCE(r.{t}_sum5, 0) + {SHRINK_K} * pp.pos_prior_{t}) / (r.games_last5 + {SHRINK_K}) AS {t}_shrunk"
        for t in TARGETS
    )

    conn.execute(f"""
    CREATE OR REPLACE TEMP TABLE _base AS
    WITH keys AS (
        SELECT player_id, game_id FROM nfl_player_weekly
        UNION
        SELECT player_id, game_id FROM nfl_snap_counts WHERE player_id IS NOT NULL AND offense_snaps > 0
    ),
    sc AS (SELECT player_id, game_id, max(offense_pct) AS snap_pct, any_value(team) AS team, any_value(position) AS position
           FROM nfl_snap_counts WHERE player_id IS NOT NULL GROUP BY ALL)
    SELECT k.player_id, k.game_id, s.season, s.week, s.game_type, s.kickoff_utc,
           COALESCE(w.team, sc.team) AS team,
           COALESCE(w.position, sc.position) AS position,
           any_value(w.player_name) OVER (PARTITION BY k.player_id) AS player_name,
           CASE WHEN COALESCE(w.team, sc.team) = s.home_team THEN s.away_team ELSE s.home_team END AS opponent,
           (COALESCE(w.team, sc.team) = s.home_team)::INT AS home_flag,
           CASE WHEN COALESCE(w.team, sc.team) = s.home_team THEN s.home_rest ELSE s.away_rest END AS rest_days,
           s.div_game::INT AS div_game, (s.game_type <> 'REG')::INT AS playoff_flag,
           (s.roof NOT IN ('outdoors', 'open'))::INT AS indoor_flag,
           (s.surface ILIKE '%grass%')::INT AS grass_flag,
           wx.temperature_c, wx.wind_kph, wx.precip_mm,
           {stat_cols},
           sc.snap_pct
    FROM keys k
    JOIN nfl_schedule s USING (game_id)
    LEFT JOIN nfl_player_weekly w ON w.player_id = k.player_id AND w.game_id = k.game_id
    LEFT JOIN sc ON sc.player_id = k.player_id AND sc.game_id = k.game_id
    LEFT JOIN nfl_weather wx ON wx.game_id = k.game_id
    WHERE COALESCE(w.position, sc.position) IN ('QB', 'RB', 'WR', 'TE');

    -- Team totals per game (carry share isn't in nflverse's weekly file).
    CREATE OR REPLACE TEMP TABLE _team_game AS
    SELECT team, game_id, any_value(kickoff_utc) AS kickoff_utc, any_value(season) AS season,
           SUM(attempts) AS team_attempts, SUM(carries) AS team_carries
    FROM _base GROUP BY team, game_id;

    ALTER TABLE _base ADD COLUMN carry_share DOUBLE;
    UPDATE _base SET carry_share = _base.carries / NULLIF(t.team_carries, 0)
    FROM _team_game t WHERE t.team = _base.team AND t.game_id = _base.game_id;

    -- Point-in-time depth rank: week chart through 2024, latest snapshot before kickoff from 2025.
    CREATE OR REPLACE TEMP TABLE _depth_raw AS
    WITH snaps AS (SELECT DISTINCT team, snapshot_at FROM nfl_depth_charts WHERE snapshot_at IS NOT NULL),
    team_games AS (SELECT DISTINCT team, game_id, kickoff_utc, season, week FROM _base),
    chosen AS (
        SELECT tg.team, tg.game_id, sn.snapshot_at
        FROM team_games tg ASOF JOIN snaps sn ON sn.team = tg.team AND sn.snapshot_at < tg.kickoff_utc
        WHERE tg.season >= 2025
    )
    SELECT b.player_id, b.game_id, b.team, b.position, MIN(d.depth_rank) AS depth_rank_raw
    FROM _base b
    LEFT JOIN chosen c ON c.team = b.team AND c.game_id = b.game_id
    JOIN nfl_depth_charts d ON d.team = b.team AND d.player_id = b.player_id AND d.position = b.position
         AND ((b.season < 2025 AND d.season = b.season AND d.week = b.week)
              OR (b.season >= 2025 AND d.snapshot_at = c.snapshot_at))
    GROUP BY ALL;

    -- Opponent defense vs. position: what each defense allowed to each position, last 6 games.
    CREATE OR REPLACE TEMP TABLE _def AS
    WITH allowed AS (
        SELECT opponent AS def_team, game_id, any_value(kickoff_utc) AS kickoff_utc, position,
               SUM(passing_yards) AS pass_yds, SUM(rushing_yards) AS rush_yds,
               SUM(receiving_yards) AS rec_yds, SUM(receptions) AS rec
        FROM _base GROUP BY ALL
    )
    SELECT def_team, game_id, position,
           AVG(pass_yds) OVER wd AS opp_allowed_pass_yds_roll6,
           AVG(rush_yds) OVER wd AS opp_allowed_rush_yds_roll6,
           AVG(rec_yds) OVER wd AS opp_allowed_rec_yds_roll6,
           AVG(rec) OVER wd AS opp_allowed_rec_roll6
    FROM allowed
    WINDOW wd AS (PARTITION BY def_team, position ORDER BY kickoff_utc, game_id ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING);

    CREATE OR REPLACE TEMP TABLE _team_roll AS
    SELECT team, game_id,
           AVG(team_attempts) OVER wt AS team_attempts_roll5,
           AVG(team_carries) OVER wt AS team_carries_roll5
    FROM _team_game
    WINDOW wt AS (PARTITION BY team ORDER BY kickoff_utc, game_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING);

    CREATE OR REPLACE TEMP TABLE _roll AS
    SELECT player_id, game_id, {', '.join(roll_cols)},
           COUNT(*) OVER (PARTITION BY player_id ORDER BY kickoff_utc, game_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS games_last5,
           COUNT(*) OVER (PARTITION BY player_id ORDER BY kickoff_utc, game_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS career_games_prior
    FROM _base;

    -- Each player's most recent form, including the game itself: what a
    -- teammate "brings" when he's later listed Out. Only ever joined to a
    -- strictly later game below.
    CREATE OR REPLACE TEMP TABLE _form_incl AS
    SELECT player_id, team, kickoff_utc,
           AVG(target_share) OVER wf AS tshare5, AVG(carry_share) OVER wf AS cshare5, AVG(attempts) OVER wf AS att5
    FROM _base
    WINDOW wf AS (PARTITION BY player_id ORDER BY kickoff_utc, game_id ROWS BETWEEN 4 PRECEDING AND CURRENT ROW);

    CREATE OR REPLACE TEMP TABLE _team_out AS
    WITH tg AS (SELECT DISTINCT team, game_id, season, week, kickoff_utc FROM _base),
    out_players AS (
        SELECT tg.team, tg.game_id, tg.kickoff_utc, i.player_id
        FROM tg JOIN nfl_injuries i ON i.team = tg.team AND i.season = tg.season AND i.week = tg.week
        WHERE i.report_status IN ('Out', 'Doubtful')
    ),
    with_form AS (
        SELECT o.team, o.game_id, f.tshare5, f.cshare5, f.att5
        FROM out_players o ASOF JOIN _form_incl f ON f.player_id = o.player_id AND f.kickoff_utc < o.kickoff_utc
    )
    SELECT team, game_id,
           COALESCE(SUM(tshare5), 0) AS team_out_target_share,
           COALESCE(SUM(cshare5), 0) AS team_out_carry_share,
           MAX((att5 >= 20)::INT) AS team_starting_qb_out
    FROM with_form GROUP BY ALL;

    -- The two chart formats rank differently: through 2024 each slot (left
    -- WR, right WR, slot WR) has its own "1", so ~2.5 WRs per team are rank 1;
    -- from 2025 ranks run 1..n across the position. Re-rank both into one
    -- ordinal per team-game (WR1, WR2, WR3, ...), breaking ties by earlier
    -- usage, so depth_rank means the same thing in every season.
    CREATE OR REPLACE TEMP TABLE _depth AS
    SELECT d.player_id, d.game_id, d.depth_rank_raw,
           ROW_NUMBER() OVER (
               PARTITION BY d.team, d.game_id, d.position
               ORDER BY d.depth_rank_raw,
                        COALESCE(r.targets_roll5, 0) + COALESCE(r.carries_roll5, 0) + COALESCE(r.attempts_roll5, 0) DESC,
                        d.player_id
           ) AS depth_rank
    FROM _depth_raw d JOIN _roll r USING (player_id, game_id);

    CREATE OR REPLACE TEMP TABLE _prior AS
    SELECT player_id, season + 1 AS season, COUNT(*) AS prior_season_games, {prior_avgs}
    FROM _base GROUP BY player_id, season;
    """)

    conn.execute(f"""
    CREATE OR REPLACE TABLE nfl_features AS
    WITH bucketed AS (
        SELECT b.*, d.depth_rank, d.depth_rank_raw, LEAST(COALESCE(d.depth_rank, 4), 4) AS depth_bucket
        FROM _base b LEFT JOIN _depth d USING (player_id, game_id)
    ),
    pos_prior AS (
        SELECT season + 1 AS season, position, depth_bucket, {pos_prior_avgs}
        FROM bucketed GROUP BY ALL
        UNION ALL  -- the warm-up season's prior is its own average (see module docstring)
        SELECT season, position, depth_bucket, {pos_prior_avgs}
        FROM bucketed WHERE season = (SELECT MIN(season) FROM bucketed) GROUP BY ALL
    )
    SELECT b.*, r.* EXCLUDE (player_id, game_id),
           COALESCE(p.prior_season_games, 0) AS prior_season_games,
           p.* EXCLUDE (player_id, season, prior_season_games),
           {shrunk},
           df.opp_allowed_pass_yds_roll6, df.opp_allowed_rush_yds_roll6,
           df.opp_allowed_rec_yds_roll6, df.opp_allowed_rec_roll6,
           tr.team_attempts_roll5, tr.team_carries_roll5,
           COALESCE(o.team_out_target_share, 0) AS team_out_target_share,
           COALESCE(o.team_out_carry_share, 0) AS team_out_carry_share,
           COALESCE(o.team_starting_qb_out, 0) AS team_starting_qb_out,
           (inj.report_status = 'Questionable')::INT AS questionable_flag,
           (r.career_games_prior = 0)::INT AS no_history_flag
    FROM bucketed b
    JOIN _roll r USING (player_id, game_id)
    LEFT JOIN _prior p ON p.player_id = b.player_id AND p.season = b.season
    LEFT JOIN pos_prior pp ON pp.season = b.season AND pp.position = b.position AND pp.depth_bucket = b.depth_bucket
    LEFT JOIN _def df ON df.def_team = b.opponent AND df.game_id = b.game_id AND df.position = b.position
    LEFT JOIN _team_roll tr ON tr.team = b.team AND tr.game_id = b.game_id
    LEFT JOIN _team_out o ON o.team = b.team AND o.game_id = b.game_id
    LEFT JOIN nfl_injuries inj ON inj.player_id = b.player_id AND inj.team = b.team AND inj.season = b.season AND inj.week = b.week
    """)

    # Market universes, defined only from pre-game information (depth chart
    # and earlier games), never from the game's own usage.
    conn.execute("""
    ALTER TABLE nfl_features ADD COLUMN in_passing_market BOOLEAN;
    ALTER TABLE nfl_features ADD COLUMN in_rushing_market BOOLEAN;
    ALTER TABLE nfl_features ADD COLUMN in_receiving_market BOOLEAN;
    UPDATE nfl_features SET
        in_passing_market = position = 'QB' AND (depth_rank = 1 OR COALESCE(attempts_roll3, 0) >= 15),
        in_rushing_market = (position = 'RB' AND (depth_rank <= 2 OR COALESCE(carries_roll3, 0) >= 5))
                         OR (position = 'QB' AND (depth_rank = 1 OR COALESCE(attempts_roll3, 0) >= 15)),
        in_receiving_market = (position = 'WR' AND (depth_rank <= 3 OR COALESCE(targets_roll3, 0) >= 3))
                           OR (position = 'TE' AND (depth_rank = 1 OR COALESCE(targets_roll3, 0) >= 3))
                           OR (position = 'RB' AND (depth_rank = 1 OR COALESCE(targets_roll3, 0) >= 2));
    """)
    return conn.execute("SELECT COUNT(*) FROM nfl_features").fetchone()[0]


if __name__ == "__main__":
    from backend.ingest.backup_db import backup_warehouse
    from backend.nfl_db import DEFAULT_NFL_DB_PATH

    conn = get_nfl_connection()
    try:
        n = build_features(conn)
    finally:
        conn.close()
    print(f"nfl_features: {n} rows")
    backup_warehouse(db_path=DEFAULT_NFL_DB_PATH)
