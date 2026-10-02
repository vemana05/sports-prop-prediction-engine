"""Current rolling-form snapshot per active player — the "as of right now"
equivalent of nba_features.py's point-in-time rolling windows, used to
score upcoming (not-yet-played) games rather than backtest historical ones.

Each player's most recent games (from nba_player_games, regardless of
season — so form carries over the offseason gap) feed the same rolling
windows nba_features uses: last 3/5/10 games and season-to-date for
minutes/points (which have full windows in nba_features), last-10 only for
rebounds/assists/3PM/steals/blocks/turnovers (which only ever had a 10-game
window — see nba_features.py's schema). "games_played_prior_season" and the
season-scoped averages are null/zero for the first game(s) of a new season,
same as any other early-season row in nba_features — not a special case.

Regression test's backstory: the season-scoped features above (
`games_played_prior_season`, `*_roll_season`) used to be computed against
the season of each player's own LAST PLAYED game, not the season of the
game actually being predicted. That's the same thing everywhere except
right at a season boundary: called before the new season's first games
exist in `nba_player_games` (e.g. scoring opening night itself), every
player's "last played game" is still last season's finale, so
`games_played_prior_season` came out as last season's full game count and
`pts_roll_season` as last season's full average — instead of the 0 /
NULL that `nba_features.py` computes for that exact same row, point-in-time
(its `wseason` window is season-partitioned and empty on a season's first
game). A model trained on rows where those two features are always
consistent with each other sees wildly out-of-distribution, self-
contradictory inputs the one time it matters most: predicting a player's
first game of a new season. `_season_as_of` below fixes this by asking
`nba_schedule` what season the next scheduled game on/after `as_of`
actually belongs to, instead of inferring it from historical data that
doesn't yet contain a row for that season. See
`backend/validate/test_nba_current_form.py`'s
`test_season_opener_features_match_nba_features_point_in_time`.

Prior-season (`prior_season_*`) and teammate-out (`teammates_out_*`)
features mirror nba_features.py's exactly. "Prior season" is the season
before the one being predicted (`_season_as_of`), not before the player's
last played one — same season-boundary reasoning as above. Teammate-out
needs to be told who is out (`out_players_by_team`): live, that's
`teammates_out_from_injury_report` (latest official report before the
slate's first tip-off); for point-in-time tests against nba_features,
`teammates_out_from_availability`. With neither, every team gets 0 out —
the same value nba_features stores for a game where nobody was listed Out.

Real limitation: this has no roster source beyond "who recently played for
this team" (nba_player_games), so a player who changed teams since their
last recorded game (trade, free agency, waived) shows up under their OLD
team here until they've played at least one game for the new one. Good
enough for forward prediction logging; not a substitute for a real
depth-chart/roster feed.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

from backend.db import get_connection

# (stat column in nba_player_games, feature-name prefix, windows needed)
_FULL_WINDOW_STATS = [("minutes", "min"), ("pts", "pts")]
_ROLL10_ONLY_STATS = [("reb", "reb"), ("ast", "ast"), ("fg3m", "fg3m"), ("stl", "stl"), ("blk", "blk"), ("tov", "tov")]


def _prior_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{(start + 1) % 100:02d}"


def teammates_out_from_availability(conn: duckdb.DuckDBPyConnection, game_date: date) -> dict[str, list[str]]:
    """{team_abbr: [player_id_nba, ...]} listed Out pre-tip for games on
    `game_date`, from nba_player_availability — the exact set
    nba_features.py's teammates_out_* columns are built from. For
    historical/point-in-time comparisons only (it needs the box score)."""
    rows = conn.execute(
        """
        SELECT DISTINCT t.team, a.player_id_nba
        FROM nba_player_availability a
        JOIN nba_schedule s ON s.game_id = a.game_id
        JOIN (SELECT DISTINCT b.game_id, b.team_id, g.team
              FROM nba_box_status b
              JOIN nba_player_games g ON g.game_id = b.game_id AND g.player_id_nba = b.player_id_nba) t
          ON t.game_id = a.game_id AND t.team_id = a.team_id
        WHERE s.game_date = ? AND a.status = 'dnp_listed_injury_report'
          AND a.detail LIKE 'Out%' AND a.detail NOT ILIKE '%G League%'
        """,
        [game_date],
    ).fetchall()
    out: dict[str, list[str]] = {}
    for team, pid in rows:
        out.setdefault(team, []).append(pid)
    return out


def teammates_out_from_injury_report(conn: duckdb.DuckDBPyConnection, game_date: date) -> dict[str, list[str]]:
    """{team_abbr: [player_id_nba, ...]} listed Out (G League assignments
    excluded) on the latest official injury report covering `game_date`
    and published before that date's first tip-off — the live-serving
    counterpart of `teammates_out_from_availability`. Empty if no report
    for that date has been ingested yet."""
    from nba_api.stats.static import teams as nba_static_teams

    name_to_abbr = {t["full_name"]: t["abbreviation"] for t in nba_static_teams.get_teams()}
    rows = conn.execute(
        """
        WITH first_tip AS (SELECT MIN(tip_utc) AS tip FROM nba_schedule WHERE game_date = ?),
        latest AS (
            SELECT MAX(i.report_time) AS rt FROM nba_injuries i, first_tip f
            WHERE i.game_date = ? AND (i.report_time AT TIME ZONE 'America/New_York') < f.tip
        )
        SELECT DISTINCT i.team, i.player_id_nba
        FROM nba_injuries i, latest l
        WHERE i.game_date = ? AND i.report_time = l.rt AND i.player_id_nba IS NOT NULL
          AND i.status = 'Out' AND COALESCE(i.reason, '') NOT ILIKE '%G League%'
        """,
        [game_date, game_date, game_date],
    ).fetchall()
    out: dict[str, list[str]] = {}
    for team_name, pid in rows:
        abbr = name_to_abbr.get(team_name)
        if abbr:
            out.setdefault(abbr, []).append(pid)
    return out


def _season_as_of(conn: duckdb.DuckDBPyConnection, as_of: date) -> str | None:
    """The season of the next scheduled game on/after `as_of` — the season
    an upcoming prediction actually belongs to, which can differ from the
    season of a player's most recent PLAYED game (see module docstring).
    None if `nba_schedule` has no row on/after `as_of` (not yet ingested
    that far ahead) — callers fall back to the old, pre-season-boundary-
    safe behavior in that case."""
    row = conn.execute("SELECT season FROM nba_schedule WHERE game_date >= ? ORDER BY game_date LIMIT 1", [as_of]).fetchone()
    return row[0] if row else None


def _team_pace_by_team(games: pd.DataFrame) -> dict[str, float]:
    """Each team's box-score pace proxy (FGA - OREB + TOV + 0.44*FTA),
    rolled over its last 10 games played to date — same formula as
    nba_features.py's team_pace_roll10, computed here over ALL known games
    (not point-in-time per historical row) since this is "current form"."""
    totals = games.groupby(["team", "game_id", "game_date"], as_index=False).agg(
        fga=("fga", "sum"), fta=("fta", "sum"), tov=("tov", "sum"), oreb=("oreb", "sum")
    )
    totals["pace"] = totals["fga"] - totals["oreb"] + totals["tov"] + 0.44 * totals["fta"]
    out = {}
    for team, g in totals.sort_values("game_date").groupby("team"):
        out[team] = g["pace"].tail(10).mean()
    return out


def _usage_rate_roll10_by_player(games: pd.DataFrame) -> dict[str, float]:
    """Standard box-score USG% formula, rolled over each player's own last
    10 GAMES (not the last 10 games where they happened to log minutes —
    same window nba_features.py's usage_roll now uses, over ALL of
    nba_player_games, not just rows that already have a usage_rate: a
    0-minute game still occupies one of the "10 preceding" slots, it just
    contributes no value — `.mean()` skips the NaN same as DuckDB's `AVG`
    does). Rolling over only qualifying games here, while nba_features
    rolls over the last 10 PHYSICAL games, drifted the two out of exact
    agreement — caught by
    backend/validate/test_nba_current_form.py's point-in-time test."""
    team_totals = games.groupby(["team", "game_id"], as_index=False).agg(
        team_fga=("fga", "sum"), team_fta=("fta", "sum"), team_tov=("tov", "sum"), team_minutes=("minutes", "sum")
    )
    merged = games.merge(team_totals, on=["team", "game_id"])
    denom = merged["minutes"] * (merged["team_fga"] + 0.44 * merged["team_fta"] + merged["team_tov"])
    numer = (merged["fga"] + 0.44 * merged["fta"] + merged["tov"]) * (merged["team_minutes"] / 5.0)
    merged["usage_rate"] = 100.0 * numer / denom.replace(0, pd.NA)
    out = {}
    for player_id, g in merged.sort_values("game_date").groupby("player_id_nba"):
        out[player_id] = g["usage_rate"].tail(10).mean()
    return out


def _position_group_by_player(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    rows = conn.execute(
        """
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
    ).fetchdf()
    return dict(zip(rows["player_id_nba"], rows["position_group"]))


def _def_vs_position_roll10(games: pd.DataFrame, position_by_player: dict[str, str]) -> dict[tuple[str, str], float]:
    """{(def_team, position_group): points allowed to that position group,
    rolled over the defending team's last 10 games} — same formula as
    nba_features.py's opp_def_vs_position_pts_roll10."""
    g = games.copy()
    g["position_group"] = g["player_id_nba"].map(position_by_player)
    g = g.dropna(subset=["position_group", "opponent"])
    per_game = g.groupby(["opponent", "position_group", "game_id", "game_date"], as_index=False)["pts"].sum()
    out = {}
    for (def_team, pos), grp in per_game.sort_values("game_date").groupby(["opponent", "position_group"]):
        out[(def_team, pos)] = grp["pts"].tail(10).mean()
    return out


def build_current_form(
    conn: duckdb.DuckDBPyConnection,
    as_of: date | None = None,
    opponent_by_team: dict[str, str] | None = None,
    out_players_by_team: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """One row per player who has ever appeared in nba_player_games, with
    rolling-average features computed from their most recent games strictly
    before `as_of` (default: today) — point-in-time correct the same way
    nba_features.py's historical rows are.

    `opponent_by_team` (optional: {team: upcoming_opponent_team}) fills in
    opp_pace_roll10 / opp_def_vs_position_pts_roll10 for players whose team
    is a key — these two features are about an UPCOMING game, so without a
    known next opponent they're left NULL (LightGBM handles missing values
    natively, same as nba_features.py's own NULLs for unmatched positions)."""
    as_of = as_of or date.today()
    opponent_by_team = opponent_by_team or {}
    out_players_by_team = out_players_by_team or {}
    games = conn.execute(
        """
        SELECT player_id_nba, player_name, team, opponent, season, game_date, game_id,
               minutes, pts, reb, ast, fg3m, stl, blk, tov, fga, fta, oreb
        FROM nba_player_games
        WHERE game_date < ?
        ORDER BY player_id_nba, game_date, game_id
        """,
        [as_of],
    ).fetchdf()
    if games.empty:
        return pd.DataFrame()

    schedule_season = _season_as_of(conn, as_of)
    team_pace = _team_pace_by_team(games)
    usage_roll10 = _usage_rate_roll10_by_player(games)
    position_by_player = _position_group_by_player(conn)
    def_vs_position = _def_vs_position_roll10(games, position_by_player)

    # teammates_out_*: each out player's last-10 pts/minutes INCLUDING their
    # most recent game (nba_features.py's player_form_l10 + ASOF join).
    last10 = games.groupby("player_id_nba").tail(10).groupby("player_id_nba")[["pts", "minutes"]].mean()
    teammates_out = {}
    for team, out_ids in out_players_by_team.items():
        known = last10.reindex([pid for pid in set(out_ids) if pid in last10.index])
        teammates_out[team] = (len(set(out_ids)), float(known["pts"].sum()), float(known["minutes"].sum()))

    rows = []
    for player_id, g in games.groupby("player_id_nba", sort=False):
        last = g.iloc[-1]
        current_season = schedule_season or last["season"]
        season_games = g[g["season"] == current_season]
        prior_games = g[g["season"] == _prior_season(current_season)]
        # fetchdf() returns game_date as a pandas Timestamp, not a Python
        # date — normalize before subtracting from `as_of` (a plain date),
        # or this raises TypeError (the same type-mismatch class of bug as
        # nba_availability.py's — see that module's test for the backstory).
        last_game_date = last["game_date"].date() if hasattr(last["game_date"], "date") else last["game_date"]
        rest_days = (as_of - last_game_date).days
        team = last["team"]
        opponent = opponent_by_team.get(team)
        position_group = position_by_player.get(player_id)

        row = {
            "player_id_nba": player_id,
            "player_name": last["player_name"],
            "team": team,
            "last_game_date": last_game_date,
            "rest_days": rest_days,
            "is_back_to_back": rest_days == 1,
            "games_played_prior_season": len(season_games),
            "usage_rate_roll10": usage_roll10.get(player_id),
            "team_pace_roll10": team_pace.get(team),
            "opp_pace_roll10": team_pace.get(opponent) if opponent else None,
            "opp_def_vs_position_pts_roll10": (
                def_vs_position.get((opponent, position_group)) if opponent and position_group else None
            ),
            "prior_season_games": len(prior_games),
        }
        for stat_col, prefix in [("minutes", "min"), ("pts", "pts"), ("reb", "reb"), ("ast", "ast"), ("fg3m", "fg3m")]:
            row[f"prior_season_{prefix}_avg"] = prior_games[stat_col].mean() if len(prior_games) else None
        n_out, out_pts, out_min = teammates_out.get(team, (0, 0.0, 0.0))
        row["teammates_out_count"] = n_out
        row["teammates_out_pts_l10"] = out_pts
        row["teammates_out_min_l10"] = out_min
        for stat_col, prefix in _FULL_WINDOW_STATS:
            for w in (3, 5, 10):
                row[f"{prefix}_roll{w}"] = g[stat_col].tail(w).mean()
            row[f"{prefix}_roll_season"] = season_games[stat_col].mean() if len(season_games) else None
            row[f"{prefix}_std10"] = g[stat_col].tail(10).std() if len(g) >= 2 else None
        for stat_col, prefix in _ROLL10_ONLY_STATS:
            row[f"{prefix}_roll10"] = g[stat_col].tail(10).mean()
        rows.append(row)

    df = pd.DataFrame(rows)
    # Every numeric feature column defaults to Python `None` for a player
    # with no qualifying games (e.g. every *_roll_season column on a
    # season's first scored slate — see module docstring). `pd.DataFrame`
    # infers a column's dtype from ALL its rows: if every row in a given
    # batch happens to be None for one column (plausible for small batches,
    # certain on a season opener), pandas gives it `object` dtype instead
    # of `float64` with NaN — which LightGBM's `.predict()` then rejects
    # outright ("pandas dtypes must be int, float or bool"). Casting
    # explicitly avoids depending on what the rest of the batch looks like.
    non_numeric = {"player_id_nba", "player_name", "team", "last_game_date", "is_back_to_back"}
    numeric_cols = [c for c in df.columns if c not in non_numeric]
    df[numeric_cols] = df[numeric_cols].astype("float64")
    return df


if __name__ == "__main__":
    conn = get_connection()
    try:
        df = build_current_form(conn)
        print(f"Current form computed for {len(df)} players")
        print(df.sort_values("pts_roll10", ascending=False)[["player_name", "team", "min_roll10", "pts_roll10", "rest_days"]].head(10).to_string(index=False))
    finally:
        conn.close()
