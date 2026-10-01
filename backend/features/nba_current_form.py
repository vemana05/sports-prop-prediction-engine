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
    10 games — same formula as nba_features.py's usage_rate_roll10."""
    team_totals = games.groupby(["team", "game_id"], as_index=False).agg(
        team_fga=("fga", "sum"), team_fta=("fta", "sum"), team_tov=("tov", "sum"), team_minutes=("minutes", "sum")
    )
    merged = games.merge(team_totals, on=["team", "game_id"])
    merged = merged[merged["minutes"] > 0]
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

    team_pace = _team_pace_by_team(games)
    usage_roll10 = _usage_rate_roll10_by_player(games)
    position_by_player = _position_group_by_player(conn)
    def_vs_position = _def_vs_position_roll10(games, position_by_player)

    rows = []
    for player_id, g in games.groupby("player_id_nba", sort=False):
        last = g.iloc[-1]
        current_season = last["season"]
        season_games = g[g["season"] == current_season]
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
        }
        for stat_col, prefix in _FULL_WINDOW_STATS:
            for w in (3, 5, 10):
                row[f"{prefix}_roll{w}"] = g[stat_col].tail(w).mean()
            row[f"{prefix}_roll_season"] = season_games[stat_col].mean() if len(season_games) else None
            row[f"{prefix}_std10"] = g[stat_col].tail(10).std() if len(g) >= 2 else None
        for stat_col, prefix in _ROLL10_ONLY_STATS:
            row[f"{prefix}_roll10"] = g[stat_col].tail(10).mean()
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    conn = get_connection()
    try:
        df = build_current_form(conn)
        print(f"Current form computed for {len(df)} players")
        print(df.sort_values("pts_roll10", ascending=False)[["player_name", "team", "min_roll10", "pts_roll10", "rest_days"]].head(10).to_string(index=False))
    finally:
        conn.close()
