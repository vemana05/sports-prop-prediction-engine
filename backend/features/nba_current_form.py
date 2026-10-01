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


def build_current_form(conn: duckdb.DuckDBPyConnection, as_of: date | None = None) -> pd.DataFrame:
    """One row per player who has ever appeared in nba_player_games, with
    rolling-average features computed from their most recent games strictly
    before `as_of` (default: today) — point-in-time correct the same way
    nba_features.py's historical rows are."""
    as_of = as_of or date.today()
    games = conn.execute(
        """
        SELECT player_id_nba, player_name, team, season, game_date,
               minutes, pts, reb, ast, fg3m, stl, blk, tov
        FROM nba_player_games
        WHERE game_date < ?
        ORDER BY player_id_nba, game_date, game_id
        """,
        [as_of],
    ).fetchdf()
    if games.empty:
        return pd.DataFrame()

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

        row = {
            "player_id_nba": player_id,
            "player_name": last["player_name"],
            "team": last["team"],
            "last_game_date": last_game_date,
            "rest_days": rest_days,
            "is_back_to_back": rest_days == 1,
            "games_played_prior_season": len(season_games),
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
