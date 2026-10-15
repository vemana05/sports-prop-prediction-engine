"""Early-season window (NBA): until every team has played
`EARLY_SEASON_GAMES` games, rolling features still lean on last season's
stats, star projections run low (spec 5.9), and probabilities are
unreliable. The Builder, Schedule and Replay pages show a banner while the
window is open, and the parlay sanity guard uses stricter limits
(`backend.parlay.search`).

Games played are counted from `nba_schedule` (regular season only): games
dated before today in the current season — the season of the next
scheduled game, or the latest season once the schedule runs out.
"""

from __future__ import annotations

from datetime import date

import duckdb

from backend.predict.schedule_projections import _abbr, _connect, today_et

EARLY_SEASON_GAMES = 10


def current_season(conn: duckdb.DuckDBPyConnection, as_of: date) -> str | None:
    row = conn.execute(
        """
        SELECT COALESCE(
            (SELECT season FROM nba_schedule WHERE game_date >= ? ORDER BY game_date LIMIT 1),
            (SELECT max(season) FROM nba_schedule))
        """,
        [as_of],
    ).fetchone()
    return row[0]


def team_games_played(conn: duckdb.DuckDBPyConnection, as_of: date) -> tuple[str | None, dict[str, int]]:
    """(season, {team abbreviation: games played before `as_of`}) — every
    team on that season's schedule is listed, at 0 before its opener."""
    season = current_season(conn, as_of)
    if season is None:
        return None, {}
    rows = conn.execute(
        """
        WITH team_games AS (
            SELECT home_team_id AS team_id, game_date FROM nba_schedule WHERE season = ?
            UNION ALL SELECT away_team_id, game_date FROM nba_schedule WHERE season = ?
        )
        SELECT team_id, count(*) FILTER (WHERE game_date < ?) FROM team_games GROUP BY team_id
        """,
        [season, season, as_of],
    ).fetchall()
    abbr = _abbr()  # skips placeholder ids (0) on not-yet-decided games
    return season, {abbr[str(tid)]: int(n) for tid, n in rows if str(tid) in abbr}


def summarize(season: str | None, played: dict[str, int], threshold: int = EARLY_SEASON_GAMES) -> dict:
    """`active` while any team is under `threshold` games. `min_games_played`
    is what the banner counts down from; `teams_below` lists who's short."""
    if not played:
        return {"active": False, "season": season, "threshold": threshold, "min_games_played": None,
                "games_remaining": 0, "teams_below": [], "games_played": {}}
    low = min(played.values())
    return {
        "active": low < threshold,
        "season": season,
        "threshold": threshold,
        "min_games_played": low,
        "games_remaining": max(threshold - low, 0),
        "teams_below": sorted(t for t, n in played.items() if n < threshold),
        "games_played": dict(sorted(played.items())),
    }


def status(as_of: date | None = None, conn: duckdb.DuckDBPyConnection | None = None) -> dict:
    as_of = as_of or today_et()
    own = conn is None
    conn = conn or _connect()
    try:
        return {"as_of": as_of.isoformat(), **summarize(*team_games_played(conn, as_of))}
    finally:
        if own:
            conn.close()
