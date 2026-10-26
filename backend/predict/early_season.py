"""Early-season window (NBA): until every team has played
`EARLY_SEASON_GAMES` games, rolling features still lean on last season's
stats, star projections run low (spec 5.9), and probabilities are
unreliable. The Builder, Schedule and Replay pages show a banner while the
window is open, and the parlay sanity guard uses stricter limits
(`backend.parlay.search`).

Games played are counted from `nba_player_games` — the same game logs the
model's season features (`games_played_prior_season`, `*_roll_season`)
are built from — not from the schedule: a game that tipped but whose logs
aren't loaded yet doesn't count, so the banner and the model inputs never
disagree. The season is the current one per `nba_schedule` (the season of
the next scheduled game, or the latest season once the schedule runs
out), and every team on that season's schedule is listed, at 0 before
its first loaded game. `status()` also reports `games_missing_logs`:
scheduled games before today with no logs loaded yet.
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
    """(season, {team abbreviation: games with loaded logs before `as_of`})
    — every team on that season's schedule is listed, at 0 before its
    first loaded game."""
    season = current_season(conn, as_of)
    if season is None:
        return None, {}
    scheduled = conn.execute(
        """
        SELECT DISTINCT team_id FROM (
            SELECT home_team_id AS team_id FROM nba_schedule WHERE season = ?
            UNION ALL SELECT away_team_id FROM nba_schedule WHERE season = ?
        )
        """,
        [season, season],
    ).fetchall()
    abbr = _abbr()  # skips placeholder ids (0) on not-yet-decided games
    played = {abbr[str(tid)]: 0 for (tid,) in scheduled if str(tid) in abbr}
    for team, n in conn.execute(
        """
        SELECT team, count(DISTINCT game_id) FROM nba_player_games
        WHERE season = ? AND game_date < ? GROUP BY team
        """,
        [season, as_of],
    ).fetchall():
        if team in played:
            played[team] = int(n)
    return season, played


def games_missing_logs(conn: duckdb.DuckDBPyConnection, season: str | None, as_of: date) -> int:
    """Scheduled games of `season` before `as_of` with no player logs loaded."""
    if season is None:
        return 0
    return conn.execute(
        """
        SELECT count(*) FROM nba_schedule s
        WHERE s.season = ? AND s.game_date < ?
          AND s.game_id NOT IN (SELECT DISTINCT game_id FROM nba_player_games WHERE season = ?)
        """,
        [season, as_of, season],
    ).fetchone()[0]


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
        season, played = team_games_played(conn, as_of)
        return {
            "as_of": as_of.isoformat(),
            **summarize(season, played),
            "games_missing_logs": games_missing_logs(conn, season, as_of),
        }
    finally:
        if own:
            conn.close()
