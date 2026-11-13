"""Fallback for finished NBA games when stats.nba.com is unreachable: ESPN's
public, keyless site API (scoreboard + game summary JSON).

Since 2026-10-28 every stats.nba.com endpoint hangs from this network while
www.nba.com and cdn.nba.com answer 403 "Access Denied" (Akamai) — a block on
the network, not something headers or request pacing fix. nba_daily_games.py
calls this when LeagueGameLog or ScheduleLeagueV2 fails.

Covers, per finished game on a date:
  - nba_game_results: final status (3) and home/away scores.
  - nba_player_games: one row per player who played (the same columns
    LeagueGameLog fills), source = 'espn'.
Not covered: nba_box_status / nba_player_availability (DNP comments), and
tip times for games not already in nba_schedule.

ESPN events are matched to NBA game ids through nba_schedule (date + home
and away team), so only games already on the stored schedule are loaded.
ESPN athlete ids are matched to NBA player ids by normalized name: players
already in nba_player_games first, then nba_api's static list (no network).
A player matched by neither is skipped and logged, not guessed.
"""

from __future__ import annotations

from datetime import date, datetime

import duckdb
import pandas as pd
import requests
from nba_api.stats.static import players as nba_static_players
from nba_api.stats.static import teams as nba_static_teams

from backend.ingest.util import logger, retry
from backend.xref.build_xref import normalize_name

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
TIMEOUT_S = 15
FINAL_STATUS = 3
# ESPN abbreviations that differ from nba.com's.
ESPN_TO_NBA_ABBR = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
# ESPN box score key -> nba_player_games column(s); "a-b" keys are made-attempted.
_STAT_KEYS = {
    "minutes": "minutes", "points": "pts", "rebounds": "reb", "assists": "ast", "turnovers": "tov",
    "steals": "stl", "blocks": "blk", "offensiveRebounds": "oreb", "defensiveRebounds": "dreb",
    "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", None),
    "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
}


def _get(path: str, params: dict) -> dict:
    def _call():
        r = requests.get(f"{BASE}/{path}", params=params, timeout=TIMEOUT_S)
        r.raise_for_status()
        return r.json()

    return retry(_call, attempts=2, what=f"ESPN {path}({params})")


def fetch_scoreboard(day: date) -> dict:
    return _get("scoreboard", {"dates": day.strftime("%Y%m%d")})


def fetch_summary(event_id: str) -> dict:
    return _get("summary", {"event": event_id})


def nba_abbr(espn_abbr: str) -> str:
    return ESPN_TO_NBA_ABBR.get(espn_abbr, espn_abbr)


def final_games(scoreboard: dict) -> list[dict]:
    """[{event_id, home, away, home_score, away_score}] for completed games,
    teams as nba.com abbreviations."""
    out = []
    for ev in scoreboard.get("events", []):
        if not ev.get("status", {}).get("type", {}).get("completed"):
            continue
        comp = ev["competitions"][0]
        side = {c["homeAway"]: c for c in comp["competitors"]}
        out.append({
            "event_id": str(ev["id"]),
            "home": nba_abbr(side["home"]["team"]["abbreviation"]),
            "away": nba_abbr(side["away"]["team"]["abbreviation"]),
            "home_score": int(side["home"]["score"]),
            "away_score": int(side["away"]["score"]),
        })
    return out


def schedule_games(conn: duckdb.DuckDBPyConnection, day: date) -> dict[tuple[str, str], tuple[str, str]]:
    """{(home_abbr, away_abbr): (game_id, season)} from nba_schedule for `day`."""
    abbr = {str(t["id"]): t["abbreviation"] for t in nba_static_teams.get_teams()}
    rows = conn.execute("SELECT game_id, season, home_team_id, away_team_id FROM nba_schedule WHERE game_date = ?",
                        [day]).fetchall()
    return {(abbr.get(h), abbr.get(a)): (g, s) for g, s, h, a in rows}


def player_ids(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """{normalized name: NBA player id}. Logged players win over the static
    list (it can lag on rookies and trades but not on ids); a name shared by
    two ids keeps the one with the latest game."""
    ids = {normalize_name(p["full_name"]): str(p["id"]) for p in nba_static_players.get_players()}
    for pid, name in conn.execute(
        "SELECT player_id_nba, arg_max(player_name, game_date) FROM nba_player_games GROUP BY 1 ORDER BY max(game_date)"
    ).fetchall():
        ids[normalize_name(name)] = pid
    return ids


def _num(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def box_rows(summary: dict, game_id: str, game_date: date, season: str, home: str, away: str,
             ids: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """nba_player_games rows for everyone with a stat line (DNPs have none),
    plus the names that matched no NBA id."""
    rows, unmatched = [], []
    for team in summary.get("boxscore", {}).get("players", []):
        abbr = nba_abbr(team["team"]["abbreviation"])
        home_away, opponent = ("home", away) if abbr == home else ("away", home)
        for block in team.get("statistics", []):
            keys = block.get("keys", [])
            for a in block.get("athletes", []):
                stats = a.get("stats") or []
                if a.get("didNotPlay") or len(stats) != len(keys):
                    continue
                name = a["athlete"]["displayName"]
                pid = ids.get(normalize_name(name))
                if pid is None:
                    unmatched.append(name)
                    continue
                row = {"player_id_nba": pid, "player_name": name, "team": abbr, "opponent": opponent,
                       "game_id": game_id, "game_date": game_date, "season": season, "home_away": home_away}
                for key, val in zip(keys, stats):
                    col = _STAT_KEYS.get(key)
                    if isinstance(col, tuple):
                        made, att = (val.split("-") + [None])[:2] if val else (None, None)
                        row[col[0]] = _num(made)
                        if col[1]:
                            row[col[1]] = _num(att)
                    elif col:
                        row[col] = _num(val)
                rows.append(row)
    return pd.DataFrame(rows), unmatched


def write_player_games(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, now: datetime) -> int:
    if df.empty:
        return 0
    cols = ["player_id_nba", "player_name", "team", "opponent", "game_id", "game_date", "season", "home_away",
            "minutes", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "fga", "fgm", "fta", "ftm", "oreb", "dreb"]
    df = df.reindex(columns=cols).assign(source_published_at=df["game_date"], ingested_at=now, source="espn")
    conn.register("df_espn_games", df)
    conn.execute(
        f"""
        INSERT INTO nba_player_games ({", ".join(cols)}, source_published_at, ingested_at, source)
        SELECT {", ".join(cols)}, source_published_at, ingested_at, source FROM df_espn_games
        ON CONFLICT (player_id_nba, game_id) DO NOTHING
        """
    )
    conn.unregister("df_espn_games")
    return len(df)


def write_results(conn: duckdb.DuckDBPyConnection, results: list[tuple], now: datetime) -> int:
    """results: [(game_id, season, home_score, away_score)], all final."""
    conn.executemany(
        """
        INSERT INTO nba_game_results VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (game_id) DO UPDATE SET game_status = excluded.game_status,
            home_score = excluded.home_score, away_score = excluded.away_score, updated_at = excluded.updated_at
        """,
        [(g, s, FINAL_STATUS, h, a, now) for g, s, h, a in results],
    )
    return len(results)
