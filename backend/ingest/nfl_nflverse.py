"""NFL ingestion from nflverse via nflreadpy (spec Section 3.1, Phase 6).

Loads player stats, snap counts, schedules, injuries, depth charts and
weekly rosters into data/nfl_warehouse.duckdb (backend/nfl_db.py).

Every download is saved to data/raw/nfl/<loader>/<season>.parquet first and
loaded from there. Completed seasons are read from that cache on later
runs, so a rebuild never re-downloads them; the current season is always
re-downloaded because nflverse updates it weekly.

Seasons default to 2020 through the current one. 2020 is a warm-up season:
it gives 2021 games real prior-season features, but Open-Meteo's
historical-forecast archive starts in 2021, so 2020 outdoor games have no
weather.

    python -m backend.ingest.nfl_nflverse            # all stages, backup after each
    python -m backend.ingest.nfl_nflverse --stage schedule
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from backend.ingest.util import IngestResult, logger, retry, utcnow
from backend.nfl_db import get_nfl_connection

RAW_DIR = Path("data/raw/nfl")
FIRST_SEASON = 2020
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
EASTERN = ZoneInfo("America/New_York")


def current_season(today: datetime | None = None) -> int:
    """NFL seasons are named by the year they start; anything before
    September belongs to the previous season."""
    today = today or utcnow()
    return today.year if today.month >= 9 else today.year - 1


def all_seasons() -> list[int]:
    return list(range(FIRST_SEASON, current_season() + 1))


def _fetch(loader: str, season: int, refresh_current: bool = True) -> pd.DataFrame:
    """Download one loader-season to the raw cache (or reuse the cache)
    and return it as pandas."""
    import nflreadpy as nfl

    path = RAW_DIR / loader / f"{season}.parquet"
    if path.exists() and not (refresh_current and season == current_season()):
        return pd.read_parquet(path)

    fn = getattr(nfl, f"load_{loader}")
    df = retry(lambda: fn([season]), what=f"nflreadpy load_{loader}({season})")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    logger.info("nflverse: saved %s (%d rows)", path, df.height)
    return df.to_pandas()


def _replace(conn: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame, seasons: list[int]) -> int:
    """Season-scoped replace: delete then insert, so a re-run is idempotent
    and a partial season (current year) is fully refreshed."""
    conn.execute(f"DELETE FROM {table} WHERE season IN ({','.join(str(s) for s in seasons)})")
    if df.empty:
        return 0
    conn.register("_incoming", df)
    cols = ", ".join(df.columns)
    conn.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _incoming")
    conn.unregister("_incoming")
    return len(df)


def _load_venues(path: str = "config/nfl_venues.csv") -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _match_venue(stadium_id: str | None, stadium: str | None, venues: list[dict]) -> dict | None:
    """Match by stadium name first: nflverse occasionally reuses a home
    team's stadium_id for an international game (e.g. JAX00 for a game at
    Tottenham), so the id alone would put London's weather in Jacksonville."""
    if stadium:
        for v in venues:
            if stadium in v["names"].split("|"):
                return v
    for v in venues:
        if v["stadium_id"] == stadium_id:
            return v
    return None


def _kickoff_utc(gameday: str | None, gametime: str | None) -> datetime | None:
    if not gameday or not gametime:
        return None
    local = datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=EASTERN)
    return local.astimezone(ZoneInfo("UTC"))


def ingest_schedule(conn: duckdb.DuckDBPyConnection, seasons: list[int]) -> int:
    venues = _load_venues()
    frames = []
    for season in seasons:
        raw = _fetch("schedules", season)
        df = pd.DataFrame({
            "game_id": raw["game_id"],
            "season": raw["season"],
            "week": raw["week"],
            "game_type": raw["game_type"],
            "kickoff_utc": [_kickoff_utc(d, t) for d, t in zip(raw["gameday"], raw["gametime"])],
            "home_team": raw["home_team"],
            "away_team": raw["away_team"],
            "home_score": raw["home_score"],
            "away_score": raw["away_score"],
            "home_rest": raw["home_rest"],
            "away_rest": raw["away_rest"],
            "div_game": raw["div_game"].astype("boolean"),
            "roof": raw["roof"],
            "surface": raw["surface"],
            "stadium_id": raw["stadium_id"],
            "stadium": raw["stadium"],
        })
        matched = [_match_venue(sid, name, venues) for sid, name in zip(df["stadium_id"], df["stadium"])]
        unmatched = sorted({f"{sid} {name}" for sid, name, m in zip(df["stadium_id"], df["stadium"], matched) if m is None})
        if unmatched:
            logger.warning("nflverse schedule %d: no venue row for %s (no weather for these games)", season, unmatched)
        df["venue_id"] = [m["stadium_id"] if m else None for m in matched]
        df["roof"] = [r if isinstance(r, str) else (m["default_roof"] if m else None) for r, m in zip(df["roof"], matched)]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["ingested_at"] = utcnow().replace(tzinfo=None)
    return _replace(conn, "nfl_schedule", df, seasons)


PLAYER_STAT_COLUMNS = {
    "player_id": "player_id", "player_display_name": "player_name", "position": "position",
    "position_group": "position_group", "season": "season", "week": "week", "season_type": "season_type",
    "game_id": "game_id", "team": "team", "opponent_team": "opponent",
    "completions": "completions", "attempts": "attempts", "passing_yards": "passing_yards",
    "passing_tds": "passing_tds", "passing_interceptions": "interceptions", "sacks_suffered": "sacks",
    "passing_air_yards": "passing_air_yards", "passing_epa": "passing_epa",
    "carries": "carries", "rushing_yards": "rushing_yards", "rushing_tds": "rushing_tds", "rushing_epa": "rushing_epa",
    "receptions": "receptions", "targets": "targets", "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds", "receiving_air_yards": "receiving_air_yards",
    "target_share": "target_share", "air_yards_share": "air_yards_share",
}


def ingest_player_stats(conn: duckdb.DuckDBPyConnection, seasons: list[int]) -> int:
    frames = []
    for season in seasons:
        raw = _fetch("player_stats", season)
        df = raw[list(PLAYER_STAT_COLUMNS)].rename(columns=PLAYER_STAT_COLUMNS)
        frames.append(df[df["position"].isin(SKILL_POSITIONS)])
    df = pd.concat(frames, ignore_index=True)
    df["ingested_at"] = utcnow().replace(tzinfo=None)
    return _replace(conn, "nfl_player_weekly", df, seasons)


def ingest_rosters(conn: duckdb.DuckDBPyConnection, seasons: list[int]) -> int:
    frames = []
    for season in seasons:
        raw = _fetch("rosters_weekly", season)
        df = pd.DataFrame({
            "season": raw["season"], "week": raw["week"], "team": raw["team"],
            "player_id": raw["gsis_id"], "player_name": raw["full_name"], "position": raw["position"],
            "status": raw["status"], "pfr_id": raw["pfr_id"], "sleeper_id": raw["sleeper_id"],
            "years_exp": raw["years_exp"], "rookie_year": raw["rookie_year"],
        })
        frames.append(df.dropna(subset=["player_id"]).drop_duplicates(["season", "week", "team", "player_id"]))
    df = pd.concat(frames, ignore_index=True)
    df["ingested_at"] = utcnow().replace(tzinfo=None)
    return _replace(conn, "nfl_rosters_weekly", df, seasons)


def ingest_snap_counts(conn: duckdb.DuckDBPyConnection, seasons: list[int]) -> int:
    """Runs after ingest_rosters: fills player_id from the roster's
    pfr_id -> gsis crosswalk."""
    frames = []
    for season in seasons:
        raw = _fetch("snap_counts", season)
        df = pd.DataFrame({
            "game_id": raw["game_id"], "season": raw["season"], "week": raw["week"],
            "pfr_player_id": raw["pfr_player_id"], "player_name": raw["player"],
            "position": raw["position"], "team": raw["team"],
            "offense_snaps": raw["offense_snaps"], "offense_pct": raw["offense_pct"],
        })
        frames.append(df[df["position"].isin(SKILL_POSITIONS)].drop_duplicates(["game_id", "pfr_player_id"]))
    df = pd.concat(frames, ignore_index=True)
    xref = conn.execute(
        "SELECT DISTINCT pfr_id, player_id FROM nfl_rosters_weekly WHERE pfr_id IS NOT NULL"
    ).fetchdf().drop_duplicates("pfr_id")
    df = df.merge(xref.rename(columns={"pfr_id": "pfr_player_id"}), on="pfr_player_id", how="left")
    df["ingested_at"] = utcnow().replace(tzinfo=None)
    n = _replace(conn, "nfl_snap_counts", df, seasons)
    unmatched = int(df["player_id"].isna().sum())
    if unmatched:
        logger.warning("nflverse snap_counts: %d of %d rows have no pfr->gsis match", unmatched, len(df))
    return n


def ingest_injuries(conn: duckdb.DuckDBPyConnection, seasons: list[int]) -> int:
    frames = []
    for season in seasons:
        raw = _fetch("injuries", season)
        df = pd.DataFrame({
            "season": raw["season"], "week": raw["week"], "game_type": raw["game_type"], "team": raw["team"],
            "player_id": raw["gsis_id"], "player_name": raw["full_name"], "position": raw["position"],
            "report_status": raw["report_status"], "practice_status": raw["practice_status"],
            "primary_injury": raw["report_primary_injury"],
        })
        frames.append(df.dropna(subset=["player_id"]).drop_duplicates(["season", "week", "team", "player_id"], keep="last"))
    df = pd.concat(frames, ignore_index=True)
    df["ingested_at"] = utcnow().replace(tzinfo=None)
    return _replace(conn, "nfl_injuries", df, seasons)


def ingest_depth_charts(conn: duckdb.DuckDBPyConnection, seasons: list[int]) -> int:
    frames = []
    for season in seasons:
        raw = _fetch("depth_charts", season)
        if "dt" in raw.columns:  # 2025+ format: dated snapshots
            df = raw[raw["pos_abb"].isin(SKILL_POSITIONS) & raw["gsis_id"].notna()]
            # A player can appear in several formation groups; keep his best rank.
            df = (
                df.groupby(["dt", "team", "gsis_id", "pos_abb"], as_index=False)
                .agg(player_name=("player_name", "first"), depth_rank=("pos_rank", "min"))
            )
            frames.append(pd.DataFrame({
                "season": season, "week": pd.NA,
                "snapshot_at": pd.to_datetime(df["dt"], utc=True),
                "team": df["team"], "player_id": df["gsis_id"], "player_name": df["player_name"],
                "position": df["pos_abb"], "depth_rank": df["depth_rank"],
            }))
        else:  # through 2024: one chart per week
            pos = raw["depth_position"].where(raw["depth_position"].isin(SKILL_POSITIONS), raw["position"])
            df = raw.assign(pos=pos)
            df = df[df["pos"].isin(SKILL_POSITIONS) & df["gsis_id"].notna() & (df["formation"] == "Offense")]
            df = df.assign(depth_rank=pd.to_numeric(df["depth_team"], errors="coerce")).dropna(subset=["depth_rank"])
            df = (
                df.groupby(["week", "club_code", "gsis_id", "pos"], as_index=False)
                .agg(player_name=("full_name", "first"), depth_rank=("depth_rank", "min"))
            )
            frames.append(pd.DataFrame({
                "season": season, "week": df["week"], "snapshot_at": pd.NaT,
                "team": df["club_code"], "player_id": df["gsis_id"], "player_name": df["player_name"],
                "position": df["pos"], "depth_rank": df["depth_rank"].astype(int),
            }))
    df = pd.concat(frames, ignore_index=True)
    return _replace(conn, "nfl_depth_charts", df, seasons)


STAGES = {
    "schedule": ingest_schedule,
    "player_stats": ingest_player_stats,
    "rosters": ingest_rosters,
    "snap_counts": ingest_snap_counts,
    "injuries": ingest_injuries,
    "depth_charts": ingest_depth_charts,
}


def run(conn: duckdb.DuckDBPyConnection | None = None, seasons: list[int] | None = None) -> IngestResult:
    """Runner entrypoint (config/sources.yaml). Ignores an NBA connection
    if one is passed: NFL data lives in its own warehouse file."""
    seasons = seasons or all_seasons()
    nfl_conn = get_nfl_connection()
    try:
        counts = {name: fn(nfl_conn, seasons) for name, fn in STAGES.items()}
    finally:
        nfl_conn.close()
    return IngestResult(rows_written=sum(counts.values()), notes=counts)


if __name__ == "__main__":
    import argparse

    from backend.ingest.backup_db import backup_warehouse
    from backend.nfl_db import DEFAULT_NFL_DB_PATH

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=list(STAGES), action="append", help="run only these stages (repeatable)")
    parser.add_argument("--seasons", type=int, nargs="*", help="default: 2020 through the current season")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    seasons = args.seasons or all_seasons()
    for name in args.stage or list(STAGES):
        conn = get_nfl_connection()
        try:
            n = STAGES[name](conn, seasons)
        finally:
            conn.close()  # checkpoint before the file copy below
        print(f"{name}: {n} rows")
        if not args.no_backup:
            backup_warehouse(db_path=DEFAULT_NFL_DB_PATH)
