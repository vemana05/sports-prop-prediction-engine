"""Kickoff-hour weather for outdoor NFL games from Open-Meteo (spec
Section 3.4). Free, no API key, ~10,000 calls/day for non-commercial use.

- Played games: the Historical Forecast archive, i.e. what the forecast
  models said around kickoff, not observed weather. Training on forecasts
  matches what's actually knowable before an upcoming game.
- Upcoming games within the 16-day forecast horizon: the live forecast API.
- Domes and closed roofs are skipped (roof from nfl_schedule).

One request per (venue, season) for the archive, covering every outdoor
game there that season — about 30 venues x 6 seasons, far under the daily
limit. Raw JSON responses are cached under data/raw/nfl/weather/ keyed by
venue and date range, so a rebuild doesn't re-request them.

    python -m backend.ingest.weather_openmeteo
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests

from backend.ingest.util import IngestResult, logger, retry, utcnow
from backend.nfl_db import get_nfl_connection

RAW_DIR = Path("data/raw/nfl/weather")
ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_FIRST_SEASON = 2021
ARCHIVE_LAG_DAYS = 3        # archive trails real time by a day or two
FORECAST_HORIZON_DAYS = 15
HOURLY = "temperature_2m,precipitation,wind_speed_10m"
OUTDOOR_ROOFS = ("outdoors", "open")


def _venues() -> dict[str, dict]:
    with open("config/nfl_venues.csv") as f:
        return {r["stadium_id"]: r for r in csv.DictReader(f)}


def _get_hourly(url: str, venue: dict, start: str, end: str, cache_name: str) -> pd.DataFrame:
    path = RAW_DIR / cache_name
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        params = {
            "latitude": venue["lat"], "longitude": venue["lon"], "start_date": start, "end_date": end,
            "hourly": HOURLY, "timezone": "GMT", "wind_speed_unit": "kmh",
        }

        def call():
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()

        payload = retry(call, what=f"open-meteo {venue['stadium_id']} {start}..{end}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    h = payload["hourly"]
    return pd.DataFrame({
        "hour": pd.to_datetime(h["time"], utc=True),
        "temperature_c": h["temperature_2m"],
        "precip_mm": h["precipitation"],
        "wind_kph": h["wind_speed_10m"],
    })


def _games_needing_weather(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        f"""
        SELECT game_id, season, kickoff_utc, venue_id
        FROM nfl_schedule
        WHERE roof IN {OUTDOOR_ROOFS} AND venue_id IS NOT NULL AND kickoff_utc IS NOT NULL
          AND season >= {ARCHIVE_FIRST_SEASON}
        """
    ).fetchdf()


def run(conn: duckdb.DuckDBPyConnection | None = None) -> IngestResult:
    venues = _venues()
    nfl_conn = get_nfl_connection()
    try:
        games = _games_needing_weather(nfl_conn)
        games["kickoff_utc"] = pd.to_datetime(games["kickoff_utc"], utc=True)
        now = utcnow()
        archive_cutoff = now - timedelta(days=ARCHIVE_LAG_DAYS)
        horizon = now + timedelta(days=FORECAST_HORIZON_DAYS)

        rows = []
        for (venue_id, season), grp in games.groupby(["venue_id", "season"]):
            venue = venues[venue_id]
            for source, url, mask in [
                ("historical_forecast", ARCHIVE_URL, grp["kickoff_utc"] < archive_cutoff),
                ("forecast", FORECAST_URL, (grp["kickoff_utc"] >= archive_cutoff) & (grp["kickoff_utc"] <= horizon)),
            ]:
                sub = grp[mask]
                if sub.empty:
                    continue
                start = sub["kickoff_utc"].min().strftime("%Y-%m-%d")
                end = (sub["kickoff_utc"].max() + timedelta(hours=4)).strftime("%Y-%m-%d")
                # Live forecasts change hour to hour; only archive responses are reused.
                cache = f"{venue_id}_{start}_{end}.json" if source == "historical_forecast" else f"forecast/{venue_id}_{now:%Y%m%dT%H}.json"
                hourly = _get_hourly(url, venue, start, end, cache).set_index("hour")
                for g in sub.itertuples():
                    kickoff_hour = g.kickoff_utc.floor("h")
                    if kickoff_hour not in hourly.index:
                        continue
                    w = hourly.loc[kickoff_hour]
                    rows.append({
                        "game_id": g.game_id, "kickoff_utc": g.kickoff_utc.to_pydatetime(),
                        "temperature_c": w["temperature_c"], "wind_kph": w["wind_kph"], "precip_mm": w["precip_mm"],
                        "source": source, "fetched_at": now.replace(tzinfo=None),
                    })

        df = pd.DataFrame(rows)
        if not df.empty:
            nfl_conn.register("_incoming", df)
            nfl_conn.execute("DELETE FROM nfl_weather WHERE game_id IN (SELECT game_id FROM _incoming)")
            nfl_conn.execute(f"INSERT INTO nfl_weather ({', '.join(df.columns)}) SELECT {', '.join(df.columns)} FROM _incoming")
            nfl_conn.unregister("_incoming")
        missing = len(games[games["kickoff_utc"] <= horizon]) - len(df)
        if missing:
            logger.warning("open-meteo: %d outdoor games in range got no weather row", missing)
        return IngestResult(rows_written=len(df), notes={"missing": missing})
    finally:
        nfl_conn.close()


if __name__ == "__main__":
    from backend.ingest.backup_db import backup_warehouse
    from backend.nfl_db import DEFAULT_NFL_DB_PATH

    result = run()
    print(f"nfl_weather: {result.rows_written} rows ({result.notes})")
    backup_warehouse(db_path=DEFAULT_NFL_DB_PATH)
