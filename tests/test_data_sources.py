"""Data Sources page: log parsing, lateness, key redaction, usage from the
stored logs, table freshness, model grouping and the review queue. Temp
files and DuckDB files only; no network, no real warehouse."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from backend.api import data_sources as ds
from backend.db import SCHEMA
from backend.nfl_db import NFL_SCHEMA

# Wednesday 2026-10-14 18:00 ET
NOW = datetime(2026, 10, 14, 22, 0, tzinfo=timezone.utc)
SECRET = "sk_live_9f8e7d6c5b4a"


def _log(d: Path, name: str, body: str, age: timedelta = timedelta(days=1)) -> Path:
    p = d / name
    p.write_text(body)
    t = (NOW - age).timestamp()
    os.utime(p, (t, t))
    return p


def test_redact_hides_keys(monkeypatch):
    monkeypatch.setenv("ODDSPAPI_API_KEY", SECRET)
    line = (f"GET https://api.example.com/odds?apiKey=abc123XYZ&x=1 api_key: q1w2e3 Authorization: Bearer tok.en-1 "
            f"raw {SECRET} done")
    out = ds.redact(line)
    for leaked in ("abc123XYZ", "q1w2e3", "tok.en-1", SECRET):
        assert leaked not in out
    assert "apiKey=REDACTED" in out and "&x=1" in out


def test_parse_success_and_failed_logs(tmp_path):
    ok = _log(tmp_path, "daily_odds_job_20261014_100000.log",
              "=== daily_odds_job starting Wed Oct 14 10:00:00 EDT 2026 ===\n"
              "2026-10-14 14:00:05 WARNING ingest: nbainjuries failed validation\n"
              "=== daily_odds_job finished Wed Oct 14 10:02:30 EDT 2026 (predict exit 0, odds exit 0) ===\n")
    r = ds.parse_log(ok, NOW)
    assert (r["outcome"], r["duration_s"]) == ("success", 150)
    assert "failed validation" in r["last_error"]

    bad = _log(tmp_path, "weekly_nfl_job_results_20261013_090000.log",
               "=== weekly_nfl_job (results) starting Tue Oct 13 09:00:00 EDT 2026 ===\n"
               f"requests.exceptions.ConnectionError: https://api.oddspapi.io/v4/odds?apiKey={SECRET}\n"
               "=== weekly_nfl_job (results) finished Tue Oct 13 09:40:00 EDT 2026 (exit 1) ===\n")
    r = ds.parse_log(bad, NOW)
    assert r["outcome"] == "failed" and r["duration_s"] == 2400
    assert SECRET not in r["last_error"] and "REDACTED" in r["last_error"]

    docker = _log(tmp_path, "weekly_nfl_job_prep_20261010_090000.log",
                  "=== weekly_nfl_job (prep) starting Sat Oct 10 09:00:00 EDT 2026 ===\n"
                  "ERROR: Docker daemon is not running — cannot proceed. Is Docker Desktop open?\n")
    assert ds.parse_log(docker, NOW)["outcome"] == "failed"

    crashed = _log(tmp_path, "weekly_nfl_job_lines_20261007_090000.log", "=== weekly_nfl_job (lines) starting ===\n")
    r = ds.parse_log(crashed, NOW)
    assert r["outcome"] == "failed" and "never logged a finish" in r["last_error"]

    running = _log(tmp_path, "weekly_nfl_job_lines_20261014_175000.log", "=== starting ===\n", age=timedelta(minutes=5))
    assert ds.parse_log(running, NOW)["outcome"] == "running"


def test_last_scheduled():
    daily, tue, sat = ds.JOBS[0], ds.JOBS[3], ds.JOBS[1]
    assert ds.last_scheduled(daily, NOW).isoformat() == "2026-10-14T10:00:00-04:00"
    assert ds.last_scheduled(tue, NOW).isoformat() == "2026-10-13T09:00:00-04:00"
    assert ds.last_scheduled(sat, NOW).isoformat() == "2026-10-10T09:00:00-04:00"
    early = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)  # 8:00 ET, before today's 10:00 run
    assert ds.last_scheduled(daily, early).isoformat() == "2026-10-13T10:00:00-04:00"


def test_job_lateness(tmp_path):
    sat = ds.JOBS[1]
    assert ds.job_status(sat, tmp_path, NOW)["outcome"] == "never ran"
    assert ds.job_status(sat, tmp_path, NOW)["late"]  # due Saturday, 4 days ago
    # A manual run at 8:10 the morning of a 9:00 slot covers it.
    _log(tmp_path, "weekly_nfl_job_prep_20261010_081000.log", "=== x finished Sat Oct 10 08:40:00 EDT 2026 (exit 0) ===\n")
    assert not ds.job_status(sat, tmp_path, NOW)["late"]
    # The last run was the week before: more than a day past Saturday's slot.
    tue = ds.JOBS[3]
    _log(tmp_path, "weekly_nfl_job_results_20261006_090000.log", "=== x finished Tue Oct 6 09:30:00 EDT 2026 (exit 0) ===\n")
    assert ds.job_status(tue, tmp_path, NOW)["late"]
    # Not late within a day of the slot.
    assert not ds.job_status(tue, tmp_path, datetime(2026, 10, 13, 20, 0, tzinfo=timezone.utc))["late"]


@pytest.fixture
def warehouses(tmp_path):
    wh, nfl = str(tmp_path / "wh.duckdb"), str(tmp_path / "nfl.duckdb")
    c = duckdb.connect(wh)
    c.execute(SCHEMA)
    c.execute("INSERT INTO odds_api_usage_log VALUES ('2026-10-14 14:00:00', 'events', 7, 493, 'success', NULL)")
    c.execute("INSERT INTO odds_api_usage_log VALUES ('2026-10-14 14:00:01', 'odds', NULL, NULL, 'error', NULL)")
    c.execute("INSERT INTO odds_snapshots VALUES ('2026-10-02 19:00:00', 'e1', NULL, NULL, 'A B', 'player_points', 'fd', 20.5, 1.9, 1.9)")
    c.execute("INSERT INTO xref_review_queue VALUES ('sleeper', 'Jon Doe', '1', 'no match', 'John Doe', 0.81, '2026-10-02', false), "
              "('sleeper', 'Done Already', '2', 'no match', NULL, NULL, '2026-10-02', true)")
    c.close()
    n = duckdb.connect(nfl)
    n.execute(NFL_SCHEMA)
    n.execute("INSERT INTO oddspapi_usage_log VALUES ('2026-09-30 13:00:00', 'account', 180, 250, 'account', 'success', NULL), "
              "('2026-10-14 13:00:00', 'account', 30, 250, 'account', 'success', NULL), "
              "('2026-10-14 13:01:00', '/odds', 31, 250, 'estimate', 'success', NULL)")
    n.execute("INSERT INTO nfl_oddspapi_fixtures (fixture_id, game_id, season, week, kickoff_utc, home_team, away_team) "
              "VALUES ('fx1', '2026_06_BUF_KC', 2026, 6, '2026-10-18 17:00:00+00', 'KC', 'BUF')")
    n.execute("INSERT INTO nfl_oddspapi_players VALUES ('fx1', '9', 'Ward, Cameron', NULL), ('fx1', '8', 'Mahomes, Patrick', '00-1')")
    n.execute("INSERT INTO nfl_schedule (game_id, season, week, game_type, home_team, away_team, ingested_at) "
              "VALUES ('2026_06_BUF_KC', 2026, 6, 'REG', 'KC', 'BUF', '2026-10-13 13:00:00')")
    n.close()
    return wh, nfl


def test_build_status(tmp_path, warehouses, monkeypatch):
    monkeypatch.setenv("ODDS_API_MONTHLY_CREDIT_BUDGET", "450")
    monkeypatch.setenv("ODDSPAPI_MONTHLY_REQUEST_BUDGET", "200")
    monkeypatch.setenv("ODDS_API_KEY", SECRET)
    logs, models = tmp_path / "logs", tmp_path / "models"
    logs.mkdir()
    models.mkdir()
    for f in ("nfl_receptions_thru2025.txt", "nfl_receptions_thru2025_spread.pkl", "nfl_passing_yards_thru2025.txt",
              "nfl_passing_yards_thru2026w04.txt", "points.txt", "points_calibration.pkl"):
        (models / f).write_text("x")
    _log(logs, "daily_odds_job_20261014_100000.log", f"apiKey={SECRET} Error\n=== d finished Wed Oct 14 10:01:00 EDT 2026 (predict exit 0, odds exit 1) ===\n")

    s = ds.build_status(NOW, logs, *warehouses, models)
    assert SECRET not in json.dumps(s)

    usage = {u["name"]: u for u in s["usage"]}
    assert usage["OddsPapi"]["used"] == 31 and usage["OddsPapi"]["limit"] == 250 and usage["OddsPapi"]["budget"] == 200
    assert usage["The Odds API"]["used"] == 7 and usage["The Odds API"]["limit"] == 500 and usage["The Odds API"]["budget"] == 450

    tables = {t["table"]: t for w in s["warehouses"] for t in w["tables"]}
    assert tables["odds_snapshots"]["late"] and tables["odds_snapshots"]["rows"] == 1
    assert not tables["odds_api_usage_log"]["late"]
    assert not tables["nfl_schedule"]["late"]
    assert tables["nfl_weather"]["cadence_days"] is None and not tables["nfl_weather"]["late"]  # on demand: never flagged
    assert tables["model_predictions_log"]["late"]  # empty and expected daily

    jobs = {j["key"]: j for j in s["jobs"]}
    assert jobs["daily_nba"]["outcome"] == "failed"
    assert jobs["nfl_lines"]["outcome"] == "never ran" and not jobs["nfl_lines"]["late"]  # due 9:00 today
    assert jobs["nfl_prep"]["late"]

    versions = {(m["name"], m["version"]): m for m in s["models"]}
    assert versions[("NFL market models", "thru2025")]["markets"] == ["passing_yards", "receptions"]
    assert ("NFL market models", "thru2026w04") in versions and ("points.txt", None) in versions

    q = s["review_queue"]
    assert q["total"] == 2 and q["by_source"] == {"sleeper": 1, "oddspapi_nfl": 1}
    assert {r["name"] for r in q["rows"]} == {"Jon Doe", "Ward, Cameron"}

    flags = " | ".join(s["flags"])
    assert "Daily NBA: last run failed" in flags and "NFL prep (Saturday) is late" in flags
    assert "odds_snapshots is stale" in flags and "2 unmatched player names" in flags


def test_usage_over_cap_and_missing_warehouse(tmp_path, warehouses, monkeypatch):
    monkeypatch.setenv("ODDSPAPI_MONTHLY_REQUEST_BUDGET", "30")
    s = ds.build_status(NOW, tmp_path / "nologs", str(tmp_path / "missing.duckdb"), warehouses[1], tmp_path / "nomodels")
    assert any("OddsPapi: 31 used, at or over our 30 cap" in f for f in s["flags"])
    assert s["warehouses"][0]["error"].endswith("not found")
    assert next(u for u in s["usage"] if u["name"] == "The Odds API")["used"] is None
    assert s["models"] == [] and all(j["outcome"] == "never ran" for j in s["jobs"])
