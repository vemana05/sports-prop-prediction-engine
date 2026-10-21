"""Official NBA injury report parsing: fetch_report on canned DataFrames shaped
like nbainjuries' output. No PDF download, no network."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from backend.ingest import injuries_nba_official as inj

TS = datetime(2026, 10, 21, 13, 0)
COLUMNS = ["Game Date", "Game Time", "Matchup", "Team", "Player Name", "Current Status", "Reason"]


@pytest.fixture
def report(monkeypatch):
    """Feeds fetch_report a canned DataFrame in place of the cached PDF."""
    def use(df):
        monkeypatch.setattr(inj, "_ensure_cached", lambda ts: None)
        monkeypatch.setattr(inj.injury, "get_reportdata", lambda *a, **k: df)
    return use


def test_all_teams_not_yet_submitted_parses_to_zero_rows(report):
    # Early-day report: every team listed, no players, so Player Name is all NaN (float64).
    teams = ["Boston Celtics", "New York Knicks", "Los Angeles Lakers", "Golden State Warriors"]
    report(pd.DataFrame({
        "Game Date": ["10/21/2026"] * 4, "Game Time": ["07:30 (ET)"] * 4,
        "Matchup": ["NYK@BOS", "NYK@BOS", "GSW@LAL", "GSW@LAL"], "Team": teams,
        "Player Name": [np.nan] * 4, "Current Status": [np.nan] * 4, "Reason": ["NOT YET SUBMITTED"] * 4,
    }, columns=COLUMNS))
    out = inj.fetch_report(TS)
    assert out.empty
    assert list(out.columns) == ["game_date", "team", "player_name_raw", "status", "reason"]
    assert inj._write(None, out, TS) == 0  # nothing to write; never touches the connection


def test_normal_report_parses(report):
    report(pd.DataFrame([
        ["10/21/2026", "07:30 (ET)", "NYK@BOS", "Boston Celtics", "Tatum, Jayson", "Out", "Injury/Illness - Achilles"],
        ["10/21/2026", "07:30 (ET)", "NYK@BOS", "New York Knicks", "Brunson, Jalen", "Questionable", "Injury/Illness - Ankle"],
        ["10/21/2026", "07:30 (ET)", "NYK@BOS", "New York Knicks", np.nan, np.nan, np.nan],  # blank PDF row
        ["10/21/2026", "07:30 (ET)", "NYK@BOS", "New York Knicks", "  ", np.nan, np.nan],
        ["10/21/2026", "07:30 (ET)", "NYK@BOS", "New York Knicks", "Brunson, Jalen", "Probable", "Injury/Illness - Ankle"],
        ["10/21/2026", "10:00 (ET)", "GSW@LAL", "Golden State Warriors", np.nan, np.nan, "NOT YET SUBMITTED"],
    ], columns=COLUMNS))
    out = inj.fetch_report(TS)
    assert sorted(out["player_name_raw"]) == ["Brunson, Jalen", "Tatum, Jayson"]
    # A player split across PDF pages keeps the last row.
    assert out.set_index("player_name_raw").loc["Brunson, Jalen", "status"] == "Probable"
    assert set(out["game_date"]) == {pd.Timestamp("2026-10-21").date()}


def test_empty_report(report):
    report(pd.DataFrame(columns=COLUMNS))
    assert inj.fetch_report(TS).empty
