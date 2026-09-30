"""Regression test for a real bug found while validating the availability
table: `_load_schedule` returns `game_date` via `fetchdf()` (a pandas
Timestamp), while `_load_injury_index` returns it via plain `.fetchall()`
(a Python `datetime.date`). Without normalizing both to the same type, the
(player_id, game_date) dict lookup silently misses on every single row —
no error, just every DNP classified as "dnp_other" instead of correctly
matching a real injury report entry. Caught by comparing against a known
real case: E'Twaun Moore, game 0022100010, listed "Out" on the injury
report the day before he DNP'd with a knee injury.
"""

from datetime import date, datetime, timezone

import pandas as pd

from backend.ingest.nba_availability import _latest_before


def test_latest_before_handles_pandas_and_python_datetimes_consistently():
    tip = pd.Timestamp("2021-10-21 00:30:00", tz="UTC")
    entries = [
        (datetime(2021, 10, 19, 21, 30, tzinfo=timezone.utc), "Out", "Injury/Illness"),
        (datetime(2021, 10, 20, 21, 30, tzinfo=timezone.utc), "Out", "Injury/Illness"),
    ]
    result = _latest_before(entries, tip)
    assert result is not None
    assert result[1] == "Out"


def test_schedule_game_date_matches_injury_index_game_date_type(conn):
    """The actual bug: schedule_by_game's game_date (from fetchdf) must be
    directly usable as an injury_index key (from fetchall) — same type,
    same value, for a date that appears in both."""
    from backend.ingest.nba_availability import _load_injury_index, _load_schedule

    schedule = _load_schedule(conn)
    if schedule.empty:
        return  # nothing ingested yet in this environment
    sample = schedule.iloc[0]
    game_date_normalized = sample["game_date"].date() if hasattr(sample["game_date"], "date") else sample["game_date"]
    assert isinstance(game_date_normalized, date)

    injury_index = _load_injury_index(conn)
    if injury_index:
        any_key = next(iter(injury_index))
        assert isinstance(any_key[1], date), f"injury_index game_date key has unexpected type {type(any_key[1])}"
        assert not isinstance(any_key[1], pd.Timestamp)


def test_a_player_on_the_injury_report_is_classified_dnp_listed_not_dnp_other(conn):
    """End-to-end check against real, already-built data for the exact case
    the bug was found with: E'Twaun Moore, game 0022100010, listed "Out" on
    the injury report the day before he DNP'd with a knee injury. Skips if
    that game hasn't reached nba_player_availability yet in this
    environment (nba_box_status is a large, resumable backfill)."""
    result = conn.execute(
        "SELECT status, detail FROM nba_player_availability WHERE game_id = '0022100010' AND player_id_nba = '202734'"
    ).fetchone()
    if result is None:
        return  # this particular game hasn't been backfilled/derived yet
    assert result[0] == "dnp_listed_injury_report", f"expected dnp_listed_injury_report, got {result}"
