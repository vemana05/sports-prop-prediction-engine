"""Regression test for a real bug found while building the forward-
prediction pipeline: `build_current_form` computed `rest_days` as
`as_of - last["game_date"]`, but `last["game_date"]` (read via
`fetchdf()`) is a pandas Timestamp, not the plain `datetime.date` that
`as_of` (from `date.today()`) is — subtracting them raised `TypeError`
immediately, on every call. Same bug class as nba_availability.py's
(see that module's own regression test); worth a dedicated one here since
it's a different function."""

from datetime import date, timedelta

from backend.features.nba_current_form import build_current_form


def test_build_current_form_runs_without_a_date_type_error(conn):
    df = build_current_form(conn, as_of=date.today())
    assert isinstance(df, type(df))  # just: it returned, instead of raising TypeError
    if not df.empty:
        assert df["rest_days"].dtype.kind in "if"  # numeric, not an unresolved timedelta/object


def test_rest_days_is_plausible_for_a_known_recent_window(conn):
    """A loose sanity check: for an as_of date right after the last
    ingested game, rest_days for players who played recently should be
    small non-negative integers, not negative (which would mean the date
    arithmetic is backwards) or absurdly large for everyone."""
    row = conn.execute("SELECT MAX(game_date) FROM nba_player_games").fetchone()
    if row is None or row[0] is None:
        return
    last_ingested_date = row[0]
    as_of = last_ingested_date + timedelta(days=1)
    df = build_current_form(conn, as_of=as_of)
    if df.empty:
        return
    assert (df["rest_days"] >= 0).all(), "found negative rest_days — date subtraction is backwards"
