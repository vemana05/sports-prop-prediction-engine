"""The credit-budget guard is the one thing standing between this project
and accidentally blowing through The Odds API's free tier (500 credits/
month). These tests exercise its logic directly, against an isolated
in-memory database (not the shared read-only `conn` fixture, since this
needs to write), without hitting the network."""

import os
from datetime import timedelta

import duckdb

from backend.db import SCHEMA
from backend.ingest.odds_theoddsapi import _budget, _budget_ok, _last_known_usage
from backend.ingest.util import utcnow


def _isolated_conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute(SCHEMA)
    return conn


def _log_usage(conn, used, minutes_ago=0):
    conn.execute(
        "INSERT INTO odds_api_usage_log VALUES (?, ?, ?, ?, ?, ?)",
        [utcnow() - timedelta(minutes=minutes_ago), f"test-{used}-{id(object())}", used, max(0, 500 - used), "success", None],
    )


def test_budget_ok_with_no_usage_logged_yet():
    conn = _isolated_conn()
    assert _last_known_usage(conn) == 0
    assert _budget_ok(conn) is True


def test_budget_blocks_once_usage_reaches_the_configured_budget():
    conn = _isolated_conn()
    budget = _budget()
    _log_usage(conn, budget)  # exactly at budget
    assert _budget_ok(conn) is False


def test_budget_ok_just_under_the_configured_budget():
    conn = _isolated_conn()
    budget = _budget()
    _log_usage(conn, budget - 1)
    assert _budget_ok(conn) is True


def test_last_known_usage_reads_the_most_recent_call_not_the_max():
    """Usage can only go up within a billing month, but the guard should
    trust the LATEST reading (closest to ground truth), not assume
    monotonicity it doesn't actually need to assume."""
    conn = _isolated_conn()
    _log_usage(conn, 100, minutes_ago=10)
    _log_usage(conn, 150, minutes_ago=0)
    assert _last_known_usage(conn) == 150


def test_default_budget_leaves_headroom_under_the_real_free_tier_cap():
    # ODDS_API_MONTHLY_CREDIT_BUDGET defaults to 450, deliberately under
    # the real 500/mo cap — if this ever gets set to >= 500 by mistake,
    # the guard provides no real protection.
    old = os.environ.pop("ODDS_API_MONTHLY_CREDIT_BUDGET", None)
    try:
        assert _budget() < 500
    finally:
        if old is not None:
            os.environ["ODDS_API_MONTHLY_CREDIT_BUDGET"] = old
