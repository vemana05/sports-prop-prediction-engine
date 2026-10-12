"""OddsPapi NFL collector: request-budget guard, the /account body never
reaching disk, and line extraction. Fake HTTP and an in-memory NFL
warehouse throughout — no test touches the network."""

import gzip
import json
from datetime import datetime, timedelta, timezone

import duckdb
import pytest

from backend.ingest import odds_oddspapi_nfl as c
from backend.nfl_db import NFL_SCHEMA

NOW = datetime(2026, 10, 7, 14, 0, tzinfo=timezone.utc)  # a Wednesday
KICK = datetime(2026, 10, 11, 17, 0, tzinfo=timezone.utc)

# Two lines each for one market: marketId -> (handicap, over outcomeId, under outcomeId)
MARKETS = [
    {"marketId": 100, "marketType": "playertotals-receivingyards", "period": "result", "handicap": 45.5,
     "outcomes": [{"outcomeId": 100, "outcomeName": "Over"}, {"outcomeId": 101, "outcomeName": "Under"}]},
    {"marketId": 102, "marketType": "playertotals-receivingyards", "period": "result", "handicap": 50.5,
     "outcomes": [{"outcomeId": 102, "outcomeName": "Over"}, {"outcomeId": 103, "outcomeName": "Under"}]},
    {"marketId": 200, "marketType": "playertotals-tdpasses", "period": "result", "handicap": 1.5,
     "outcomes": [{"outcomeId": 200, "outcomeName": "Over"}, {"outcomeId": 201, "outcomeName": "Under"}]},
]


class Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code, self.text = body, status, json.dumps(body)[:100]

    def json(self):
        return self.body


def _p(price, name="Ja'Marr Chase", main=False, active=True):
    return {"active": active, "price": price, "playerName": name, "mainLine": main, "changedAt": "2026-10-07T12:00:00Z"}


ODDS = {
    "fixtureId": "fx1",
    "bookmakerOdds": {
        "fanduel": {"markets": {
            "100": {"outcomes": {"100": {"players": {"77": _p(1.70)}}, "101": {"players": {"77": _p(2.10)}}}},
            "102": {"outcomes": {"102": {"players": {"77": _p(1.91)}}, "103": {"players": {"77": _p(1.91)}}}},
            "200": {"outcomes": {"200": {"players": {"77": _p(1.5)}}, "201": {"players": {"77": _p(2.5)}}}},
        }},
        "pinnacle": {"markets": {"100": {"outcomes": {"100": {"players": {"77": _p(1.9)}}}}}},
    },
}


def _snap(t, price, active=True):
    return {"createdAt": t, "price": price, "active": active}


HIST = {
    "fixtureId": "fx1",
    "bookmakers": {"draftkings": {"markets": {
        # 45.5: even before kickoff, then pulled after kickoff (ignored)
        "100": {"outcomes": {
            "100": {"players": {"77": [_snap("2026-10-08T00:00:00Z", 1.8), _snap("2026-10-11T16:50:00Z", 1.92), _snap("2026-10-11T18:00:00Z", 1.2)]}},
            "101": {"players": {"77": [_snap("2026-10-08T00:00:00Z", 2.0), _snap("2026-10-11T16:50:00Z", 1.90)]}}}},
        # 50.5: closer to even at close, but suspended -> not the close
        "102": {"outcomes": {
            "102": {"players": {"77": [_snap("2026-10-11T16:55:00Z", 1.91, active=False)]}},
            "103": {"players": {"77": [_snap("2026-10-11T16:55:00Z", 1.91, active=False)]}}}},
    }}},
}

FIXTURES = [{"fixtureId": "fx1", "tournamentId": 31, "statusId": 0, "startTime": "2026-10-11T17:00:00.000Z",
             "participant1Abbr": "CIN", "participant2Abbr": "MIA"}]


def _conn():
    conn = duckdb.connect(":memory:")
    conn.execute(NFL_SCHEMA)
    conn.execute("INSERT INTO nfl_schedule (game_id, season, week, game_type, kickoff_utc, home_team, away_team, ingested_at) "
                 "VALUES ('2026_05_CIN_MIA', 2026, 5, 'REG', ?, 'MIA', 'CIN', now())", [KICK])
    conn.execute("INSERT INTO nfl_rosters_weekly (season, week, team, player_id, player_name, ingested_at) "
                 "VALUES (2026, 5, 'CIN', '00-0036900', 'Ja''Marr Chase', now())")
    return conn


class FakeHTTP:
    def __init__(self, account_count=10):
        self.calls = []
        self.account_count = account_count

    def __call__(self, url, params, timeout):
        path = url.removeprefix(c.BASE_URL)
        self.calls.append(path)
        return Resp({
            "/account": {"request_count": self.account_count, "request_limit": 250, "last_request": None, "email": "secret"},
            "/markets": MARKETS,
            "/fixtures": FIXTURES,
            "/odds": ODDS,
            "/historical-odds": HIST,
        }[path])


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CACHE_ROOT", tmp_path / "oddspapi")
    monkeypatch.setenv("ODDSPAPI_API_KEY", "test-key")
    monkeypatch.setenv("ODDSPAPI_COLLECT_ENABLED", "1")
    monkeypatch.delenv("ODDSPAPI_MONTHLY_REQUEST_BUDGET", raising=False)


def _run(step, conn, http, now=NOW, **kw):
    return c.run(step, conn=conn, http_get=http, sleep=lambda s: None, now=lambda: now, **kw)


# ---- guard -----------------------------------------------------------------------

def test_default_budget_is_200_of_250():
    assert c._budget() == 200


def test_missing_key_spends_nothing(monkeypatch):
    monkeypatch.delenv("ODDSPAPI_API_KEY")
    http = FakeHTTP()
    assert "skipped" in _run("wednesday", _conn(), http)
    assert http.calls == []


def test_not_enabled_spends_nothing(monkeypatch):
    monkeypatch.delenv("ODDSPAPI_COLLECT_ENABLED")
    http = FakeHTTP()
    assert "skipped" in _run("wednesday", _conn(), http)
    assert http.calls == []


def test_logged_count_at_budget_skips_even_the_account_call():
    conn, http = _conn(), FakeHTTP()
    conn.execute("INSERT INTO oddspapi_usage_log VALUES (?, '/odds', 200, 250, 'estimate', 'success', NULL)",
                 [NOW.replace(tzinfo=None) - timedelta(days=1)])
    assert "stopped" in _run("wednesday", conn, http)
    assert http.calls == []


def test_last_months_count_does_not_block_this_month():
    conn = _conn()
    conn.execute("INSERT INTO oddspapi_usage_log VALUES (?, '/odds', 240, 250, 'estimate', 'success', NULL)",
                 [datetime(2026, 9, 28)])
    assert c.last_known_count(conn, NOW) == 0


def test_account_at_budget_stops_after_the_account_call():
    http = FakeHTTP(account_count=199)  # +1 for the /account call itself -> 200
    result = _run("wednesday", _conn(), http)
    assert "stopped" in result
    assert http.calls == ["/account"]


def test_never_passes_budget_mid_run():
    http = FakeHTTP(account_count=197)  # 198 after /account; room for 2 more
    result = _run("wednesday", _conn(), http)
    assert "stopped" in result
    assert http.calls == ["/account", "/markets", "/fixtures"]
    assert result["account_count"] == 200


def test_account_body_is_never_written(tmp_path):
    conn = _conn()
    _run("wednesday", conn, FakeHTTP())
    for f in (tmp_path / "oddspapi").rglob("*"):
        if f.is_file():
            text = gzip.decompress(f.read_bytes()).decode()
            assert "request_count" not in text and "secret" not in text
    logged = conn.execute("SELECT endpoint, request_count, request_limit, source FROM oddspapi_usage_log ORDER BY called_at, endpoint").fetchall()
    assert ("/account", 11, 250, "account") in logged


# ---- end to end, offline ------------------------------------------------------------

def test_wednesday_then_tuesday():
    conn, http = _conn(), FakeHTTP()
    wed = _run("wednesday", conn, http)
    assert wed["collected"] == 1 and http.calls == ["/account", "/markets", "/fixtures", "/odds"]
    assert conn.execute("SELECT player_name, player_id FROM nfl_oddspapi_players").fetchall() == [("Ja'Marr Chase", "00-0036900")]
    # Pinnacle and the TD-passes market are dropped; the even-money 50.5 is the open.
    assert conn.execute("SELECT bookmaker, market, line, over_price FROM nfl_prop_lines WHERE kind = 'open'").fetchall() == [
        ("fanduel", "receiving_yards", 50.5, 1.91)]

    # Rerun is free: everything comes from the raw cache.
    http2 = FakeHTTP()
    _run("wednesday", conn, http2)
    assert http2.calls == ["/account"]

    # Tuesday before the game is over does nothing; after, it rebuilds the close.
    assert _run("tuesday", conn, FakeHTTP(), now=KICK + timedelta(hours=1))["games"] == 0
    http3 = FakeHTTP()
    tue = _run("tuesday", conn, http3, now=KICK + timedelta(days=2))
    assert http3.calls == ["/account", "/historical-odds"] and tue["close_lines"] == 1
    row = conn.execute("SELECT p.player_name, l.bookmaker, l.line, l.over_price, l.under_price FROM nfl_prop_lines l "
                       "JOIN nfl_oddspapi_players p USING (fixture_id, op_player_id) WHERE kind = 'close'").fetchone()
    assert row == ("Ja'Marr Chase", "draftkings", 45.5, 1.92, 1.90)


def test_open_prefers_bookmaker_main_line_flag():
    markets = {int(k): v for k, v in c.filter_markets(MARKETS).items()}
    odds = json.loads(json.dumps(ODDS))
    odds["bookmakerOdds"]["fanduel"]["markets"]["100"]["outcomes"]["100"]["players"]["77"]["mainLine"] = True
    rows = c.open_lines(odds, markets)
    assert [(r["line"], r["over_price"]) for r in rows] == [(45.5, 1.70)]
