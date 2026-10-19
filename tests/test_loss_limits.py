"""Loss limits (spec 10.4 G): ET windows, real vs paper vs bonus, the
24-hour delay on raising, pauses, and API key status never leaking a value."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from backend.api import limits, main, tracker
from backend.bankroll import kelly

UTC = timezone.utc
# Wed Oct 21, 2026, 10:00 ET (EDT, UTC-4).
NOW = datetime(2026, 10, 21, 14, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(tracker, "DB_PATH", tmp_path / "bets.sqlite")
    yield
    kelly.set_kelly_fraction(None)


def _bet(result, settled_at, stake=10.0, odds=3.0, paper=False, bonus=False):
    return {"stake": stake, "decimal_odds": odds, "paper": paper, "bonus": bonus, "result": result,
            "settled_at": settled_at.isoformat() if settled_at else None}


def test_daily_window_is_the_et_calendar_day():
    bets = [
        _bet("lost", datetime(2026, 10, 21, 4, 30, tzinfo=UTC)),  # 00:30 ET today: counts
        _bet("lost", datetime(2026, 10, 21, 3, 30, tzinfo=UTC)),  # 23:30 ET yesterday: doesn't
        _bet("lost", datetime(2026, 10, 22, 3, 59, tzinfo=UTC)),  # 23:59 ET today: counts
    ]
    t = limits.window_totals(bets, "daily", NOW)
    assert t["lost"] == 20.0 and t["net_loss"] == 20.0
    assert t["start"] == "2026-10-21T04:00:00+00:00" and t["end"] == "2026-10-22T04:00:00+00:00"


def test_weekly_window_starts_monday_midnight_et():
    bets = [
        _bet("lost", datetime(2026, 10, 19, 4, 0, tzinfo=UTC)),   # Mon 00:00 ET: counts
        _bet("lost", datetime(2026, 10, 19, 3, 59, tzinfo=UTC)),  # Sun 23:59 ET: last week
    ]
    t = limits.window_totals(bets, "weekly", NOW)
    assert t["lost"] == 10.0 and t["start"] == "2026-10-19T04:00:00+00:00"
    # Across the Nov 1 DST change the week still runs Monday 00:00 ET to Monday 00:00 ET.
    start, end = limits.window("weekly", datetime(2026, 11, 3, 15, 0, tzinfo=UTC))
    assert start.isoformat() == "2026-11-02T00:00:00-05:00" and end - start == timedelta(days=7)


def test_only_real_non_bonus_settled_bets_count_and_winnings_offset():
    today = datetime(2026, 10, 21, 13, 0, tzinfo=UTC)
    bets = [
        _bet("lost", today, stake=50),                 # real loss: 50
        _bet("won", today, stake=10, odds=3.0),        # real win: +20
        _bet("lost", today, stake=100, paper=True),    # paper: ignored
        _bet("lost", today, stake=25, bonus=True),     # bonus loss: ignored
        _bet("won", today, stake=5, odds=5.0, bonus=True),  # bonus win: reported separately
        _bet("push", today, stake=40),                 # push: nothing
        _bet("pending", None, stake=60),               # unsettled: nothing
    ]
    t = limits.window_totals(bets, "daily", NOW)
    assert (t["lost"], t["won"], t["net_loss"]) == (50.0, 20.0, 30.0)
    assert t["bonus_bets"] == 2 and t["bonus_winnings"] == 20.0
    # Net winnings never go below $0 of loss.
    assert limits.window_totals([_bet("won", today)], "daily", NOW)["net_loss"] == 0.0


def test_status_levels_warn_at_80_percent_and_hit_at_100():
    today = datetime(2026, 10, 21, 13, 0, tzinfo=UTC)
    limits.set_limit("daily", 100, NOW)
    limits.set_limit("weekly", 500, NOW)
    st = limits.status(NOW, bets=[_bet("lost", today, stake=80)])
    by = {r["period"]: r for r in st["limits"]}
    assert by["daily"]["level"] == "warn" and by["weekly"]["level"] == "ok" and by["monthly"]["level"] == "none"
    st = limits.status(NOW, bets=[_bet("lost", today, stake=100)])
    assert st["level"] == "hit" and st["hit"] == ["daily"]


def test_lowering_is_immediate_raising_and_removing_wait_24_hours():
    limits.set_limit("daily", 100, NOW)
    assert limits.get_limits(NOW)["daily"] == {"amount": 100, "pending": None}

    limits.set_limit("daily", 200, NOW)  # raise: queued
    lim = limits.get_limits(NOW + timedelta(hours=23, minutes=59))["daily"]
    assert lim["amount"] == 100 and lim["pending"] == {"amount": 200, "effective_at": "2026-10-22T14:00:00+00:00"}
    assert limits.get_limits(NOW + timedelta(hours=24))["daily"] == {"amount": 200, "pending": None}

    limits.set_limit("daily", None, NOW + timedelta(hours=25))  # remove: queued too
    assert limits.get_limits(NOW + timedelta(hours=30))["daily"]["amount"] == 200
    assert limits.get_limits(NOW + timedelta(hours=49))["daily"]["amount"] is None


def test_lowering_cancels_a_pending_raise_and_cancel_keeps_the_old_limit():
    limits.set_limit("weekly", 300, NOW)
    limits.set_limit("weekly", 600, NOW)
    limits.set_limit("weekly", 250, NOW + timedelta(hours=1))
    assert limits.get_limits(NOW + timedelta(days=2))["weekly"] == {"amount": 250, "pending": None}

    limits.set_limit("weekly", 400, NOW)
    limits.cancel_pending("weekly")
    assert limits.get_limits(NOW + timedelta(days=2))["weekly"] == {"amount": 250, "pending": None}


def test_pause_cant_be_shortened_and_blocks_only_real_bets(monkeypatch):
    until = limits.pause(7, NOW)
    assert limits.pause(1, NOW) == until  # a shorter break doesn't replace a longer one
    assert limits.paused_until(NOW + timedelta(days=6)) == until
    assert limits.paused_until(NOW + timedelta(days=7, seconds=1)) is None

    monkeypatch.setattr(limits, "_utcnow", lambda: NOW)
    bet = dict(league="nba", tier="risky", name="P", legs=[], decimal_odds=3.0, american_odds=200, model_p=0.4, stake=5)
    with pytest.raises(HTTPException) as e:
        main.log_bet(main.BetIn(**bet))
    assert e.value.status_code == 423
    assert main.log_bet(main.BetIn(**bet, paper=True))["paper"] is True


def test_hit_limit_needs_confirmation_naming_the_limit(monkeypatch):
    monkeypatch.setattr(limits, "_utcnow", lambda: NOW)
    limits.set_limit("daily", 20, NOW)
    bet = dict(league="nba", tier="risky", name="P", legs=[], decimal_odds=3.0, american_odds=200, model_p=0.4, stake=20)
    monkeypatch.setattr(tracker, "_now", lambda: NOW.isoformat())
    tracker.settle_bet(tracker.add_bet({**bet, "paper": False})["id"], "lost")

    with pytest.raises(HTTPException) as e:
        main.log_bet(main.BetIn(**bet))
    assert e.value.status_code == 409 and "daily loss limit" in e.value.detail
    assert main.log_bet(main.BetIn(**bet, confirm_over_limit=True))["paper"] is False
    assert main.log_bet(main.BetIn(**bet, paper=True))["paper"] is True


def test_bonus_bet_loss_costs_nothing_in_tracker_profit():
    assert tracker.profit({"result": "lost", "stake": 25, "decimal_odds": 3.0, "bonus": True}) == 0.0
    assert tracker.profit({"result": "lost", "stake": 25, "decimal_odds": 3.0, "bonus": False}) == -25


def test_kelly_fraction_is_saved_and_used_for_sizing():
    limits.set_kelly(0.5)
    assert limits.get_kelly() == 0.5 and kelly.kelly_fraction() == 0.5
    with pytest.raises(ValueError):
        limits.set_kelly(1.5)


def test_api_never_returns_a_key_value(monkeypatch):
    secrets = {n: f"sk-SECRET-{i}-do-not-leak" for i, (n, _) in enumerate(main.API_KEYS)}
    for n, v in secrets.items():
        monkeypatch.setenv(n, v)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    body = json.dumps([main.settings(), main.loss_limits()])
    for n, v in secrets.items():
        assert v not in body and "SECRET" not in body
    status = {k["name"]: k["status"] for k in main.settings()["api_keys"]}
    assert status["ANTHROPIC_API_KEY"] == "missing" and status["ODDS_API_KEY"] == "set"
    assert set(status.values()) <= {"set", "missing"}
