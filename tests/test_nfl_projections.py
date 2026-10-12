"""NFL model versions: saved models are reused, not retrained; concurrent
callers share one training run; week N is only ever projected by a model
trained before week N; training happens off the request thread. No
warehouse needed."""

import os
import threading
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from backend.models.nfl_market_model import MARKETS, Spread
from backend.predict import nfl_projections as np_
from backend.predict.nfl_projections import Through

MARKET = MARKETS[0]
BASE = Through(2025)


def _tiny_booster() -> lgb.Booster:
    x = np.arange(40, dtype=float).reshape(-1, 1)
    return lgb.train({"objective": "regression", "verbose": -1, "min_data_in_leaf": 1}, lgb.Dataset(x, label=x[:, 0]), num_boost_round=2)


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """Isolated model dir + empty memo; training stubbed and recorded as
    (market, version tag, thread name)."""
    monkeypatch.setenv("NFL_WEEKLY_RETRAIN", "1")
    monkeypatch.setattr(np_, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(np_, "_MODELS", {})
    monkeypatch.setattr(np_, "_LOCKS", {})
    monkeypatch.setattr(np_, "_warm_thread", None)
    monkeypatch.setattr(np_, "_warm_error", None)
    monkeypatch.setattr(np_, "complete", lambda t: t.week is not None or t.season < 2026)
    calls, frames = [], []

    def fake_frame(m, t):
        frames.append(np_._where(t))
        return pd.DataFrame({"position": ["WR"], m.target: [1.0], "tag": [t.tag]})

    def fake_train(df, market):
        calls.append((market.name, df["tag"].iloc[0], threading.current_thread().name))
        time.sleep(0.2)  # long enough for concurrent callers to pile up
        return _tiny_booster()

    monkeypatch.setattr(np_, "_training_frame", fake_frame)
    monkeypatch.setattr(np_, "_oof_predictions", lambda df, m: np.ones(len(df)))
    monkeypatch.setattr(np_, "fit_spread", lambda y, mu, fam: Spread(a=1.0, b=0.1, z_sorted=np.array([-1.0, 0.0, 1.0])))
    monkeypatch.setattr(np_, "_train", fake_train)
    stub_ns = type("Stub", (), {"calls": calls, "frames": frames})
    return stub_ns


def test_second_call_loads_from_disk_without_training(stub, monkeypatch):
    model, spread = np_._get_or_train(MARKET, BASE)
    assert len(stub.calls) == 1
    assert (np_.MODEL_DIR / f"nfl_{MARKET.name}_thru2025.txt").exists()

    # Fresh process: nothing in memory, and the files are a month old.
    monkeypatch.setattr(np_, "_MODELS", {})
    old = time.time() - 30 * 86400
    for p in np_._paths(MARKET, BASE):
        os.utime(p, (old, old))
    monkeypatch.setattr(np_, "_train", lambda *a: pytest.fail("retrained a model that was saved on disk"))

    model2, spread2 = np_._get_or_train(MARKET, BASE)
    assert len(stub.calls) == 1
    assert spread2.a == spread.a and model2.num_trees() == model.num_trees()


def test_concurrent_callers_share_one_training_run(stub):
    threads = [threading.Thread(target=np_._get_or_train, args=(MARKET, BASE)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(stub.calls) == 1


def test_never_trains_on_games_that_arent_final(stub, monkeypatch):
    monkeypatch.setattr(np_, "complete", lambda t: False)
    with pytest.raises(ValueError):
        np_._get_or_train(MARKET, Through(2026, 5))
    with pytest.raises(ValueError):
        np_._get_or_train(MARKET, Through(2026))
    assert stub.calls == []


def test_weekly_training_frame_stops_at_its_week():
    assert np_._where(Through(2026, 4)) == ("(season < ? OR (season = ? AND week <= ?))", [2026, 2026, 4])
    assert np_._where(BASE) == ("season <= ?", [2025])


def test_weekly_versions_are_saved_separately_and_kept(stub):
    for w in (1, 2, 3):
        np_.warm_models(Through(2026, w))
    for w in (1, 2, 3):
        assert np_.models_ready(Through(2026, w))
    assert (np_.MODEL_DIR / f"nfl_{MARKET.name}_thru2026w03.txt").exists()


def test_week_n_uses_the_model_from_before_week_n(stub):
    # Nothing weekly yet: every week falls back to the completed-seasons model.
    assert np_.version_for(2026, 1) == BASE
    assert np_.version_for(2026, 5) == BASE
    for w in (1, 2, 3):
        np_.warm_models(Through(2026, w))
    assert np_.version_for(2026, 1) == BASE            # week 1: nothing from 2026 before it
    assert np_.version_for(2026, 3) == Through(2026, 2)
    assert np_.version_for(2026, 4) == Through(2026, 3)
    assert np_.version_for(2026, 6) == Through(2026, 3)  # thru w05 missing: newest earlier one
    for week in range(1, 10):
        v = np_.version_for(2026, week)
        assert v.week is None or v.week < week        # never a model that saw week N


def test_ensure_models_trains_in_the_background_and_reports_warming(stub, monkeypatch):
    monkeypatch.setattr(np_, "current_season", lambda: 2026)
    np_.warm_models(BASE)
    stub.calls.clear()

    warming = np_.ensure_models(2026, week=4)
    assert warming and warming["status"] == "warming"
    np_._warm_thread.join(timeout=10)
    assert {c[1] for c in stub.calls} == {"thru2026w03"}
    assert all(c[2] == "nfl-model-warm" for c in stub.calls)  # never on the request thread
    assert np_.ensure_models(2026, week=4) is None
    assert np_.version_for(2026, 4) == Through(2026, 3)


def test_ensure_models_serves_the_fallback_while_last_week_isnt_final(stub, monkeypatch):
    monkeypatch.setattr(np_, "current_season", lambda: 2026)
    monkeypatch.setattr(np_, "complete", lambda t: t.week is None)
    np_.warm_models(BASE)
    assert np_.ensure_models(2026, week=6) is None
    assert np_.version_for(2026, 6) == BASE


def test_weekly_retrain_off_serves_and_trains_only_the_base(stub, monkeypatch):
    monkeypatch.setattr(np_, "current_season", lambda: 2026)
    np_.warm_models(BASE)
    np_.warm_models(Through(2026, 3))
    monkeypatch.delenv("NFL_WEEKLY_RETRAIN")
    stub.calls.clear()
    assert np_.version_for(2026, 4) == BASE  # saved weekly versions are ignored
    assert np_.ensure_models(2026, week=6) is None
    assert stub.calls == []
