"""NFL model cache: saved thru-last-season models are reused, not retrained,
and concurrent callers share one training run. No warehouse needed."""

import os
import threading
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from backend.models.nfl_market_model import MARKETS, Spread
from backend.predict import nfl_projections as np_

MARKET = MARKETS[0]


def _tiny_booster() -> lgb.Booster:
    x = np.arange(40, dtype=float).reshape(-1, 1)
    return lgb.train({"objective": "regression", "verbose": -1, "min_data_in_leaf": 1}, lgb.Dataset(x, label=x[:, 0]), num_boost_round=2)


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """Isolated model dir + empty memo; training stubbed and counted."""
    monkeypatch.setattr(np_, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(np_, "_MODELS", {})
    monkeypatch.setattr(np_, "_LOCKS", {})
    calls = []

    def fake_train(df, market):
        calls.append(market.name)
        time.sleep(0.2)  # long enough for concurrent callers to pile up
        return _tiny_booster()

    monkeypatch.setattr(np_, "_training_frame", lambda m, t: pd.DataFrame({"position": ["WR"], m.target: [1.0]}))
    monkeypatch.setattr(np_, "_oof_predictions", lambda df, m: np.ones(len(df)))
    monkeypatch.setattr(np_, "fit_spread", lambda y, mu, fam: Spread(a=1.0, b=0.1, z_sorted=np.array([-1.0, 0.0, 1.0])))
    monkeypatch.setattr(np_, "_train", fake_train)
    return calls


def test_second_call_loads_from_disk_without_training(stub, monkeypatch):
    model, spread = np_._get_or_train(MARKET, 2025)
    assert stub == [MARKET.name]
    assert (np_.MODEL_DIR / f"nfl_{MARKET.name}_thru2025.txt").exists()

    # Fresh process: nothing in memory, and the files are older than the old 7-day rule.
    monkeypatch.setattr(np_, "_MODELS", {})
    old = time.time() - 30 * 86400
    for p in np_._paths(MARKET, 2025):
        os.utime(p, (old, old))
    monkeypatch.setattr(np_, "_train", lambda *a: pytest.fail("retrained a model that was saved on disk"))

    model2, spread2 = np_._get_or_train(MARKET, 2025)
    assert stub == [MARKET.name]
    assert spread2.a == spread.a and model2.num_trees() == model.num_trees()


def test_concurrent_callers_share_one_training_run(stub):
    threads = [threading.Thread(target=np_._get_or_train, args=(MARKET, 2025)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert stub == [MARKET.name]


def test_never_trains_on_the_season_in_progress(stub):
    with pytest.raises(ValueError):
        np_._get_or_train(MARKET, np_.current_season())
    assert stub == []


def test_ensure_models_is_ready_once_models_are_saved(stub):
    assert np_.models_ready(2025) is False
    np_.warm_models(2025)
    assert np_.models_ready(2025) and np_.ensure_models(2026) is None
