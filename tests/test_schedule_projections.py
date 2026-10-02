"""Unit tests for the schedule page's rest flags and projection ranges — no DB needed."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend.predict.schedule_projections import _nb_range, _normal_range, _rest


def test_rest_back_to_back():
    r = _rest(date(2026, 11, 5), pd.Timestamp("2026-11-04"), pd.Timestamp("2026-11-04"))
    assert r == {"rest_days": 1, "back_to_back": True, "season_opener": False}


def test_rest_season_opener_when_no_earlier_game_this_season():
    r = _rest(date(2026, 10, 20), pd.Timestamp("2026-04-12"), pd.NaT)
    assert r["season_opener"] is True and r["back_to_back"] is False and r["rest_days"] == 191


def test_normal_range_is_the_middle_fifty_percent_and_never_negative():
    lo, hi = _normal_range(np.array([20.0, 1.0]), 5.0)
    assert lo[0] == pytest.approx(20 - 0.6745 * 5, abs=1e-3) and hi[0] == pytest.approx(20 + 0.6745 * 5, abs=1e-3)
    assert lo[1] == 0.0


def test_nb_range_brackets_the_mean():
    lo, hi = _nb_range(np.array([8.0]), 15.0)
    assert lo[0] <= 8.0 <= hi[0] and lo[0] < hi[0]
