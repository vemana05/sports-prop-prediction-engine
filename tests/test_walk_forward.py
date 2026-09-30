"""Section 5.4: "Walk-forward backtest by season; never random splits."
Pure unit test of the split logic — no DB needed — guarding against a
regression that leaks a test season's games into its own training set or
uses a season out of chronological order.
"""

import pandas as pd

from backend.models.nba_minutes_model import SEASON_ORDER, walk_forward_splits


def _fake_frame() -> pd.DataFrame:
    rows = []
    for season in SEASON_ORDER:
        for i in range(10):
            rows.append({"season": season, "row": f"{season}-{i}"})
    return pd.DataFrame(rows)


def test_splits_cover_every_season_after_the_first_exactly_once():
    df = _fake_frame()
    test_seasons = [test_season for _, _, test_season in walk_forward_splits(df)]
    assert test_seasons == SEASON_ORDER[1:]


def test_train_never_contains_the_test_season_or_a_later_one():
    df = _fake_frame()
    for train_df, test_df, test_season in walk_forward_splits(df):
        train_seasons = set(train_df["season"])
        assert test_season not in train_seasons
        test_idx = SEASON_ORDER.index(test_season)
        for s in train_seasons:
            assert SEASON_ORDER.index(s) < test_idx, f"train season {s} is not strictly before test season {test_season}"


def test_train_and_test_rows_are_disjoint():
    df = _fake_frame()
    for train_df, test_df, _ in walk_forward_splits(df):
        assert set(train_df["row"]).isdisjoint(set(test_df["row"]))
