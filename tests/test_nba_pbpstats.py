"""Regression test for the pbpstats `data_provider="live"` workaround.

Background (full story in backend/ingest/nba_pbpstats.py's module docstring):
while building Phase 1, live-testing showed pbpstats 1.3.11's own two other
data providers are dead — `stats_nba` (playbyplayv2) returns `{}` for every
game now, and `data_nba` (data.nba.com) 403s, that domain having been
decommissioned. The `live` provider (NBA's live-data S3 feed) is the only
one that actually returns data, and — despite the name — serves full
historical games, not just in-progress ones.

But `live` has its own real bug: pbpstats' own aggregation helpers,
`Possessions.player_stats` and `.lineup_stats`, crash on rebound events,
because the live-data `FieldGoal` subclass doesn't fully implement the
`shot_type` property chain those helpers rely on. We worked around this by
never calling those two properties — `_possession_rows` reads only
possession-level fields (`offense_team_id`, `period`, `start_time`,
`end_time`, `start_score_margin`) plus lineup ids straight off the first
event, all verified safe against real games.

There's a second, smaller real edge case in live data itself (not a pbpstats
bug): `get_team_ids()` raises `AttributeError` for roughly 1% of possessions
in every game we ingested, because it walks into a `LiveStartOfPeriod` event
that has no `team_id`. `_possession_rows` must keep the possession (with
defense/lineup fields null) rather than drop it or crash the whole game.

These two tests guard both: (1) nobody reintroduces a call to the broken
aggregation properties, and (2) the team-id-resolution edge case degrades
gracefully instead of losing data or crashing.
"""

from backend.ingest.nba_pbpstats import _possession_rows


class _FakeEvent:
    def __init__(self, lineup_ids=None):
        self.lineup_ids = lineup_ids


class _FakePossession:
    """Stands in for pbpstats.resources.possessions.possession.Possession,
    exposing only what _possession_rows actually touches."""

    def __init__(self, period, offense_team_id, events, team_ids_ok=True, margin=0):
        self.period = period
        self.offense_team_id = offense_team_id
        self.events = events
        self.start_time = "12:00"
        self.end_time = "11:45"
        self.start_score_margin = margin
        self._team_ids_ok = team_ids_ok

    def get_team_ids(self):
        if not self._team_ids_ok:
            # the real failure mode: a LiveStartOfPeriod event has no team_id
            raise AttributeError("'LiveStartOfPeriod' object has no attribute 'team_id'")
        return [self.offense_team_id, 999]


def test_possession_rows_never_touches_the_broken_aggregation_properties():
    class ExplodingPossession(_FakePossession):
        @property
        def player_stats(self):  # pragma: no cover - must never be called
            raise AssertionError("must not call Possessions.player_stats — see module docstring")

        @property
        def lineup_stats(self):  # pragma: no cover - must never be called
            raise AssertionError("must not call Possessions.lineup_stats — see module docstring")

    events = [_FakeEvent(lineup_ids={1: "a-b-c-d-e", 999: "f-g-h-i-j"})]
    rows = _possession_rows("0022500099", "2025-26", [ExplodingPossession(1, 1, events)])

    assert len(rows) == 1
    row = rows[0]
    assert row[4] == "1"  # offense_team_id
    assert row[5] == "999"  # defense_team_id
    assert row[9] == "a-b-c-d-e"  # lineup_offense
    assert row[10] == "f-g-h-i-j"  # lineup_defense


def test_possession_with_unresolvable_team_ids_is_kept_not_dropped():
    events = [_FakeEvent(lineup_ids=None)]
    possessions = [_FakePossession(1, 1, events, team_ids_ok=False, margin=3)]

    rows = _possession_rows("0022500099", "2025-26", possessions)

    assert len(rows) == 1, "a possession where get_team_ids() fails must still produce a row"
    game_id, season, num, period, offense, defense, start, end, margin, lineup_off, lineup_def, game_date, ingested = rows[0]
    assert offense == "1"
    assert defense is None
    assert lineup_off is None and lineup_def is None
    assert margin == 3  # fields that don't depend on get_team_ids() are still captured


def test_one_bad_possession_does_not_drop_the_rest_of_the_game():
    good_events = [_FakeEvent(lineup_ids={1: "a-b-c-d-e", 999: "f-g-h-i-j"})]
    possessions = [
        _FakePossession(1, 1, good_events),
        _FakePossession(1, 1, good_events, team_ids_ok=False),
        _FakePossession(2, 999, good_events),
    ]

    rows = _possession_rows("0022500099", "2025-26", possessions)

    assert len(rows) == 3
