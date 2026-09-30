"""Freshness checks (Section 4.3 / 10.4-F): is each registered source up to
date with its own declared refresh cadence?

`compute_freshness` returns a status per source so the Data Sources page
(Phase 5) can render a dot per row. "stale" is a normal, expected state for
game-day-only sources during the offseason — this module reports state, it
doesn't assert the business should currently be green.
"""

from __future__ import annotations

from datetime import timedelta

import duckdb

from backend.ingest.runner import load_sources
from backend.ingest.util import utcnow

# Grace period before a source's last success counts as "stale", keyed by a
# rough read of its `refresh` string in sources.yaml.
_CADENCE_GRACE_HOURS = {
    "hourly": 3,
    "daily": 36,
}
_DEFAULT_GRACE_HOURS = 48  # e.g. "every 15m on game days" — lenient outside game days


def _grace_hours(refresh: str) -> int:
    refresh = (refresh or "").lower()
    for key, hours in _CADENCE_GRACE_HOURS.items():
        if key in refresh:
            return hours
    return _DEFAULT_GRACE_HOURS


def compute_freshness(conn: duckdb.DuckDBPyConnection, sources_path: str = "config/sources.yaml") -> dict[str, dict]:
    sources = load_sources(sources_path)
    now = utcnow()
    result: dict[str, dict] = {}

    for entry in sources:
        name = entry["name"]
        last_success = conn.execute(
            """
            SELECT MAX(finished_at) FROM ingestion_log
            WHERE source = ? AND status IN ('success', 'fallback')
            """,
            [name],
        ).fetchone()[0]

        grace = _grace_hours(entry.get("refresh", ""))
        if last_success is None:
            status = "never_run"
        else:
            last_success_utc = last_success if last_success.tzinfo else last_success.replace(tzinfo=now.tzinfo)
            status = "fresh" if now - last_success_utc <= timedelta(hours=grace) else "stale"

        result[name] = {
            "status": status,
            "last_success": last_success.isoformat() if last_success else None,
            "expected_within_hours": grace,
        }
    return result
