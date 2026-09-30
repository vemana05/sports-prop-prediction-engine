"""Resolve `nba_injuries.player_name_raw` ("Last, First" from the PDF) to a
canonical `player_id_nba`, reusing the same normalize/fuzzy-match machinery
as the Sleeper crosswalk (`build_xref.py`). Without this, injury status
can't be joined onto a player's game row — the availability table
(`backend/ingest/nba_availability.py`) and any Phase 2 injury feature
depend on it.

Matches against `player_xref` (one row per nba.com player id, already
seeded by `build_xref.run()` — run that first) rather than re-deriving the
nba.com player list, so a name that was already manually overridden for
Sleeper also resolves correctly here.
"""

from __future__ import annotations

import duckdb
from rapidfuzz import fuzz, process

from backend.db import get_connection
from backend.ingest.util import IngestResult, logger, utcnow
from backend.xref.build_xref import FUZZY_MATCH_THRESHOLD, normalize_name


def _xref_index(conn: duckdb.DuckDBPyConnection) -> tuple[dict[str, str], list[str]]:
    """Returns ({normalized_name: player_id}, [normalized_name, ...])."""
    rows = conn.execute("SELECT player_id, full_name FROM player_xref WHERE full_name IS NOT NULL").fetchall()
    index: dict[str, str] = {}
    for player_id, full_name in rows:
        norm = normalize_name(full_name)
        if norm:
            index[norm] = player_id
    return index, list(index.keys())


def build(conn: duckdb.DuckDBPyConnection) -> IngestResult:
    now = utcnow()
    xref_index, xref_names = _xref_index(conn)

    raw_names = [r[0] for r in conn.execute("SELECT DISTINCT player_name_raw FROM nba_injuries").fetchall()]

    matched = 0
    review_rows = []
    update_rows = []

    for raw_name in raw_names:
        norm = normalize_name(raw_name)
        player_id = None
        if norm in xref_index:
            player_id = xref_index[norm]
        elif norm:
            result = process.extractOne(norm, xref_names, scorer=fuzz.WRatio)
            if result and result[1] >= FUZZY_MATCH_THRESHOLD:
                player_id = xref_index[result[0]]
            else:
                review_rows.append(
                    # raw_id is part of xref_review_queue's primary key, so it can't be NULL —
                    # injury names have no separate id the way Sleeper/nba.com ids do, hence "".
                    ("nba_injuries", raw_name, "", "no player_xref match above threshold",
                     xref_index[result[0]] if result else None, float(result[1]) if result else None, now)
                )
                continue

        update_rows.append((player_id, raw_name))
        matched += 1

    if update_rows:
        conn.executemany(
            "UPDATE nba_injuries SET player_id_nba = ? WHERE player_name_raw = ?",
            update_rows,
        )
    if review_rows:
        conn.executemany(
            """
            INSERT INTO xref_review_queue (source, raw_name, raw_id, reason, best_candidate, best_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source, raw_name, raw_id) DO UPDATE SET
                reason = excluded.reason, best_candidate = excluded.best_candidate,
                best_score = excluded.best_score, created_at = excluded.created_at
            """,
            review_rows,
        )

    logger.info(
        "injury name match: %d/%d distinct names resolved, %d unmatched sent to review queue",
        matched, len(raw_names), len(review_rows),
    )
    return IngestResult(rows_written=matched, notes={"distinct_names": len(raw_names), "unmatched": len(review_rows)})


def run(conn: duckdb.DuckDBPyConnection | None = None) -> IngestResult:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        return build(conn)
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    result = run()
    print(f"injury name match complete: {result.rows_written}/{result.notes['distinct_names']} names resolved, {result.notes['unmatched']} in review queue")
