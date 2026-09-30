"""Player ID crosswalk (Section 4.2): nba.com id <-> Sleeper id.

Seeds `player_xref` from nba_api's static player list (local, no network)
matched against the `sleeper_players` table (must be ingested first — see
backend.ingest.sleeper). Exact normalized-name matches first, then fuzzy
matching via rapidfuzz for anything left. `manual_overrides.csv` lets an
operator force a specific nba_player_id for a source name that matches
wrong or not at all. Anything still unmatched goes to `xref_review_queue`
rather than being silently dropped.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import duckdb
from nba_api.stats.static import players as nba_static_players
from rapidfuzz import fuzz, process

from backend.db import get_connection
from backend.ingest.util import IngestResult, logger, utcnow

DEFAULT_OVERRIDES_PATH = "config/manual_overrides.csv"
FUZZY_MATCH_THRESHOLD = 90.0

_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.strip()
    if "," in name:
        # "Last, First" -> "First Last" (nbainjuries' report format)
        last, _, first = name.partition(",")
        name = f"{first.strip()} {last.strip()}"
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = _SUFFIX_RE.sub("", name)
    name = _PUNCT_RE.sub(" ", name)
    name = _WS_RE.sub(" ", name).strip().lower()
    return name


def load_manual_overrides(path: str = DEFAULT_OVERRIDES_PATH) -> dict[str, str]:
    """Returns {normalized_source_name: nba_player_id}."""
    overrides: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return overrides
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("raw_name") and row.get("nba_player_id"):
                overrides[normalize_name(row["raw_name"])] = row["nba_player_id"].strip()
    return overrides


def _nba_player_index() -> tuple[dict[str, dict], list[str]]:
    """Returns ({normalized_name: player_dict}, [normalized_name, ...])."""
    index: dict[str, dict] = {}
    for p in nba_static_players.get_players():
        norm = normalize_name(p["full_name"])
        if norm:
            index[norm] = p
    return index, list(index.keys())


def build(conn: duckdb.DuckDBPyConnection, overrides_path: str = DEFAULT_OVERRIDES_PATH) -> IngestResult:
    now = utcnow()
    nba_index, nba_names = _nba_player_index()
    overrides = load_manual_overrides(overrides_path)

    sleeper_rows = conn.execute(
        "SELECT sleeper_id, full_name FROM sleeper_players WHERE full_name IS NOT NULL"
    ).fetchall()

    xref_rows = []
    review_rows = []
    matched_nba_ids: set[str] = set()

    for sleeper_id, full_name in sleeper_rows:
        norm = normalize_name(full_name)
        nba_player = None
        match_method = None
        match_score = None

        if norm in overrides:
            nba_id = overrides[norm]
            nba_player = next((p for p in nba_index.values() if str(p["id"]) == nba_id), {"id": nba_id, "full_name": full_name})
            match_method, match_score = "manual_override", 100.0
        elif norm in nba_index:
            nba_player = nba_index[norm]
            match_method, match_score = "exact", 100.0
        elif norm:
            result = process.extractOne(norm, nba_names, scorer=fuzz.WRatio)
            if result and result[1] >= FUZZY_MATCH_THRESHOLD:
                nba_player = nba_index[result[0]]
                match_method, match_score = "fuzzy", float(result[1])
            else:
                review_rows.append(
                    ("sleeper", full_name, sleeper_id, "no nba_api match above threshold",
                     nba_index[result[0]]["full_name"] if result else None, float(result[1]) if result else None, now)
                )
                continue

        nba_id_str = str(nba_player["id"])
        matched_nba_ids.add(nba_id_str)
        xref_rows.append((nba_id_str, nba_id_str, sleeper_id, full_name, match_method, match_score, now, now))

    # Every nba.com player gets an xref row even with no Sleeper match, so
    # downstream joins against nba_player_games never fail to find a player_id.
    for norm, nba_player in nba_index.items():
        nba_id_str = str(nba_player["id"])
        if nba_id_str not in matched_nba_ids:
            xref_rows.append((nba_id_str, nba_id_str, None, nba_player["full_name"], "nba_only", None, now, now))

    if xref_rows:
        conn.executemany(
            """
            INSERT INTO player_xref
                (player_id, nba_player_id, sleeper_id, full_name, match_method, match_score, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (player_id) DO UPDATE SET
                sleeper_id = excluded.sleeper_id, match_method = excluded.match_method,
                match_score = excluded.match_score, updated_at = excluded.updated_at
            """,
            xref_rows,
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
        "xref: %d players crosswalked, %d unmatched sent to review queue", len(xref_rows), len(review_rows)
    )
    return IngestResult(rows_written=len(xref_rows), notes={"unmatched": len(review_rows)})


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
    print(f"xref build complete: {result.rows_written} matched, {result.notes['unmatched']} in review queue")
