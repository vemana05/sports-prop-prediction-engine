"""Ingestion runner (Section 4.1).

Reads `config/sources.yaml`, runs each registered source's `run(conn)`
entrypoint, and logs every attempt to the `ingestion_log` table. If a
`priority: primary` source fails (or is never registered) for a given
`provides` data type, any `priority: secondary` source that also provides
that data type is run as a fallback and the substitution is logged.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import duckdb
import yaml

from backend.db import get_connection, next_log_id
from backend.ingest.util import logger, utcnow

DEFAULT_SOURCES_PATH = "config/sources.yaml"


def load_sources(path: str = DEFAULT_SOURCES_PATH) -> list[dict[str, Any]]:
    with open(path) as f:
        return yaml.safe_load(f) or []


def _provides(entry: dict[str, Any]) -> set[str]:
    p = entry["provides"]
    return set(p) if isinstance(p, list) else {p}


def _run_source(conn: duckdb.DuckDBPyConnection, entry: dict[str, Any], fallback_from: str | None = None) -> dict[str, Any]:
    name = entry["name"]
    started = utcnow()
    status, rows, error = "success", 0, None
    try:
        module = importlib.import_module(entry["module"])
        result = module.run(conn)
        rows = result.rows_written
    except Exception as exc:  # noqa: BLE001 - any source failure must not kill the run
        status = "failed" if fallback_from is None else "fallback_failed"
        error = f"{type(exc).__name__}: {exc}"
        logger.error("ingest runner: %s failed: %s", name, error)
    finished = utcnow()

    conn.execute(
        "INSERT INTO ingestion_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [next_log_id(conn), name, "fallback" if (fallback_from and status == "success") else status,
         rows, started, finished, error, fallback_from],
    )
    return {"status": status, "rows": rows, "error": error}


def run_all(sources_path: str = DEFAULT_SOURCES_PATH) -> dict[str, dict[str, Any]]:
    sources = load_sources(sources_path)
    primaries = [s for s in sources if s.get("priority") == "primary"]
    secondaries = [s for s in sources if s.get("priority") != "primary"]

    conn = get_connection()
    results: dict[str, dict[str, Any]] = {}
    covered_provides: set[str] = set()

    try:
        for entry in primaries:
            results[entry["name"]] = _run_source(conn, entry)
            if results[entry["name"]]["status"] == "success":
                covered_provides |= _provides(entry)

        for entry in secondaries:
            needed = _provides(entry) - covered_provides
            if not needed:
                logger.info("ingest runner: skipping %s, already covered by a primary", entry["name"])
                continue
            # find which primary (if any) this is standing in for, for the log
            fallback_from = next(
                (p["name"] for p in primaries if _provides(p) & needed), None
            )
            results[entry["name"]] = _run_source(conn, entry, fallback_from=fallback_from)
            if results[entry["name"]]["status"] in ("success", "fallback"):
                covered_provides |= _provides(entry)
    finally:
        conn.close()

    return results


if __name__ == "__main__":
    results = run_all()
    print("Ingestion run summary:")
    for name, r in results.items():
        print(f"  {name}: {r['status']} ({r['rows']} rows)" + (f" — {r['error']}" if r["error"] else ""))
