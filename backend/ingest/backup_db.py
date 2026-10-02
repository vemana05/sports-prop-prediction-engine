"""Timestamped backups of warehouse.duckdb.

Meant to run (via Docker — see the rebuild commands in the repo) after each
ingestion rebuild stage, so a bad or interrupted stage can be rolled back by
restoring the previous backup instead of re-ingesting from scratch. Keeps
only the most recent `--keep` (default 3) backups in data/backups/; older
ones are deleted automatically. Backups are plain file copies, so run this
only after the ingest container that wrote warehouse.duckdb has exited
cleanly (checkpointed its WAL), not concurrently with one still writing.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from backend.db import DEFAULT_DB_PATH
from backend.ingest.util import logger

DEFAULT_KEEP_LAST = 3


def backup_warehouse(db_path: str | None = None, backup_dir: str | None = None, keep_last: int = DEFAULT_KEEP_LAST) -> Path | None:
    db_path = Path(db_path or os.environ.get("DUCKDB_PATH", DEFAULT_DB_PATH))
    if not db_path.is_file():
        logger.warning("backup: %s does not exist, nothing to back up", db_path)
        return None

    backup_dir = Path(backup_dir or os.environ.get("DUCKDB_BACKUP_DIR", "data/backups"))
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = backup_dir / f"warehouse_{timestamp}.duckdb"
    shutil.copy2(db_path, dest)
    logger.info("backup: wrote %s", dest)

    existing = sorted(backup_dir.glob("warehouse_*.duckdb"), key=lambda p: p.name)
    for stale in existing[:-keep_last]:
        stale.unlink()
        logger.info("backup: pruned old backup %s", stale)

    return dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Back up warehouse.duckdb to data/backups/, keeping the last N")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP_LAST, help="Number of backups to retain (default 3)")
    args = parser.parse_args()

    result = backup_warehouse(keep_last=args.keep)
    print(f"backup complete: {result}" if result else "backup skipped: no warehouse.duckdb found")
